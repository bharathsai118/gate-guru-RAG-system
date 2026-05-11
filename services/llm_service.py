from __future__ import annotations

import json
import re
from threading import Thread

from .config import AppConfig


class LLMService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        self.load_error: str | None = None

    @property
    def provider(self) -> str:
        return (self.config.llm_provider or "local").strip().lower()

    def _load(self) -> bool:
        if self.model is not None and self.tokenizer is not None:
            return True
        if self.load_error:
            return False

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.llm_model_name,
                trust_remote_code=self.config.llm_trust_remote_code,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.llm_model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=self.config.llm_trust_remote_code,
            )
            self.model.to(self.device)
            self.model.eval()
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            return True
        except Exception as exc:
            error_text = str(exc)
            memory_terms = ("out of memory", "cuda", "cannot allocate", "not enough memory")
            if any(term in error_text.lower() for term in memory_terms):
                self.load_error = (
                    f"{error_text}. The configured model may be too large for the available "
                    "RAM/VRAM. Try a smaller DeepSeek-R1 distilled checkpoint such as "
                    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B, or use a larger GPU Space."
                )
            else:
                self.load_error = error_text
            return False

    def _messages(self, system_prompt: str, user_prompt: str) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _strip_thinking(self, answer: str) -> str:
        if not self.config.llm_strip_thinking:
            return answer
        return re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    def readiness(self, load_model: bool = False) -> dict:
        if self.provider in {"api", "openai_compatible", "openai-compatible"}:
            missing = [
                name
                for name, value in {
                    "LLM_API_BASE_URL": self.config.llm_api_base_url,
                    "LLM_API_KEY": self.config.llm_api_key,
                    "LLM_API_MODEL_NAME": self.config.llm_api_model_name,
                }.items()
                if not value
            ]
            return {
                "provider": self.provider,
                "model": self.config.llm_api_model_name,
                "status": "error" if missing else "configured",
                "missing": missing,
            }

        if self.load_error:
            return {
                "provider": "local",
                "model": self.config.llm_model_name,
                "status": "error",
                "error": self.load_error,
            }
        if self.model is not None and self.tokenizer is not None:
            return {
                "provider": "local",
                "model": self.config.llm_model_name,
                "status": "loaded",
                "device": self.device,
            }
        if load_model:
            loaded = self._load()
            return {
                "provider": "local",
                "model": self.config.llm_model_name,
                "status": "loaded" if loaded else "error",
                "device": self.device,
                "error": self.load_error,
            }
        return {
            "provider": "local",
            "model": self.config.llm_model_name,
            "status": "not_loaded_lazy",
            "note": "Model loads on the first question. Set LLM_HEALTH_CHECK_LOAD=true to load-check it in /health.",
        }

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "".join(self.stream_generate(system_prompt, user_prompt)).strip()

    def stream_generate(self, system_prompt: str, user_prompt: str):
        if self.provider in {"api", "openai_compatible", "openai-compatible"}:
            yield from self._stream_openai_compatible(system_prompt, user_prompt)
            return

        if not self._load():
            yield (
                "The local Hugging Face model could not be loaded. "
                f"Configured model: {self.config.llm_model_name}. "
                f"Error: {self.load_error}"
            )
            return

        try:
            import torch
            from transformers import TextIteratorStreamer

            messages = self._messages(system_prompt, user_prompt)
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_prompt_tokens,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            generation_kwargs = {
                "max_new_tokens": self.config.max_new_tokens,
                "do_sample": False,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            def run_generation():
                try:
                    with torch.no_grad():
                        self.model.generate(**inputs, **generation_kwargs, streamer=streamer)
                except Exception as exc:
                    message = f"\n\nThe local Hugging Face model failed during generation: {exc}"
                    try:
                        streamer.on_finalized_text(message, stream_end=True)
                    except Exception:
                        try:
                            streamer.end()
                        except Exception:
                            pass

            thread = Thread(target=run_generation, daemon=True)
            thread.start()
            yielded_any = False
            for token in self._filter_thinking_stream(streamer):
                yielded_any = True
                yield token
            thread.join(timeout=1)
            if not yielded_any:
                yield "I could not generate an answer from the retrieved context."
        except Exception as exc:
            yield f"The local Hugging Face model failed during generation: {exc}"

    def _stream_openai_compatible(self, system_prompt: str, user_prompt: str):
        if not self.config.llm_api_key or not self.config.llm_api_base_url:
            yield (
                "API LLM provider is enabled, but LLM_API_KEY or LLM_API_BASE_URL is missing. "
                "Set those environment variables or use LLM_PROVIDER=local."
            )
            return

        try:
            import requests

            url = self.config.llm_api_base_url.rstrip("/") + "/chat/completions"
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.config.llm_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.llm_api_model_name,
                    "messages": self._messages(system_prompt, user_prompt),
                    "max_tokens": self.config.max_new_tokens,
                    "temperature": 0,
                    "stream": True,
                },
                stream=True,
                timeout=self.config.llm_api_timeout_seconds,
            )
            if response.status_code >= 400:
                yield f"API LLM request failed with HTTP {response.status_code}: {response.text[:800]}"
                return

            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                payload = line.removeprefix("data:").strip()
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                    delta = data["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
                except Exception:
                    continue
        except Exception as exc:
            yield f"API LLM generation failed: {exc}"

    def _filter_thinking_stream(self, chunks):
        if not self.config.llm_strip_thinking:
            yield from chunks
            return

        start_tag = "<think>"
        end_tag = "</think>"
        start_keep = len(start_tag) - 1
        end_keep = len(end_tag) - 1
        buffer = ""
        in_think = False

        for chunk in chunks:
            buffer += chunk
            while buffer:
                if in_think:
                    end = buffer.find(end_tag)
                    if end != -1:
                        buffer = buffer[end + len(end_tag) :]
                        in_think = False
                        continue
                    if len(buffer) > end_keep:
                        buffer = buffer[-end_keep:]
                    break

                start = buffer.find(start_tag)
                if start != -1:
                    if start > 0:
                        yield buffer[:start]
                    buffer = buffer[start + len(start_tag) :]
                    in_think = True
                    continue

                if len(buffer) > start_keep:
                    safe_text = buffer[:-start_keep]
                    if safe_text:
                        yield safe_text
                    buffer = buffer[-start_keep:]
                break

        if buffer and not in_think:
            yield buffer
