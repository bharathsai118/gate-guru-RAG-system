from __future__ import annotations

from functools import lru_cache
import re


@lru_cache(maxsize=4)
def _load_tokenizer(tokenizer_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_name)


def _split_long_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


def _text_units(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) > 1:
        return paragraphs
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    return sentences or [text.strip()]


def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    current = ""

    for unit in _text_units(text):
        if len(unit) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_long_text(unit, chunk_size, chunk_overlap))
            continue

        separator = "\n\n" if current else ""
        candidate = f"{current}{separator}{unit}".strip()
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            overlap = current[-chunk_overlap:].strip() if chunk_overlap > 0 else ""
            current = f"{overlap}\n\n{unit}".strip() if overlap else unit
        else:
            current = unit

    if current:
        chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk]


def split_text_into_token_chunks(
    text: str,
    tokenizer_name: str,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> list[str]:
    text = text.strip()
    if not text:
        return []

    tokenizer = _load_tokenizer(tokenizer_name)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []
    if len(token_ids) <= chunk_size_tokens:
        return [text]

    chunks: list[str] = []
    step = max(1, chunk_size_tokens - chunk_overlap_tokens)
    for start in range(0, len(token_ids), step):
        chunk_ids = token_ids[start : start + chunk_size_tokens]
        chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size_tokens >= len(token_ids):
            break
    return chunks


def chunk_pages(pages: list[dict], chunk_size: int = 900, chunk_overlap: int = 150) -> list[dict]:
    chunks: list[dict] = []
    chunk_index = 0
    for page in pages:
        for text in split_text_into_chunks(page["text"], chunk_size, chunk_overlap):
            chunks.append(
                {
                    "text": text,
                    "page_number": int(page["page_number"]),
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1
    return chunks


def chunk_pages_token_aware(
    pages: list[dict],
    tokenizer_name: str,
    chunk_size_tokens: int = 256,
    chunk_overlap_tokens: int = 40,
    fallback_chunk_size: int = 900,
    fallback_chunk_overlap: int = 150,
) -> list[dict]:
    try:
        chunks: list[dict] = []
        chunk_index = 0
        for page in pages:
            for text in split_text_into_token_chunks(
                page["text"],
                tokenizer_name=tokenizer_name,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
            ):
                chunks.append(
                    {
                        "text": text,
                        "page_number": int(page["page_number"]),
                        "chunk_index": chunk_index,
                    }
                )
                chunk_index += 1
        return chunks
    except Exception:
        return chunk_pages(
            pages,
            chunk_size=fallback_chunk_size,
            chunk_overlap=fallback_chunk_overlap,
        )
