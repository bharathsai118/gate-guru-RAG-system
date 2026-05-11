from __future__ import annotations

from .config import AppConfig
from .utils import dedupe_sources, normalize_subject_group, sanitize_visitor_id


NO_CONTEXT_MESSAGE = (
    "I could not find enough relevant information in the available GATE resources or "
    "your uploaded PDFs. Please upload a more relevant PDF or rephrase the question."
)


SYSTEM_PROMPT = """You are GATE Guru, an expert AI tutor for GATE Computer Science and Engineering preparation.

Rules:
1. Answer using the retrieved context.
2. If context is insufficient, clearly say the provided resources do not contain enough information.
3. Do not hallucinate facts, formulas, page numbers, or citations.
4. Keep answers GATE CSE exam-focused.
5. For solve mode, provide step-by-step solution.
6. For revision mode, provide concise notes, formulas, and key points.
7. For quiz mode, generate practice questions only from the retrieved context.
8. Always include source references using filename and page number.
9. Be clear, structured, and student-friendly."""


MODE_INSTRUCTIONS = {
    "ask": """Ask mode:
- Explain the concept clearly.
- Include examples if useful.
- Keep it exam-focused.
- Add sources.""",
    "solve": """Solve mode:
- Identify the concept.
- Give formula or rule if needed.
- Solve step by step.
- Give the final answer.
- Add sources.""",
    "revision": """Revision mode:
- Give short notes.
- Include important formulas.
- Add key definitions.
- Mention common GATE traps.
- Add sources.""",
    "quiz": """Quiz mode:
- Generate 5 practice questions from the retrieved context.
- Include an answer key after the questions.
- Do not generate outside-context questions.
- Add sources.""",
}


class RAGService:
    FORMULA_WARNING = (
        "Formula-heavy PDFs can lose equation layout during text extraction. "
        "Verify formulas against the cited PDF pages."
    )

    def __init__(self, config: AppConfig, vector_store, llm_service, reranker=None, history=None):
        self.config = config
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.reranker = reranker
        self.history = history

    @staticmethod
    def _format_context(chunks: list[dict]) -> str:
        context_parts: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata", {})
            filename = metadata.get("filename", "unknown.pdf")
            page_number = metadata.get("page_number", "?")
            text = chunk.get("text", "")
            context_parts.append(f"[{index}] {filename}, page {page_number}:\n{text}")
        return "\n\n".join(context_parts)

    @staticmethod
    def _sources_from_chunks(chunks: list[dict]) -> list[dict]:
        sources: list[dict] = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            sources.append(
                {
                    "filename": metadata.get("filename", "unknown.pdf"),
                    "page_number": metadata.get("page_number"),
                    "category": metadata.get("category", "Unknown"),
                    "source_type": metadata.get("source_type", "unknown"),
                    "subject_group": metadata.get("subject_group", "Unknown"),
                }
            )
        return dedupe_sources(sources)

    def _build_user_prompt(
        self,
        question: str,
        mode: str,
        chunks: list[dict],
        conversation_context: str = "",
    ) -> str:
        context = self._format_context(chunks)
        instructions = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["ask"])
        conversation = (
            f"\nRecent conversation:\n{conversation_context}\n"
            if conversation_context
            else ""
        )
        return f"""Context:
{context}
{conversation}

Question:
{question}

Mode:
{mode}

Answer instructions:
- Direct answer
- Explanation
- GATE exam insight
- Sources

{instructions}"""

    def _retrieve_context(
        self,
        question: str,
        visitor_id: str,
        subject_group: str,
        include_da_resources: bool,
    ) -> tuple[list[dict], list[str] | None]:
        warnings = [self.FORMULA_WARNING]
        candidate_k = max(self.config.retrieval_candidate_k, self.config.top_k)
        retrieved = self.vector_store.query(
            question=question,
            visitor_id=visitor_id,
            subject_group=subject_group,
            include_da_resources=include_da_resources,
            top_k=candidate_k,
        )
        if self.reranker:
            retrieved = self.reranker.rerank(question, retrieved, self.config.max_context_chunks)
            if getattr(self.reranker, "last_error", None):
                warnings.append(f"Reranker unavailable; used vector similarity only: {self.reranker.last_error}")
        else:
            retrieved = retrieved[: self.config.max_context_chunks]
        return retrieved, warnings

    def answer_question(
        self,
        visitor_id: str,
        question: str,
        mode: str = "ask",
        subject_group: str = "All",
        include_da_resources: bool = False,
    ) -> dict:
        question = (question or "").strip()
        mode = mode if mode in MODE_INSTRUCTIONS else "ask"
        visitor_id = sanitize_visitor_id(visitor_id)
        subject_group = normalize_subject_group(subject_group)

        if not question:
            return {
                "answer": "Please enter a question.",
                "sources": [],
                "retrieved_count": 0,
            }

        try:
            retrieved, warnings = self._retrieve_context(
                question=question,
                visitor_id=visitor_id,
                subject_group=subject_group,
                include_da_resources=include_da_resources,
            )
        except Exception as exc:
            return {
                "answer": f"The vector database could not complete retrieval: {exc}",
                "sources": [],
                "retrieved_count": 0,
                "warnings": [],
            }

        if not retrieved:
            return {
                "answer": NO_CONTEXT_MESSAGE,
                "sources": [],
                "retrieved_count": 0,
                "warnings": warnings,
            }

        context_chunks = retrieved[: self.config.max_context_chunks]
        conversation_context = self.history.get_context(visitor_id) if self.history else ""
        prompt = self._build_user_prompt(question, mode, context_chunks, conversation_context)
        answer = self.llm_service.generate(SYSTEM_PROMPT, prompt)
        if self.history:
            self.history.add_turn(visitor_id, question, answer)
        return {
            "answer": answer,
            "sources": self._sources_from_chunks(context_chunks),
            "retrieved_count": len(context_chunks),
            "warnings": warnings,
        }

    def stream_answer_question(
        self,
        visitor_id: str,
        question: str,
        mode: str = "ask",
        subject_group: str = "All",
        include_da_resources: bool = False,
    ):
        question = (question or "").strip()
        mode = mode if mode in MODE_INSTRUCTIONS else "ask"
        visitor_id = sanitize_visitor_id(visitor_id)
        subject_group = normalize_subject_group(subject_group)

        if not question:
            yield {"event": "error", "data": {"message": "Question cannot be empty."}}
            return

        try:
            retrieved, warnings = self._retrieve_context(
                question=question,
                visitor_id=visitor_id,
                subject_group=subject_group,
                include_da_resources=include_da_resources,
            )
        except Exception as exc:
            yield {
                "event": "error",
                "data": {"message": f"The vector database could not complete retrieval: {exc}"},
            }
            return

        if not retrieved:
            yield {
                "event": "done",
                "data": {
                    "answer": NO_CONTEXT_MESSAGE,
                    "sources": [],
                    "retrieved_count": 0,
                    "warnings": warnings,
                },
            }
            return

        context_chunks = retrieved[: self.config.max_context_chunks]
        sources = self._sources_from_chunks(context_chunks)
        yield {
            "event": "meta",
            "data": {
                "sources": sources,
                "retrieved_count": len(context_chunks),
                "warnings": warnings,
            },
        }

        conversation_context = self.history.get_context(visitor_id) if self.history else ""
        prompt = self._build_user_prompt(question, mode, context_chunks, conversation_context)
        answer_parts: list[str] = []
        for token in self.llm_service.stream_generate(SYSTEM_PROMPT, prompt):
            answer_parts.append(token)
            yield {"event": "token", "data": {"text": token}}

        answer = "".join(answer_parts).strip()
        if self.history:
            self.history.add_turn(visitor_id, question, answer)

        yield {
            "event": "done",
            "data": {
                "answer": answer,
                "sources": sources,
                "retrieved_count": len(context_chunks),
                "warnings": warnings,
            },
        }
