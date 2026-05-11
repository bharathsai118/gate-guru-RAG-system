from __future__ import annotations

from .config import AppConfig


class Reranker:
    def __init__(self, config: AppConfig):
        self.config = config
        self._model = None
        self.last_error: str | None = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.config.reranker_model_name)
        return self._model

    def rerank(self, question: str, chunks: list[dict], top_n: int) -> list[dict]:
        if not self.config.reranker_enabled or not chunks:
            return chunks[:top_n]

        try:
            model = self._load()
            pairs = [(question, chunk.get("text", "")) for chunk in chunks]
            scores = model.predict(pairs)
            ranked = []
            for chunk, score in zip(chunks, scores):
                updated = dict(chunk)
                updated["rerank_score"] = float(score)
                ranked.append(updated)
            self.last_error = None
            return sorted(ranked, key=lambda item: item["rerank_score"], reverse=True)[:top_n]
        except Exception as exc:
            self.last_error = str(exc)
            return chunks[:top_n]

