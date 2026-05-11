from __future__ import annotations

from typing import Any

from .config import DA_RELATED_SUBJECTS, DEFAULT_SUBJECT_GROUPS, AppConfig
from .utils import normalize_subject_group, sanitize_visitor_id, stable_chunk_id


class VectorStore:
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = None
        self.collection = None
        self._embedding_model = None
        self.init_error: str | None = None

        try:
            import chromadb

            self.config.chroma_db_dir.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.config.chroma_db_dir))
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name
            )
        except Exception as exc:
            self.init_error = str(exc)

    @property
    def available(self) -> bool:
        return self.collection is not None

    def _require_collection(self) -> None:
        if not self.available:
            raise RuntimeError(f"ChromaDB is unavailable: {self.init_error}")

    def _get_embedding_model(self):
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. Run pip install -r requirements.txt."
                ) from exc
            self._embedding_model = SentenceTransformer(self.config.dense_model_name)
        return self._embedding_model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        model = self._get_embedding_model()
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    @staticmethod
    def _and_filter(conditions: dict[str, Any]) -> dict[str, Any]:
        parts = [{key: {"$eq": value}} for key, value in conditions.items()]
        if not parts:
            return {}
        if len(parts) == 1:
            return parts[0]
        return {"$and": parts}

    def add_chunks(self, chunks: list[dict], base_metadata: dict) -> int:
        self._require_collection()
        if not chunks:
            return 0

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for chunk in chunks:
            metadata = dict(base_metadata)
            metadata["page_number"] = int(chunk["page_number"])
            metadata["chunk_index"] = int(chunk["chunk_index"])
            metadata.setdefault("visitor_id", "")

            chunk_id = stable_chunk_id(
                metadata["source_type"],
                metadata.get("visitor_id"),
                metadata["filename"],
                metadata["page_number"],
                metadata["chunk_index"],
                chunk["text"],
            )
            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append(metadata)

        batch_size = 64
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            embeddings = self._embed(documents[start:end])
            self.collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=embeddings,
            )

        return len(ids)

    def is_file_indexed(
        self, filename: str, source_type: str, visitor_id: str | None = None
    ) -> bool:
        self._require_collection()
        conditions: dict[str, Any] = {"filename": filename, "source_type": source_type}
        if source_type == "user_upload":
            conditions["visitor_id"] = sanitize_visitor_id(visitor_id)
        where = self._and_filter(conditions)
        result = self.collection.get(where=where, limit=1)
        return bool(result.get("ids"))

    def delete_user_file(self, visitor_id: str, filename: str) -> int:
        self._require_collection()
        where = self._and_filter(
            {
                "source_type": "user_upload",
                "visitor_id": sanitize_visitor_id(visitor_id),
                "filename": filename,
            }
        )
        result = self.collection.get(where=where)
        ids = result.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def count_documents(self) -> int:
        self._require_collection()
        return int(self.collection.count())

    def _query_once(
        self,
        query_embedding: list[float],
        where: dict[str, Any],
        n_results: int,
    ) -> list[dict]:
        if not where:
            return []
        try:
            response = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: list[dict] = []
        for idx, doc_id in enumerate(ids):
            results.append(
                {
                    "id": doc_id,
                    "text": documents[idx],
                    "metadata": metadatas[idx],
                    "distance": float(distances[idx]) if idx < len(distances) else 1.0,
                }
            )
        return results

    def query(
        self,
        question: str,
        visitor_id: str,
        subject_group: str = "All",
        include_da_resources: bool = False,
        top_k: int | None = None,
    ) -> list[dict]:
        self._require_collection()
        top_k = top_k or self.config.top_k
        subject_group = normalize_subject_group(subject_group)
        visitor_id = sanitize_visitor_id(visitor_id)
        include_da = include_da_resources or subject_group in DA_RELATED_SUBJECTS
        query_embedding = self._embed([question])[0]

        filters: list[dict[str, Any]] = []

        if subject_group == "All" or subject_group in DEFAULT_SUBJECT_GROUPS:
            preloaded_conditions: dict[str, Any] = {
                "source_type": "preloaded",
                "is_default_cse_resource": True,
            }
            if subject_group in DEFAULT_SUBJECT_GROUPS:
                preloaded_conditions["subject_group"] = subject_group
            filters.append(self._and_filter(preloaded_conditions))

        if visitor_id:
            upload_conditions: dict[str, Any] = {
                "source_type": "user_upload",
                "visitor_id": visitor_id,
            }
            if subject_group != "All":
                upload_conditions["subject_group"] = subject_group
            filters.append(self._and_filter(upload_conditions))

        if include_da:
            filters.append(
                self._and_filter(
                    {
                        "source_type": "preloaded",
                        "subject_group": "Optional_DA_AI_ML",
                    }
                )
            )

        merged: dict[str, dict] = {}
        for where in filters:
            for result in self._query_once(query_embedding, where, top_k):
                existing = merged.get(result["id"])
                if existing is None or result["distance"] < existing["distance"]:
                    merged[result["id"]] = result

        return sorted(merged.values(), key=lambda item: item["distance"])[:top_k]

