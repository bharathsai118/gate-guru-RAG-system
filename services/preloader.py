from __future__ import annotations

import shutil
from pathlib import Path

from .chunking_service import chunk_pages, chunk_pages_token_aware
from .config import PRELOADED_RESOURCES, AppConfig
from .pdf_service import extract_pdf_pages


class Preloader:
    def __init__(self, config: AppConfig, vector_store):
        self.config = config
        self.vector_store = vector_store

    def _copy_from_gate_resource_if_available(self) -> list[str]:
        copied: list[str] = []
        candidates = [
            self.config.base_dir / "GATE_RESOURCE",
            self.config.base_dir.parent / "GATE_RESOURCE",
        ]
        source_dir = next((path for path in candidates if path.exists()), None)
        if source_dir is None:
            return copied

        for resource in PRELOADED_RESOURCES:
            source = source_dir / resource["filename"]
            target = self.config.preloaded_pdf_path(resource)
            if source.exists() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied.append(str(target))
        return copied

    def index_preloaded(self, force: bool = False) -> dict:
        self.config.preloaded_dir.mkdir(parents=True, exist_ok=True)
        copied = self._copy_from_gate_resource_if_available()

        report = {
            "status": "ok",
            "indexed_chunks": 0,
            "indexed_files": [],
            "skipped_files": [],
            "missing_files": [],
            "errors": [],
            "copied_files": copied,
        }

        for resource in PRELOADED_RESOURCES:
            pdf_path = self.config.preloaded_pdf_path(resource)
            filename = resource["filename"]

            if not pdf_path.exists():
                report["missing_files"].append(str(pdf_path))
                continue

            try:
                if not force and self.vector_store.is_file_indexed(filename, "preloaded"):
                    report["skipped_files"].append(filename)
                    continue

                pages = extract_pdf_pages(pdf_path)
                if self.config.token_chunking_enabled:
                    chunks = chunk_pages_token_aware(
                        pages,
                        tokenizer_name=self.config.chunk_tokenizer_model_name,
                        chunk_size_tokens=self.config.chunk_size_tokens,
                        chunk_overlap_tokens=self.config.chunk_overlap_tokens,
                        fallback_chunk_size=self.config.chunk_size,
                        fallback_chunk_overlap=self.config.chunk_overlap,
                    )
                else:
                    chunks = chunk_pages(
                        pages,
                        chunk_size=self.config.chunk_size,
                        chunk_overlap=self.config.chunk_overlap,
                    )
                if not chunks:
                    report["errors"].append({"filename": filename, "error": "No chunks created."})
                    continue

                metadata = {
                    "source_type": "preloaded",
                    "visitor_id": "",
                    "filename": filename,
                    "category": resource["category"],
                    "subject_group": resource["subject_group"],
                    "is_default_cse_resource": resource["is_default_cse_resource"],
                }
                indexed = self.vector_store.add_chunks(chunks, metadata)
                report["indexed_chunks"] += indexed
                report["indexed_files"].append(filename)
            except Exception as exc:
                report["errors"].append({"filename": filename, "error": str(exc)})

        return report
