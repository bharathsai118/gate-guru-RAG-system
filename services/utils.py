from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable


VISITOR_ID_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]")


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def sanitize_visitor_id(visitor_id: str | None) -> str:
    value = (visitor_id or "").strip()
    value = VISITOR_ID_PATTERN.sub("_", value)
    return value[:96] or "anonymous"


def normalize_subject_group(subject_group: str | None) -> str:
    value = (subject_group or "All").strip()
    return value or "All"


def stable_chunk_id(
    source_type: str,
    visitor_id: str | None,
    filename: str,
    page_number: int,
    chunk_index: int,
    text: str,
) -> str:
    visitor = visitor_id or "public"
    payload = "|".join(
        [
            source_type,
            visitor,
            filename,
            str(page_number),
            str(chunk_index),
            hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest(),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()


def dedupe_sources(sources: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    unique: list[dict] = []
    for source in sources:
        key = (
            source.get("filename"),
            source.get("page_number"),
            source.get("category"),
            source.get("source_type"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique

