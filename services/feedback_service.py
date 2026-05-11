from __future__ import annotations

import json
from datetime import datetime, timezone
from threading import Lock

from .config import AppConfig
from .utils import sanitize_visitor_id


class FeedbackService:
    def __init__(self, config: AppConfig):
        self.config = config
        self._lock = Lock()
        self.config.feedback_log_file.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        visitor_id: str,
        rating: str,
        question: str,
        answer: str,
        sources: list[dict],
    ) -> dict:
        rating = (rating or "").strip().lower()
        if rating not in {"up", "down"}:
            raise ValueError("Feedback rating must be 'up' or 'down'.")

        record = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "visitor_id": sanitize_visitor_id(visitor_id),
            "rating": rating,
            "question": (question or "").strip()[:1200],
            "answer": (answer or "").strip()[:4000],
            "sources": sources or [],
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self.config.feedback_log_file.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
        return {"status": "ok", "rating": rating}

