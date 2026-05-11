from __future__ import annotations

import sqlite3
import time
from threading import Lock

from .config import AppConfig
from .utils import sanitize_visitor_id


class ConversationHistory:
    def __init__(self, config: AppConfig):
        self.config = config
        self._lock = Lock()
        self.config.history_db_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.config.history_db_file),
            timeout=15,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visitor_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_visitor_created
                ON conversation_turns(visitor_id, created_at)
                """
            )

    def get_context(self, visitor_id: str) -> str:
        if not self.config.history_enabled:
            return ""

        visitor_id = sanitize_visitor_id(visitor_id)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT question, answer
                FROM conversation_turns
                WHERE visitor_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (visitor_id, self.config.history_max_turns),
            ).fetchall()

        turns = list(reversed(rows))
        if not turns:
            return ""

        lines: list[str] = []
        for turn in turns:
            lines.append(f"Student: {turn['question']}")
            lines.append(f"GATE Guru: {turn['answer']}")
        return "\n".join(lines)

    def add_turn(self, visitor_id: str, question: str, answer: str) -> None:
        if not self.config.history_enabled:
            return

        visitor_id = sanitize_visitor_id(visitor_id)
        clean_answer = (answer or "").strip()
        if len(clean_answer) > 1200:
            clean_answer = clean_answer[:1200] + "..."

        clean_question = (question or "").strip()[:600]
        if not clean_question or not clean_answer:
            return

        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversation_turns(visitor_id, question, answer, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (visitor_id, clean_question, clean_answer, time.time()),
            )
            connection.execute(
                """
                DELETE FROM conversation_turns
                WHERE visitor_id = ?
                AND id NOT IN (
                    SELECT id
                    FROM conversation_turns
                    WHERE visitor_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                """,
                (visitor_id, visitor_id, self.config.history_max_turns),
            )
