from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


PRELOADED_RESOURCES = [
    {
        "filename": "volume1.pdf",
        "relative_path": "Mathematics_and_Aptitude/volume1.pdf",
        "category": "Mathematics and Aptitude",
        "subject_group": "Mathematics_and_Aptitude",
        "subjects": [
            "Discrete Mathematics",
            "Engineering Mathematics",
            "General Aptitude",
        ],
        "is_default_cse_resource": True,
    },
    {
        "filename": "volume2.pdf",
        "relative_path": "Core_CSE/volume2.pdf",
        "category": "Core GATE CSE",
        "subject_group": "Core_CSE",
        "subjects": [
            "Data Structures",
            "Algorithms",
            "Operating Systems",
            "DBMS",
            "Computer Networks",
            "Theory of Computation",
            "Compiler Design",
            "Digital Logic",
            "Computer Organization and Architecture",
            "C Programming",
        ],
        "is_default_cse_resource": True,
    },
    {
        "filename": "da.pdf",
        "relative_path": "Optional_DA_AI_ML/da.pdf",
        "category": "Optional DA / AI / ML",
        "subject_group": "Optional_DA_AI_ML",
        "subjects": [
            "Artificial Intelligence",
            "Machine Learning",
            "Probability",
            "Linear Algebra",
            "Python",
            "Databases",
            "Algorithms",
        ],
        "is_default_cse_resource": False,
    },
]

DEFAULT_SUBJECT_GROUPS = {"Mathematics_and_Aptitude", "Core_CSE"}
DA_SUBJECT_GROUPS = {"Optional_DA_AI_ML"}
DA_RELATED_SUBJECTS = {
    "AI",
    "Artificial_Intelligence",
    "ML",
    "Machine_Learning",
    "Probability",
    "Linear_Algebra",
    "Python",
    "DA",
    "Data_Analytics",
    "Optional_DA_AI_ML",
}

STUDY_MODES = [
    {"value": "ask", "label": "Ask"},
    {"value": "solve", "label": "Solve"},
    {"value": "revision", "label": "Revision"},
    {"value": "quiz", "label": "Quiz"},
]

SUBJECT_GROUPS = [
    {"value": "All", "label": "All GATE CSE"},
    {"value": "Mathematics_and_Aptitude", "label": "Mathematics and Aptitude"},
    {"value": "Core_CSE", "label": "Core GATE CSE"},
    {"value": "Optional_DA_AI_ML", "label": "Optional DA / AI / ML"},
    {"value": "General", "label": "General / Uploaded Notes"},
]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _resolve_path(value: str | None, default: Path) -> Path:
    raw = value or str(default)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path = field(default_factory=lambda: BASE_DIR)
    chroma_db_dir: Path = field(
        default_factory=lambda: _resolve_path(os.getenv("CHROMA_DB_DIR"), BASE_DIR / "chroma_db")
    )
    preloaded_dir: Path = field(
        default_factory=lambda: _resolve_path(
            os.getenv("PRELOADED_DIR"), BASE_DIR / "data" / "preloaded_gate_cse"
        )
    )
    upload_dir: Path = field(
        default_factory=lambda: _resolve_path(os.getenv("UPLOAD_DIR"), BASE_DIR / "uploads")
    )
    auto_index_preloaded: bool = field(
        default_factory=lambda: _env_bool("AUTO_INDEX_PRELOADED", True)
    )
    dense_model_name: str = field(
        default_factory=lambda: os.getenv(
            "DENSE_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    llm_model_name: str = field(
        default_factory=lambda: os.getenv(
            "LLM_MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        )
    )
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "local"))
    llm_api_base_url: str = field(default_factory=lambda: os.getenv("LLM_API_BASE_URL", ""))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_api_model_name: str = field(
        default_factory=lambda: os.getenv(
            "LLM_API_MODEL_NAME",
            os.getenv("LLM_MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
        )
    )
    llm_api_timeout_seconds: int = field(
        default_factory=lambda: _env_int("LLM_API_TIMEOUT_SECONDS", 120)
    )
    llm_health_check_load: bool = field(
        default_factory=lambda: _env_bool("LLM_HEALTH_CHECK_LOAD", False)
    )
    llm_trust_remote_code: bool = field(
        default_factory=lambda: _env_bool("LLM_TRUST_REMOTE_CODE", True)
    )
    llm_strip_thinking: bool = field(
        default_factory=lambda: _env_bool("LLM_STRIP_THINKING", True)
    )
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    retrieval_candidate_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_CANDIDATE_K", 10))
    reranker_enabled: bool = field(default_factory=lambda: _env_bool("RERANKER_ENABLED", True))
    reranker_model_name: str = field(
        default_factory=lambda: os.getenv(
            "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )
    max_context_chunks: int = field(default_factory=lambda: _env_int("MAX_CONTEXT_CHUNKS", 5))
    max_new_tokens: int = field(default_factory=lambda: _env_int("MAX_NEW_TOKENS", 512))
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 900))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 150))
    token_chunking_enabled: bool = field(
        default_factory=lambda: _env_bool("TOKEN_CHUNKING_ENABLED", True)
    )
    chunk_tokenizer_model_name: str = field(
        default_factory=lambda: os.getenv(
            "CHUNK_TOKENIZER_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    chunk_size_tokens: int = field(default_factory=lambda: _env_int("CHUNK_SIZE_TOKENS", 256))
    chunk_overlap_tokens: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP_TOKENS", 40))
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 30))
    collection_name: str = "gate_guru_rag"
    max_prompt_tokens: int = field(default_factory=lambda: _env_int("MAX_PROMPT_TOKENS", 4096))
    history_enabled: bool = field(default_factory=lambda: _env_bool("HISTORY_ENABLED", True))
    history_max_turns: int = field(default_factory=lambda: _env_int("HISTORY_MAX_TURNS", 4))
    history_db_file: Path = field(
        default_factory=lambda: _resolve_path(
            os.getenv("HISTORY_DB_FILE"), BASE_DIR / "history" / "conversation_history.sqlite3"
        )
    )
    ask_rate_limit: str = field(default_factory=lambda: os.getenv("ASK_RATE_LIMIT", "20 per hour"))
    upload_rate_limit: str = field(
        default_factory=lambda: os.getenv("UPLOAD_RATE_LIMIT", "8 per hour")
    )
    index_rate_limit: str = field(default_factory=lambda: os.getenv("INDEX_RATE_LIMIT", "3 per hour"))
    feedback_log_file: Path = field(
        default_factory=lambda: _resolve_path(
            os.getenv("FEEDBACK_LOG_FILE"), BASE_DIR / "feedback" / "feedback.jsonl"
        )
    )
    cleanup_enabled: bool = field(default_factory=lambda: _env_bool("CLEANUP_ENABLED", True))
    upload_ttl_hours: float = field(default_factory=lambda: _env_float("UPLOAD_TTL_HOURS", 24.0))
    cleanup_interval_minutes: int = field(
        default_factory=lambda: _env_int("CLEANUP_INTERVAL_MINUTES", 60)
    )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def preloaded_pdf_path(self, resource: dict) -> Path:
        return self.preloaded_dir / resource["relative_path"]

    def subject_payload(self) -> dict:
        return {
            "subject_groups": SUBJECT_GROUPS,
            "study_modes": STUDY_MODES,
            "preloaded_resources": PRELOADED_RESOURCES,
            "default_subject_groups": sorted(DEFAULT_SUBJECT_GROUPS),
            "da_subject_groups": sorted(DA_SUBJECT_GROUPS),
        }
