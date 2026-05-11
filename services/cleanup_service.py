from __future__ import annotations

import time
from pathlib import Path

from .config import AppConfig


def cleanup_old_uploads(config: AppConfig, vector_store) -> dict:
    cutoff = time.time() - (config.upload_ttl_hours * 3600)
    deleted_files = 0
    deleted_chunks = 0

    if not config.upload_dir.exists():
        return {"deleted_files": 0, "deleted_chunks": 0}

    for visitor_dir in config.upload_dir.iterdir():
        if not visitor_dir.is_dir():
            continue
        visitor_id = visitor_dir.name
        for path in visitor_dir.glob("*.pdf"):
            try:
                if path.stat().st_mtime > cutoff:
                    continue
                deleted_chunks += vector_store.delete_user_file(visitor_id, path.name)
                path.unlink(missing_ok=True)
                deleted_files += 1
            except Exception:
                continue
        _remove_empty_dir(visitor_dir)

    return {"deleted_files": deleted_files, "deleted_chunks": deleted_chunks}


def _remove_empty_dir(path: Path) -> None:
    try:
        if path.exists() and not any(path.iterdir()):
            path.rmdir()
    except Exception:
        pass


def start_cleanup_scheduler(config: AppConfig, vector_store, logger=None):
    if not config.cleanup_enabled:
        return None

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except Exception as exc:
        if logger:
            logger.warning("Upload cleanup scheduler unavailable: %s", exc)
        return None

    scheduler = BackgroundScheduler(daemon=True)

    def job():
        result = cleanup_old_uploads(config, vector_store)
        if logger and (result["deleted_files"] or result["deleted_chunks"]):
            logger.info("Upload cleanup completed: %s", result)

    scheduler.add_job(
        job,
        "interval",
        minutes=max(1, config.cleanup_interval_minutes),
        id="gate_guru_upload_cleanup",
        replace_existing=True,
    )
    scheduler.start()
    return scheduler

