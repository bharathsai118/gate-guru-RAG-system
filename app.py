from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from services.chunking_service import chunk_pages, chunk_pages_token_aware
from services.cleanup_service import start_cleanup_scheduler
from services.config import AppConfig
from services.feedback_service import FeedbackService
from services.history_service import ConversationHistory
from services.llm_service import LLMService
from services.pdf_service import extract_pdf_pages
from services.preloader import Preloader
from services.rag_service import RAGService
from services.reranker import Reranker
from services.utils import ensure_directories, normalize_subject_group, sanitize_visitor_id
from services.vector_store import VectorStore


load_dotenv()


PDF_MIME_TYPES = {"application/pdf", "application/x-pdf", "application/octet-stream", ""}


def _visitor_rate_key() -> str:
    visitor_id = request.headers.get("X-Visitor-Id") or request.args.get("visitor_id") or ""
    remote_addr = request.remote_addr or "anonymous"
    if visitor_id:
        return f"{remote_addr}:{sanitize_visitor_id(visitor_id)}"
    return sanitize_visitor_id(remote_addr)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _build_limiter(app: Flask):
    try:
        from flask_limiter import Limiter

        config: AppConfig = app.extensions["gate_guru_config"]
        return Limiter(
            key_func=_visitor_rate_key,
            app=app,
            storage_uri=config.rate_limit_storage_uri,
        )
    except Exception as exc:
        app.logger.warning("Rate limiting disabled: %s", exc)
        return None


def _limit(limiter, rule: str):
    if limiter is None:
        return lambda fn: fn
    return limiter.limit(rule)


def _chunk_pages_for_config(pages: list[dict], config: AppConfig) -> list[dict]:
    if config.token_chunking_enabled:
        return chunk_pages_token_aware(
            pages,
            tokenizer_name=config.chunk_tokenizer_model_name,
            chunk_size_tokens=config.chunk_size_tokens,
            chunk_overlap_tokens=config.chunk_overlap_tokens,
            fallback_chunk_size=config.chunk_size,
            fallback_chunk_overlap=config.chunk_overlap,
        )
    return chunk_pages(
        pages,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )


def _is_pdf_upload(file) -> tuple[bool, str]:
    mimetype = (file.mimetype or "").lower()
    if mimetype not in PDF_MIME_TYPES:
        return False, f"Invalid MIME type '{file.mimetype}'. Please upload a PDF."

    try:
        head = file.stream.read(5)
        file.stream.seek(0)
    except Exception:
        return False, "Could not inspect the uploaded file. Please upload a valid PDF."

    if head != b"%PDF-":
        return False, "Invalid PDF file. The file header does not match PDF magic bytes."

    return True, ""


def create_app() -> Flask:
    config = AppConfig()
    ensure_directories(
        [
            config.upload_dir,
            config.chroma_db_dir,
            config.preloaded_dir,
            config.feedback_log_file.parent,
            config.history_db_file.parent,
        ]
    )

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.max_upload_bytes
    app.extensions["gate_guru_config"] = config
    limiter = _build_limiter(app)

    vector_store = VectorStore(config)
    llm_service = LLMService(config)
    reranker = Reranker(config)
    history_service = ConversationHistory(config)
    feedback_service = FeedbackService(config)
    rag_service = RAGService(config, vector_store, llm_service, reranker, history_service)
    preloader = Preloader(config, vector_store)
    cleanup_scheduler = start_cleanup_scheduler(config, vector_store, app.logger)

    app.extensions["gate_guru_vector_store"] = vector_store
    app.extensions["gate_guru_preloader"] = preloader
    app.extensions["gate_guru_cleanup_scheduler"] = cleanup_scheduler

    if config.auto_index_preloaded:
        try:
            app.logger.info("AUTO_INDEX_PRELOADED=true; indexing preloaded resources if present.")
            preloader.index_preloaded()
        except Exception as exc:
            app.logger.warning("Preloaded indexing did not complete: %s", exc)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_upload(_: RequestEntityTooLarge):
        return (
            jsonify(
                {
                    "error": f"Upload too large. Maximum allowed size is {config.max_upload_mb} MB."
                }
            ),
            413,
        )

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        vector_status = {
            "backend": "chroma",
            "status": "ok" if vector_store.available else "error",
            "error": vector_store.init_error,
        }
        if vector_store.available:
            try:
                vector_status["document_count"] = vector_store.count_documents()
            except Exception as exc:
                vector_status["status"] = "error"
                vector_status["error"] = str(exc)

        llm_status = llm_service.readiness(load_model=config.llm_health_check_load)
        overall_status = (
            "ok"
            if vector_status["status"] == "ok" and llm_status["status"] not in {"error"}
            else "degraded"
        )
        status_code = 200 if overall_status == "ok" else 503
        return (
            jsonify(
                {
                    "status": overall_status,
                    "app": "GATE Guru",
                    "vector_db": vector_status,
                    "llm": llm_status,
                    "llm_provider": config.llm_provider,
                    "llm_model": config.llm_model_name,
                }
            ),
            status_code,
        )

    @app.get("/api/subjects")
    def subjects():
        return jsonify(config.subject_payload())

    @app.post("/api/index-preloaded")
    @_limit(limiter, config.index_rate_limit)
    def index_preloaded():
        force = bool(request.json.get("force")) if request.is_json and request.json else False
        report = preloader.index_preloaded(force=force)
        return jsonify(report)

    @app.post("/api/upload")
    @_limit(limiter, config.upload_rate_limit)
    def upload_pdf():
        visitor_id = sanitize_visitor_id(request.form.get("visitor_id"))
        subject_group = normalize_subject_group(request.form.get("subject_group") or "General")

        if "file" not in request.files:
            return jsonify({"error": "No PDF file was provided."}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "No PDF file was selected."}), 400

        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({"error": "Invalid file name."}), 400
        if not filename.lower().endswith(".pdf"):
            return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400
        is_pdf, pdf_error = _is_pdf_upload(file)
        if not is_pdf:
            return jsonify({"error": pdf_error}), 400

        visitor_dir = config.upload_dir / visitor_id
        visitor_dir.mkdir(parents=True, exist_ok=True)
        saved_path = visitor_dir / filename

        try:
            file.save(saved_path)
            pages = extract_pdf_pages(saved_path)
            chunks = _chunk_pages_for_config(pages, config)
            if not chunks:
                saved_path.unlink(missing_ok=True)
                return jsonify({"error": "No usable text chunks were created from this PDF."}), 400

            vector_store.delete_user_file(visitor_id, filename)
            metadata = {
                "source_type": "user_upload",
                "visitor_id": visitor_id,
                "filename": filename,
                "category": "User Upload",
                "subject_group": subject_group,
                "is_default_cse_resource": False,
            }
            indexed = vector_store.add_chunks(chunks, metadata)
            return jsonify(
                {
                    "status": "ok",
                    "filename": filename,
                    "visitor_id": visitor_id,
                    "subject_group": subject_group,
                    "chunk_count": indexed,
                    "warnings": [rag_service.FORMULA_WARNING],
                }
            )
        except ValueError as exc:
            saved_path.unlink(missing_ok=True)
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            saved_path.unlink(missing_ok=True)
            return jsonify({"error": f"Upload failed: {exc}"}), 500

    @app.post("/api/ask")
    @_limit(limiter, config.ask_rate_limit)
    def ask():
        payload = request.get_json(silent=True) or {}
        question = (payload.get("question") or "").strip()
        if not question:
            return jsonify({"error": "Question cannot be empty."}), 400

        response = rag_service.answer_question(
            visitor_id=payload.get("visitor_id"),
            question=question,
            mode=payload.get("mode", "ask"),
            subject_group=payload.get("subject_group", "All"),
            include_da_resources=bool(payload.get("include_da_resources", False)),
        )
        return jsonify(response)

    @app.post("/api/ask-stream")
    @_limit(limiter, config.ask_rate_limit)
    def ask_stream():
        payload = request.get_json(silent=True) or {}
        question = (payload.get("question") or "").strip()
        if not question:
            return jsonify({"error": "Question cannot be empty."}), 400

        def generate():
            for item in rag_service.stream_answer_question(
                visitor_id=payload.get("visitor_id"),
                question=question,
                mode=payload.get("mode", "ask"),
                subject_group=payload.get("subject_group", "All"),
                include_da_resources=bool(payload.get("include_da_resources", False)),
            ):
                yield _sse(item["event"], item["data"])

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/feedback")
    @_limit(limiter, "30 per hour")
    def feedback():
        payload = request.get_json(silent=True) or {}
        try:
            result = feedback_service.record(
                visitor_id=payload.get("visitor_id"),
                rating=payload.get("rating"),
                question=payload.get("question", ""),
                answer=payload.get("answer", ""),
                sources=payload.get("sources", []),
            )
            return jsonify(result)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    return app


app = create_app()


if __name__ == "__main__":
    cfg: AppConfig = app.extensions["gate_guru_config"]
    port = int(__import__("os").getenv("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=False)
