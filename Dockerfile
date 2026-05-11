FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/.cache/huggingface \
    TRANSFORMERS_CACHE=/data/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/data/.cache/sentence_transformers \
    CHROMA_DB_DIR=/data/chroma_db \
    UPLOAD_DIR=/data/uploads \
    FEEDBACK_LOG_FILE=/data/feedback/feedback.jsonl \
    HISTORY_DB_FILE=/data/history/conversation_history.sqlite3 \
    PRELOADED_DIR=/app/data/preloaded_gate_cse \
    LLM_PROVIDER=local \
    LLM_MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    PORT=7860

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

RUN useradd -m -u 1000 appuser \
    && mkdir -p /data/chroma_db /data/uploads /data/feedback /data/history /data/.cache/huggingface /data/.cache/sentence_transformers \
    && chown -R appuser:appuser /data /app

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
