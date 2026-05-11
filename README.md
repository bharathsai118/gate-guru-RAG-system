---
title: GATE Guru Public RAG
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# GATE Guru

GATE Guru is a public Traditional RAG-based AI tutor for GATE Computer Science and Engineering preparation. It uses preloaded GATE CSE PDFs, visitor-scoped PDF uploads, ChromaDB retrieval, SentenceTransformers embeddings, a local Hugging Face LLM by default, and an optional OpenAI-compatible API LLM provider.

## Features

- Ask questions from preloaded GATE CSE PDFs.
- Upload personal PDF notes without login or signup.
- Retrieve from public resources plus only the current browser visitor's uploads.
- Study modes: Ask, Solve, Revision, and Quiz.
- Source citations with filename, page number, category, and source type.
- Token streaming for answers through `/api/ask-stream`.
- Cross-encoder reranking after vector retrieval.
- Short per-visitor conversation history for follow-up questions.
- Visitor-keyed rate limits, upload cleanup, and answer feedback logging.
- Docker-ready for Hugging Face Spaces.
- No paid API is required by default. API inference is optional through environment variables.

## RAG Pipeline

```text
PDF files
-> PyMuPDF page text extraction with formula-layout warning
-> text cleaning
-> token-aware chunking with character fallback
-> all-MiniLM-L6-v2 embeddings
-> ChromaDB persistent vector store
-> filtered similarity search
-> optional cross-encoder reranking
-> GATE-focused prompt construction
-> deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B through transformers
-> streamed answer plus citations
```

## Folder Structure

```text
gate-guru/
  app.py
  requirements.txt
  Dockerfile
  README.md
  .env.example
  .gitignore
  templates/
    index.html
  static/
    app.js
    styles.css
  services/
    __init__.py
    config.py
    pdf_service.py
    chunking_service.py
    vector_store.py
    llm_service.py
    rag_service.py
    reranker.py
    history_service.py
    feedback_service.py
    cleanup_service.py
    preloader.py
    utils.py
  data/
    preloaded_gate_cse/
      Mathematics_and_Aptitude/
        volume1.pdf
      Core_CSE/
        volume2.pdf
      Optional_DA_AI_ML/
        da.pdf
  uploads/
    .gitkeep
  chroma_db/
    .gitkeep
```

## Add Preloaded Resources

Place the PDFs here:

```text
data/preloaded_gate_cse/Mathematics_and_Aptitude/volume1.pdf
data/preloaded_gate_cse/Core_CSE/volume2.pdf
data/preloaded_gate_cse/Optional_DA_AI_ML/da.pdf
```

You can also place the original folder as `GATE_RESOURCE/` beside this project or inside this project. On startup or manual indexing, the app copies `volume1.pdf`, `volume2.pdf`, and `da.pdf` into the organized `data/preloaded_gate_cse/` layout if the target files are missing.

Default retrieval includes `volume1.pdf`, `volume2.pdf`, and the current visitor's uploads. `da.pdf` is excluded unless the DA / AI / ML checkbox is enabled or the Optional DA / AI / ML subject group is selected.

## Local Setup

```bash
python -m venv venv
source venv/bin/activate
```

For Windows:

```powershell
venv\Scripts\activate
```

Then install and run:

```bash
pip install -r requirements.txt
python app.py
```

Open:

```text
http://localhost:7860
```

## Hugging Face Spaces Deployment

1. Create a new Hugging Face Space.
2. Choose Docker as the Space SDK.
3. Upload this `gate-guru/` project to the Space repository.
4. Add your PDF resources under `data/preloaded_gate_cse/`.
5. Keep `app_port: 7860` in this README metadata.
6. The Dockerfile runs Gunicorn on `0.0.0.0:7860` with one worker and four threads.

The Dockerfile stores ChromaDB, uploads, and model caches under `/data`:

```text
CHROMA_DB_DIR=/data/chroma_db
UPLOAD_DIR=/data/uploads
FEEDBACK_LOG_FILE=/data/feedback/feedback.jsonl
HISTORY_DB_FILE=/data/history/conversation_history.sqlite3
HF_HOME=/data/.cache/huggingface
TRANSFORMERS_CACHE=/data/.cache/huggingface
SENTENCE_TRANSFORMERS_HOME=/data/.cache/sentence_transformers
```

Persistent `/data` storage is recommended for Spaces so the vector index and model cache survive restarts.

## Environment Variables

| Variable | Default |
| --- | --- |
| `CHROMA_DB_DIR` | `./chroma_db` |
| `PRELOADED_DIR` | `data/preloaded_gate_cse` |
| `UPLOAD_DIR` | `uploads` |
| `AUTO_INDEX_PRELOADED` | `true` |
| `DENSE_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` |
| `LLM_PROVIDER` | `local` |
| `LLM_MODEL_NAME` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| `LLM_API_BASE_URL` | empty |
| `LLM_API_KEY` | empty |
| `LLM_API_MODEL_NAME` | empty |
| `LLM_API_TIMEOUT_SECONDS` | `120` |
| `LLM_HEALTH_CHECK_LOAD` | `false` |
| `LLM_TRUST_REMOTE_CODE` | `true` |
| `LLM_STRIP_THINKING` | `true` |
| `TOP_K` | `5` |
| `RETRIEVAL_CANDIDATE_K` | `10` |
| `RERANKER_ENABLED` | `true` |
| `RERANKER_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `MAX_CONTEXT_CHUNKS` | `5` |
| `MAX_NEW_TOKENS` | `512` |
| `CHUNK_SIZE` | `900` |
| `CHUNK_OVERLAP` | `150` |
| `TOKEN_CHUNKING_ENABLED` | `true` |
| `CHUNK_TOKENIZER_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHUNK_SIZE_TOKENS` | `256` |
| `CHUNK_OVERLAP_TOKENS` | `40` |
| `MAX_UPLOAD_MB` | `30` |
| `HISTORY_ENABLED` | `true` |
| `HISTORY_MAX_TURNS` | `4` |
| `HISTORY_DB_FILE` | `history/conversation_history.sqlite3` |
| `ASK_RATE_LIMIT` | `20 per hour` |
| `UPLOAD_RATE_LIMIT` | `8 per hour` |
| `INDEX_RATE_LIMIT` | `3 per hour` |
| `FEEDBACK_LOG_FILE` | `feedback/feedback.jsonl` |
| `CLEANUP_ENABLED` | `true` |
| `UPLOAD_TTL_HOURS` | `24` |
| `CLEANUP_INTERVAL_MINUTES` | `60` |

To use an API model instead of local Hugging Face inference, set:

```env
LLM_PROVIDER=openai_compatible
LLM_API_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=your_api_key
LLM_API_MODEL_NAME=your_model_name
```

## User Isolation Without Login

The frontend creates a `visitor_id` with `crypto.randomUUID()` and stores it in `localStorage` as `gate_guru_visitor_id`. It sends that id in request bodies and in the `X-Visitor-Id` header for rate limiting. Uploads are saved under `uploads/{visitor_id}/`, and uploaded chunks are stored with Chroma metadata:

```json
{
  "source_type": "user_upload",
  "visitor_id": "browser-generated-id",
  "category": "User Upload"
}
```

Retrieval always filters uploaded chunks by the same `visitor_id`, so another browser visitor cannot search those uploaded PDFs.

Conversation history is stored in SQLite at `HISTORY_DB_FILE`, keeping the most recent turns per visitor so follow-up questions survive app restarts when the DB is on persistent storage.

## API Routes

- `GET /` renders the web UI.
- `GET /health` returns real vector DB readiness and LLM readiness. Local LLMs are reported as `not_loaded_lazy` until the first question unless `LLM_HEALTH_CHECK_LOAD=true`.
- `GET /api/subjects` returns subject groups and modes.
- `POST /api/index-preloaded` indexes public PDFs.
- `POST /api/upload` uploads and indexes a visitor-scoped PDF.
- `POST /api/ask` retrieves context and generates an answer.
- `POST /api/ask-stream` streams answer tokens as server-sent events.
- `POST /api/feedback` records a thumbs up/down rating to JSONL.

## Limitations

- The app intentionally defaults to `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` instead of the full `deepseek-ai/DeepSeek-R1` model. Full DeepSeek-R1 is too large for normal local or Hugging Face Space inference.
- You can change `LLM_MODEL_NAME` later to larger distilled checkpoints such as `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`, or `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` if your hardware has enough memory.
- Local CPU generation with large Hugging Face LLMs can be very slow or fail because of memory limits.
- Scanned image-only PDFs require OCR, which is not included.
- Formula-heavy PDFs can lose equation layout during text extraction. Always verify formulas against the cited PDF page.
- Reranking improves relevance but adds another model download and CPU/GPU work.
- Browser `localStorage` is the isolation boundary; clearing browser storage creates a new visitor.
- Rate limiting uses remote address plus `X-Visitor-Id` when available; API clients should send that header.
- ChromaDB and uploads should use persistent storage in production.
- The LLM is instructed not to hallucinate, but high-stakes answers should still be verified against the cited sources.

## Troubleshooting

- Missing preloaded PDFs: add `volume1.pdf`, `volume2.pdf`, and `da.pdf`, then click **Index Resources**.
- Empty PDF error: the PDF likely has scanned pages or no extractable text.
- Slow first question: the LLM and embedding model download and load lazily.
- Slow first reranked answer: the cross-encoder reranker downloads lazily.
- Model load failure: set `LLM_MODEL_NAME` to a smaller compatible causal language model, such as `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` for lightweight testing.
- Accelerate is included because `low_cpu_mem_usage=True` in the local Hugging Face loader relies on it.
- Upload rejected as invalid PDF: confirm the file has a `.pdf` extension, a PDF MIME type, and a valid `%PDF-` header.
- API provider error: confirm `LLM_PROVIDER`, `LLM_API_BASE_URL`, `LLM_API_KEY`, and `LLM_API_MODEL_NAME`.
- ChromaDB errors: delete the local `chroma_db/` directory and re-index if the database becomes corrupted.
- Upload too large: increase `MAX_UPLOAD_MB` if your deployment has enough disk and memory.

## Testing Checklist

1. Start app locally.
2. Visit `/health`.
3. Index preloaded PDFs.
4. Ask a question from `volume1.pdf`.
5. Ask a question from `volume2.pdf`.
6. Confirm `da.pdf` is not used by default.
7. Enable DA resources and confirm `da.pdf` can be used.
8. Upload a user PDF.
9. Ask a question from uploaded PDF.
10. Confirm citations show filename and page number.
11. Confirm another `visitor_id` cannot access the uploaded PDF.
12. Deploy to Hugging Face Space.
