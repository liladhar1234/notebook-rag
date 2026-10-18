# Notebook RAG — Phase 1

A clean, production-grade **Hybrid Retrieval-Augmented Generation** engine built with FastAPI. Fully local, zero cloud dependency for the core stack.

## Quick Start

```bash
# 1. Clone & enter
git clone <your-repo> && cd notebook-rag

# 2. Copy env and fill in your LLM key
cp .env.example .env
# Edit .env: set LLM_API_KEY (Gemini, Groq, etc.) or ANTHROPIC_API_KEY

# 3. Build & run
docker compose up --build
```

API is live at **http://localhost:8000**. Docs at **http://localhost:8000/docs**.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Ingest a PDF, Markdown, or TXT file |
| `POST` | `/query` | Hybrid RAG query against ingested documents |
| `GET` | `/documents?chat_id=...` | List documents for a chat session |
| `DELETE` | `/documents/{doc_id}` | Delete document + all its vectors |
| `GET` | `/health` | Health check |

### Upload a document
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@report.pdf" \
  -F "chat_id=my-session-123"
```

### Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?", "chat_id": "my-session-123"}'
```

---

## Architecture

```
api  →  generation / ingestion  →  retrieval / embeddings / registry  →  models
```

**Strict top-down layering.** The API never touches Qdrant directly; retrieval never imports LLM code.

### Data Flow

```
UPLOAD: File → Parser → Chunker (token-aware, page-bounded)
             → Dense Embed (bge-small) + Sparse Encode (BM25)
             → Qdrant Upsert + SQLite Registry

QUERY:  Question → Dense + Sparse Embed
                 → Qdrant Top-30 Dense + Top-30 Sparse
                 → Python RRF → Top-6 Chunks
                 → Evidence Gate (reject if empty/low-score)
                 → XML Context Assembly → LLM → Citation Verify → Response
```

---

## LLM Providers

Set in `.env` or `docker-compose.yml`:

| Provider | `LLM_PROVIDER` | Notes |
|----------|---------------|-------|
| Gemini (default) | `openai_compat` | Free tier via `generativelanguage.googleapis.com` |
| Groq | `openai_compat` | Set `LLM_BASE_URL=https://api.groq.com/openai/v1/` |
| Ollama | `openai_compat` | Set `LLM_BASE_URL=http://localhost:11434/v1/` |
| Claude | `anthropic` | Set `ANTHROPIC_API_KEY` |

---

## Local Development (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env   # fill in keys

uvicorn notebook_rag.api.main:app --reload --reload-dir src
```

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Tests use isolated temp directories for Qdrant and SQLite — no cleanup needed.

---

## Project Structure

```
src/notebook_rag/
├── models.py              # Domain entities (ParsedPage, Chunk, SearchHit, Citation)
├── config.py              # pydantic-settings — all knobs in one place
├── ingestion/             # Parse → Chunk → Embed → Index → Register
│   ├── parsers/pdf.py     # PyMuPDF, table-preserving
│   ├── parsers/markdown.py # Heading-aware section parser
│   ├── chunker.py         # Token-aware, page-bounded, deterministic IDs
│   └── pipeline.py        # Orchestrator
├── embeddings/            # Dense (bge-small) + Sparse (BM25)
├── retrieval/             # Qdrant store, RRF, HybridRetriever
├── generation/            # LLMClient, prompts, evidence gate, service
├── registry/              # SQLite (SQLAlchemy async) document metadata
└── api/                   # FastAPI routes, deps, schemas
```
