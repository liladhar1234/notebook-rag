# notebook-rag

Chat with your own documents. Upload PDFs, Word, Excel, PowerPoint, images or text, ask
questions, and get answers drawn **only** from what you uploaded. If your files don't contain
the answer, it says so instead of guessing.

Built with FastAPI and Qdrant. Hybrid retrieval (dense + BM25) fused with RRF, cross-encoder
reranking, an evidence gate that refuses to answer without support, and a bounded agent loop
for comparison and multi-part questions. Each chat is an isolated workspace with its own
documents and history, and every user has their own account.

---

## Features

- **Grounded answers.** The model sees only passages retrieved from the chat's documents. No web
  search, no outside knowledge. If the best passage scores below the relevance threshold, the
  question is refused *before* any LLM call is made.
- **Hybrid retrieval.** Dense embeddings (`bge-small`) and BM25 keyword search, fused with
  Reciprocal Rank Fusion, then re-scored by a cross-encoder reranker.
- **Conversational.** Follow-ups such as *"and what about the other one?"* are rewritten into
  self-contained questions using the chat history. Optional multi-query expansion.
- **Agent for hard questions.** Comparisons, "across all documents" and multi-part questions go
  to a bounded loop (at most 4 steps, 60 s) with `search_documents`, `read_pages` and
  `list_documents` tools. It has no web tool.
- **Whole-document questions.** "Summarize this" works from the opening passages of each
  document, since no single passage matches such a question.
- **Reads almost anything** (see the table below), including scanned PDFs and screenshots via
  local OCR.
- **Accounts and chats.** Sign-up and login with sessions, per-user chats, daily question limits
  to protect a shared LLM quota. A new chat starts empty.
- **Background ingestion.** Uploads return immediately; indexing runs in the background and the
  UI shows progress.
- **Evaluation.** A `/evaluate` endpoint scores an answer with RAGAS (faithfulness and others).

### Supported files

| Type | Extensions | Notes |
|---|---|---|
| PDF | `.pdf` | Text, tables (as Markdown), and OCR for scanned pages (up to 40 scanned pages per file) |
| Word | `.docx` | Headings, paragraphs, lists, tables |
| Spreadsheets | `.xlsx` `.xlsm` `.csv` `.tsv` | Each sheet becomes a table; first 20,000 rows indexed |
| PowerPoint | `.pptx` | One entry per slide, with speaker notes and tables |
| Web pages | `.html` `.htm` | Visible text, sections, tables |
| Images | `.png` `.jpg` `.jpeg` `.webp` `.bmp` `.tif` `.tiff` `.gif` | Text in the image via OCR (a photo with no text is rejected with a clear message) |
| Text and code | `.md` `.markdown` `.txt` `.json` `.xml` `.yaml` `.csv` `.log` `.py` `.js` `.sql` … | Plain text of any common kind |

Not supported (rejected with a clear message): old `.doc` `.xls` `.ppt`, `.rtf`, `.odt`, zip
archives, audio and video.

**Limits:** 20 MB per file, 5 documents per chat (both configurable).

---

## Run it on your machine

### What you need

- **Either** Docker (Docker Desktop, or Docker Engine with Compose v2) **or** Python 3.11+
- An **LLM API key**. The default is Google's Gemini free tier: create a key in
  [Google AI Studio](https://aistudio.google.com/). Groq, Ollama (fully local) and Claude also work, see
  [LLM providers](#llm-providers).
- A few GB of free disk space and RAM for the models.

### Option A: Docker (recommended)

```bash
# 1. Get the code
git clone https://github.com/<your-username>/notebook-rag.git
cd notebook-rag

# 2. Create your settings file and add your key
cp .env.example .env
#    open .env and set LLM_API_KEY=<your key>
#    also check LLM_MODEL is a model name your key can use (see AI Studio)

# 3. Build and start (the first build takes several minutes: it downloads the models)
docker compose up --build
```

Wait for the log line `Notebook RAG ready`, then open **http://localhost:8000**.

1. On the login screen, switch to **Create account**, enter any email and a password of at least
   8 characters, and press **Sign up**.
2. Click **＋ New chat**, then drag a file into the upload area. Wait until it shows its number
   of chunks.
3. Ask a question about the file.

Useful commands:

```bash
docker compose up -d            # start in the background
docker compose logs -f api      # watch the logs
docker compose down             # stop
```

Your accounts, chats, vectors and uploaded files live in `./data/` and survive restarts. To start
completely fresh, run `docker compose down` and delete the `data/` folder.

### Option B: Python, without Docker

```bash
git clone https://github.com/<your-username>/notebook-rag.git
cd notebook-rag

python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .

cp .env.example .env               # Windows: copy .env.example .env
#    open .env and set LLM_API_KEY=<your key>

uvicorn notebook_rag.api.main:app --reload --reload-dir src
```

Run it **from the project root**: the default data folder is `./data`. The first start downloads the
embedding and reranker models (a few hundred MB) and takes a while. Then open
**http://localhost:8000**.

> Run a single worker (the default). The vector store and SQLite are file-based and single-process.

### Check that it works

```bash
curl http://localhost:8000/health
```

returns `{"status":"ok", ...}`. The interactive API docs are at http://localhost:8000/docs.

---

## LLM providers

Set these in `.env`:

| Provider | `LLM_PROVIDER` | Other settings |
|---|---|---|
| Gemini (default) | `openai_compat` | `LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`, `LLM_API_KEY`, `LLM_MODEL` |
| Groq | `openai_compat` | `LLM_BASE_URL=https://api.groq.com/openai/v1/`, `LLM_API_KEY`, `LLM_MODEL` |
| Ollama (fully local) | `openai_compat` | `LLM_BASE_URL=http://localhost:11434/v1/` (use `http://host.docker.internal:11434/v1/` from inside Docker), any `LLM_API_KEY`, `LLM_MODEL=<a model you pulled>` |
| Claude | `anthropic` | `ANTHROPIC_API_KEY`, `LLM_MODEL=<a Claude model name>` |

> **Privacy:** your questions and the retrieved passages are sent to the LLM provider you choose.
> Use Ollama if documents must not leave your machine. Embeddings, reranking and OCR always run
> locally.

---

## How it works

```
UPLOAD   file → parser (PDF / Word / sheet / slides / HTML / image-OCR / text)
              → structure-aware chunks (token-bounded, never cross a page or section)
              → dense embedding (bge-small) + sparse BM25 vector
              → Qdrant  +  SQLite registry  +  original saved to data/uploads

QUERY    question + recent chat history
              → router: comparison / multi-part?  → agent loop (≤ 4 steps)
              → otherwise: query planner (rewrite follow-ups, optional multi-query)
              → dense top-N + BM25 top-N per query → one RRF fusion → candidate pool
              → cross-encoder reranks the pool → top 6
              → evidence gate: best score below threshold → "not in your documents" (no LLM call)
              → XML <source> context → LLM → inline citations stripped
              → answer + sources (file name, page or section, score)
```

Layering is strictly top-down: `api → generation / ingestion → retrieval / embeddings / registry → models`.
The API layer never touches Qdrant directly, and retrieval never imports LLM code.

---

## Configuration

Everything is an environment variable (via `.env`). The most useful ones:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_PROVIDER` | see `.env.example` | Which model answers |
| `RETRIEVAL_TOP_N` / `RETRIEVAL_TOP_K` | 30 / 6 | Candidates per retriever / chunks sent to the LLM |
| `RERANK_CANDIDATES` | 30 | Pool size the reranker scores |
| `RERANKER_ENABLED` | true | Turn the cross-encoder on or off |
| `MIN_RERANK_SCORE` | -5.0 | Evidence gate: refuse when the best reranker score is below this |
| `MULTI_QUERY_ENABLED` / `MULTI_QUERY_COUNT` | true / 3 | Extra search wordings generated by the LLM |
| `AGENT_ENABLED`, `AGENT_MAX_STEPS`, `AGENT_TIMEOUT_S` | true, 4, 60 | The agent loop for complex questions |
| `HISTORY_MESSAGES` | 6 | Recent messages used to resolve follow-ups |
| `CHUNK_TOKENS` / `CHUNK_OVERLAP_TOKENS` | 400 / 60 | Chunk size and overlap |
| `MAX_UPLOAD_BYTES`, `MAX_DOCS_PER_CHAT` | 20 MB, 5 | Upload limits |
| `OCR_ENABLED`, `MAX_PDF_OCR_PAGES`, `OCR_DPI` | true, 40, 200 | Scanned-page OCR |
| `ALLOW_SIGNUP` | true | Set `false` to close sign-up once your accounts exist |
| `DAILY_QUESTIONS_PER_USER` / `DAILY_QUESTIONS_GLOBAL` | 10 / 100 | Protect a shared (free-tier) LLM quota; 0 = unlimited |
| `COOKIE_SECURE` | false | Set `true` when served over HTTPS |

The full list, with comments, is in [`src/notebook_rag/config.py`](src/notebook_rag/config.py).

### Tuning the evidence gate

`MIN_RERANK_SCORE` is a starting point. Every query logs its top reranker score
(`Retrieval [...] top score ...`). Ask a few questions your documents answer and a few they don't,
compare the logged scores, and set the threshold between the two groups.

---

## API

Log in first: the session lives in a cookie.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/auth/register`, `/auth/login`, `/auth/logout` | Accounts and sessions |
| `GET` | `/auth/me`, `/auth/usage` | Current user, and questions left today |
| `POST` / `GET` | `/chats` | Create a chat / list your chats |
| `GET` | `/chats/{chat_id}/messages` | Message history with the sources of each answer |
| `DELETE` | `/chats/{chat_id}` | Delete a chat with its documents, vectors, files and messages |
| `POST` | `/upload` | Upload a file into a chat (`202`; indexing runs in the background) |
| `GET` | `/documents?chat_id=…`, `/documents/{doc_id}` | List documents / check one's status (`processing`, `ready`, …) |
| `DELETE` | `/documents/{doc_id}` | Delete a document and its vectors |
| `POST` | `/query` | Ask a question in a chat |
| `POST` | `/evaluate` | RAGAS scores for an answer (counts as 3 questions against your daily limit) |
| `GET` | `/health` | Health check |

Example with `curl`:

```bash
# sign up (the cookie is saved in the file "jar")
curl -c jar -X POST http://localhost:8000/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"email":"me@example.com","password":"a-long-password"}'

# new chat, upload, ask
CHAT=$(curl -s -b jar -X POST http://localhost:8000/chats | python -c "import sys,json; print(json.load(sys.stdin)['chat_id'])")
curl -b jar -X POST http://localhost:8000/upload -F "file=@report.pdf" -F "chat_id=$CHAT"
curl -b jar -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d "{\"question\": \"What are the key findings?\", \"chat_id\": \"$CHAT\"}"
```

---

## Tests

```bash
pip install -r requirements.txt
pytest -q
```

The suite runs **offline**: embeddings, reranker and LLM are replaced by small fakes, so no model
downloads and no API key are needed. Qdrant and SQLite run in temporary folders, so your real
`data/` is never touched. (The chunker uses `tiktoken`; if its vocabulary can't be downloaded,
the tests fall back to a stand-in.)

---

## Project structure

```
src/notebook_rag/
├── config.py                  # every setting, in one place
├── models.py                  # shared domain types
├── api/                       # FastAPI app, routes, schemas, auth helpers
│   └── routes/                # auth, chats, upload, documents, query, evaluate
├── ingestion/                 # parse → chunk → embed → index → register
│   ├── parsers/               # pdf, word, sheet, slides, web, image, text, markdown, OCR
│   ├── chunker.py             # token-aware, structure-preserving, deterministic IDs
│   ├── pipeline.py            # orchestration + background indexing
│   └── progress.py            # live progress for the UI
├── embeddings/                # dense (bge-small) and sparse (BM25)
├── retrieval/                 # Qdrant store, RRF, hybrid retriever, reranker,
│                              # query planner, complexity router, overview mode
├── generation/                # LLM clients, prompts, evidence gate, service, agent
├── registry/                  # SQLite: documents, chats, messages, users, usage
├── evaluation/                # RAGAS
└── static/index.html          # the web UI
tests/                         # offline tests with fakes (see conftest.py)
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Questions fail with an error about the model or a `404` | `LLM_MODEL` is not available to your key. Check the model names in Google AI Studio and update `.env`. |
| `429` or "daily limit reached" | Either your LLM provider's rate limit, or this app's `DAILY_QUESTIONS_*` limit. Wait, or raise the limit in `.env`. |
| Upload shows ⚠ under the filename | Part of the file couldn't be read (for example a PDF with more than 40 scanned pages). The reason is shown; the rest is still searchable. |
| First start is slow | The models are downloading (Docker does this during the build). Later starts are fast. |
| Port 8000 is already in use | Stop the other program, or change the port mapping in `docker-compose.yml`. |
| Docker build or start runs out of memory | Give Docker more RAM (Docker Desktop → Settings → Resources). |
| Changes to `.env` have no effect | Restart: `docker compose down && docker compose up`. |
| Answers say "not in your documents" for things that *are* there | The gate may be too strict: lower `MIN_RERANK_SCORE` (for example to -7) and compare the logged scores. |

---

## Current limitations

- **Single instance.** Qdrant runs in embedded local mode and the registry is SQLite, so run one
  worker on one machine. Moving to Qdrant Cloud, Postgres and object storage is the path to scale.
- Answers appear when fully generated (no token streaming yet).
- OCR is CPU-based, roughly 1–3 seconds per page, so large scans take a while.
- Charts and photos without text are not described (no vision model yet).

## Security notes

- Never commit `.env`. It is in `.gitignore`. If a key is ever pushed, rotate it.
- Passwords are hashed with scrypt; sessions are HTTP-only cookies; failed logins are rate-limited.
- Behind HTTPS, set `COOKIE_SECURE=true`. For a private instance, set `ALLOW_SIGNUP=false` after
  creating your accounts.

## License

Add a license before making the repository public (MIT is a common choice for projects like this).
