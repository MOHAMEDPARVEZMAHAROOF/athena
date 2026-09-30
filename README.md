# Athena — a source-cited study companion with adaptive assessments

Athena turns your own study material into a personal tutor. Upload notes (text,
markdown, PDF) and diagrams, then:

- **Study chat** — ask questions and get answers grounded *only* in your
  materials, with inline **[n] citations** linked to the exact source passages.
- **Adaptive quizzes** — multiple-choice questions generated from your notes.
  Difficulty follows a per-topic **Elo rating** (correct → harder, wrong →
  easier); missed questions return later via **Leitner spaced repetition**.
- **Progress dashboard** — per-topic mastery bars, attempt counts, reviews due.

Built for the **Multimodal AI Hackathon 2026 — Track D: Personalized Tutoring &
Adaptive Learning**.

## How it works

```
upload (.txt/.md/.pdf, images) → chunk → BM25 index (pure Python, no deps)
                                            │
question ──► retrieve top passages ──► LLM answers with [n] citations ──► UI
                                            │
quiz ──► LLM writes grounded MCQs ──► Elo update + Leitner review schedule
```

- **Retrieval is BM25**, implemented from scratch in `athena/retrieval.py` —
  honest keyword retrieval, no fake "semantic search" claims.
- **LLM backend is OpenAI-compatible** (`athena/llm.py`, stdlib `urllib`, no SDK).
  The demo runs on `nvidia/nemotron-3-super-120b-a12b` via the NVIDIA Build API
  (`https://integrate.api.nvidia.com/v1`). Point `ATHENA_BASE_URL` /
  `ATHENA_MODEL` at any OpenAI-compatible endpoint instead.
- **No API key?** The app still works in *honest fallback mode*: chat shows the
  top retrieved passages verbatim, and quizzes use template-generated
  fill-in-the-blank questions. The UI labels which mode is active.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# bring your own key (free at build.nvidia.com)
export ATHENA_API_KEY="nvapi-..."
# optional: export ATHENA_MODEL="another-model" ATHENA_BASE_URL="https://..."

.venv/bin/python -m uvicorn athena.app:app --port 8000
# open http://localhost:8000
```

Or `./run.sh` (uses port 8000, data in `./data`).

Upload the notes in `sample_docs/` to try it immediately:
`sample_docs/photosynthesis.md`, `sample_docs/indian_constitution_rights.md`.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ATHENA_API_KEY` | — | API key for AI features (else fallback mode) |
| `ATHENA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | OpenAI-compatible base URL |
| `ATHENA_MODEL` | `nvidia/nemotron-3-super-120b-a12b` | Chat model |
| `ATHENA_DATA` | `./data` | SQLite db + uploads directory |
| `ATHENA_LLM_ADAPTER` | — | Path to a Python file exposing `complete(messages, …)` — used for local demos with injected credentials; never commit keys |

## API

- `POST /api/upload` — multipart `file` (+ optional `title`, `caption` for images)
- `GET /api/documents`, `DELETE /api/documents/{id}`
- `POST /api/chat` — SSE: `sources` → `delta`* → `done` (with cited passages)
- `POST /api/quiz/next` — `{topic?}` → question (answer withheld)
- `POST /api/quiz/answer` — `{question_id, given}` → correctness, explanation, new rating
- `GET /api/topics`, `GET /api/quiz/due`, `GET /api/status`

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

9 tests: BM25 ranking, chunking, ingest round-trip, Elo updates, Leitner
scheduling, MCQ grading, cloze generation.

## Project structure

```
athena/        FastAPI app, SQLite store, BM25, ingest, RAG tutor, quiz engine, LLM client
web/           vanilla JS single-page UI (Study / Quiz / Library / Progress)
sample_docs/   original study notes bundled for demo
tests/         pytest suite
```

## License

MIT — see `LICENSE`.
