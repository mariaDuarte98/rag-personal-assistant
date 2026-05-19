# RAG Personal Assistant — Claude Context

## What this project is

A local RAG (Retrieval-Augmented Generation) assistant. It embeds your `.txt` documents into ChromaDB, then answers questions by retrieving the most relevant chunks and passing them as context to Gemini.

## Architecture

```
docs/ (your .txt files)
    → ingest.py       chunks + embeds → ChromaDB "docs" collection
    → rag_app.py      query loop: embed question → retrieve context → prompt Gemini
```

| File | Responsibility |
|---|---|
| `src/embeddings.py` | Local sentence embeddings via `sentence-transformers` (lazy-loaded) |
| `src/ingest.py` | Loads `.txt` files, chunks them, upserts into ChromaDB with metadata |
| `src/rag_app.py` | Main query loop — retrieves from both `docs` and `memory` collections |
| `src/gemini_client.py` | Lazy Gemini client, validates `GEMINI_API_KEY` on first use |

Two ChromaDB collections:
- `docs` — chunked document embeddings, with `source` and `chunk_index` metadata
- `memory` — past conversation turns, keyed by UUID

## Setup

```bash
poetry install
cp .env.example .env   # set GEMINI_API_KEY
mkdir docs             # drop your .txt files here
poetry run python src/ingest.py    # embed documents
poetry run python src/rag_app.py   # start the assistant
```

## Running tests and lint

```bash
poetry run pytest tests/ -v
poetry run ruff check src/ tests/
```

All CI checks must pass before merging.

## Key design decisions

- **Chunking**: 500-char overlapping chunks (50-char overlap) so retrieval targets specific passages, not whole files
- **Upsert over add**: re-running ingestion is safe — existing chunks are updated, not duplicated
- **Lazy model loading**: `SentenceTransformer` and Gemini are initialised on first use, not at import time
- **UUID memory IDs**: avoids collisions when entries are deleted and re-added
- **Metadata on chunks**: every chunk stores `{"source": filename, "chunk_index": i}` so the UI can show attribution
