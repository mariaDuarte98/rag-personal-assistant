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

- **Two ChromaDB collections**: `docs` is static (ingested once); `memory` is dynamic (written every turn). Separate retrieval budgets: 3 doc chunks, 2 memory turns.
- **Local embeddings**: `sentence-transformers` runs on-device — raw documents never leave the machine, only retrieved context goes to Gemini.
- **Chunking**: 500-char overlapping chunks (50-char overlap) so retrieval targets specific passages, not whole files. Overlap prevents sentences split across boundaries from being missed.
- **Upsert over add**: re-running ingestion is safe — existing chunks are updated in place, not duplicated. Chunk IDs are deterministic (`filename-chunk-N`).
- **Lazy model loading**: `SentenceTransformer` and Gemini initialise on first use, not at import time — tests run without loading heavy models.
- **Capped conversation history**: bounded to `MAX_HISTORY=5` turns in memory to keep prompt size predictable.
- **UUID memory IDs**: avoids collisions when entries are deleted and re-added.
