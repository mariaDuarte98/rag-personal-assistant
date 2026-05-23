# 🤖 RAG Personal Assistant

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Poetry](https://img.shields.io/badge/Poetry-dependency%20management-60A5FA?logo=poetry&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-orange)
![Gemini](https://img.shields.io/badge/Google%20Gemini-LLM-4285F4?logo=google&logoColor=white)
![CI](https://img.shields.io/github/actions/workflow/status/mariaDuarte98/rag-personal-assistant/ci.yaml?label=CI&logo=github-actions&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

A personal AI assistant powered by a **Retrieval-Augmented Generation (RAG)** pipeline. Ingests your own documents locally, stores them as vector embeddings in ChromaDB, and answers questions using Google Gemini — with full memory of past conversations.

---

## ✨ Features

- 📄 **Local document ingestion** — ingest any text files into a persistent vector store
- 🔍 **Semantic search** — retrieves the most relevant context before answering
- 🧠 **Conversation memory** — past interactions are stored and retrieved for context-aware responses
- 🔒 **Local embeddings** — documents are embedded and stored locally; only the prompt context is sent to the Gemini API

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph Ingestion["⬆️ Ingestion — run once"]
        A[📄 Local Documents] -->|load + embed| D[(docs_collection\nChromaDB)]
    end

    subgraph Query["💬 Query — runtime"]
        E[👤 User Input] -->|embed| F[query_emb]
        F -->|similarity search| D
        F -->|similarity search| G[(memory_collection\nChromaDB)]
        D -->|top-k doc chunks| H[retrieve_context]
        G -->|top-k past turns| H
        H -->|prompt + context| I[gemini_client.py]
        I -->|Gemini API| J[🤖 Response]
        J -->|save turn| G
    end
```

---

## 🗂️ Project Structure

```
rag-personal-assistant/
│
├── src/
│   ├── embeddings.py       # Local sentence embedding generation
│   ├── gemini_client.py    # Google Gemini API wrapper
│   ├── ingest.py           # Document loading & ingestion pipeline
│   └── rag_app.py          # Main app — orchestrates RAG loop
│
├── tests/
│   ├── __init__.py
│   ├── test_embeddings.py  # Unit tests for embedding logic
│   ├── test_ingest.py      # Unit tests for ingestion pipeline
│   ├── test_rag_app.py     # Unit tests for RAG orchestration
│   └── test_gemini_client.py # Unit tests for LLM client
│
├── docs/                   # Documents to ingest
├── chroma_db/              # Persistent vector store (git-ignored)
├── .github/
│   └── workflows/
│       └── ci.yaml         # CI: lint + tests on every push
├── Dockerfile
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation) 1.8+
- A [Google Gemini API Key](https://aistudio.google.com/app/apikey) (free tier available)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/mariaDuarte98/rag-personal-assistant.git
cd rag-personal-assistant

# 2. Set Python version and install dependencies
poetry env use python3.11
poetry install

# 3. Configure environment
cp .env.example .env
# Add your GEMINI_API_KEY to .env
```

### Usage

```bash
# Ingest documents — an example file is included in docs/ to try immediately
poetry run python src/ingest.py

# Start the assistant
poetry run python src/rag_app.py
```

> **Tip:** run `poetry shell` once to activate the virtual environment, then drop the `poetry run` prefix for all subsequent commands.

Type `exit` or `quit` to stop the session. All conversations are automatically saved.

> Replace or add `.txt` files in `docs/` with your own content (notes, articles, a personal bio) and re-run `ingest.py`.

---

## 🧠 Design Decisions

| Decision | Why |
|---|---|
| **Two ChromaDB collections** | `docs` is static — ingested once from files. `memory` is dynamic — written every conversation turn. Keeping them separate allows independent retrieval: up to 3 doc chunks and up to 2 past turns are fetched per query. |
| **Local embeddings** | `sentence-transformers` runs entirely on your machine. Raw documents never leave your environment — only the retrieved context is sent to the Gemini API. |
| **500-char overlapping chunks (50-char overlap)** | Small chunks improve retrieval precision by targeting specific passages rather than whole files. The overlap ensures sentences split across chunk boundaries are still retrievable. |
| **Upsert over insert** | Re-running `ingest.py` is safe — existing chunks are updated in place, not duplicated. Chunk IDs are deterministic (`filename-chunk-N`), so the same file always maps to the same ID. |
| **Lazy model loading** | `SentenceTransformer` and the Gemini client initialise on first use, not at import time. Tests run without loading heavy models; startup is instant. |
| **Capped conversation history** | In-memory history is bounded to `MAX_HISTORY=5` turns. Keeps prompt size predictable and avoids unbounded memory growth in long sessions. |
| **UUID memory IDs** | Past conversation turns are keyed by UUID rather than sequential IDs, so entries can be deleted and re-added without collision. |

---

## 🧪 Running Tests

```bash
poetry run pytest -v
```

---

## 🐳 Docker

```bash
# Build
docker build -t rag-personal-assistant .

# Ingest documents
docker run --rm --env-file .env \
  -v ./docs:/app/docs \
  -v ./chroma_db:/app/chroma_db \
  rag-personal-assistant python src/ingest.py

# Run the assistant (chroma_db persists between runs)
docker run -it --env-file .env \
  -v ./docs:/app/docs \
  -v ./chroma_db:/app/chroma_db \
  rag-personal-assistant
```

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Your Google Gemini API key |

Copy `.env.example` to `.env` and fill in your values.

---

## 📝 License

[MIT](LICENSE)