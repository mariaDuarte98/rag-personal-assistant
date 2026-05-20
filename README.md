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
│   ├── embeddings.py       # Embedding generation & vector search
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
│       └── ci.yml          # CI: lint + tests on every push
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
# Ingest documents — sample files are included in docs/ to try immediately
poetry run python src/ingest.py

# Start the assistant
poetry run python src/rag_app.py
```

Type `exit` or `quit` to stop the session. All conversations are automatically saved.

> Add your own `.txt` files to `docs/` and re-run `ingest.py` to extend the knowledge base.

---

## 🧪 Running Tests

```bash
poetry run pytest tests/ -v
```

---

## 🐳 Docker

```bash
# Build
docker build -t rag-personal-assistant .

# Run (mount your docs folder at runtime)
docker run -it --env-file .env -v ./docs:/app/docs rag-personal-assistant
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