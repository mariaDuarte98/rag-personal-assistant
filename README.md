# Personal AI Assistant (RAG + Google Gemini)

A personal AI assistant with two modes (creator and public), using simple RAG with ChromaDB and Google Gemini.

---

## Requirements

- Python 3.10+ or 3.11
- Poetry 1.8+
- Google account
- Gemini API Key

---

## Getting Gemini API Key

1. Go to: https://aistudio.google.com/app/apikey
2. Create an API Key
3. Copy it
4. Create a `.env` file in project root:

```bash
GEMINI_API_KEY="YOUR_KEY_HERE"
```

---

## Installation

### Set Python version:
```bash
poetry env use python3.11
```

### Install dependencies:
```bash
poetry install
```

---

## Project Structure
```bash
rag-personal-assistant/
│
├── src/
│   ├── embeddings.py
│   ├── gemini_client.py
│   ├── ingest.py
│   └── rag_app.py
├── docs/
├── chroma_db/
├── .env
├── pyproject.toml
└── README.md
```

---

## Usage

### Ingest documents
```bash
python src/ingest.py
```

### Run assistant
```bash
python src/rag_app.py
```

- Type exit or quit to stop
- New conversations are automatically saved in ChromaDB
