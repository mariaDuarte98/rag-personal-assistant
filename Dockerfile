FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.8.3

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
    && poetry install --without dev --no-interaction --no-ansi

COPY src/ ./src/

# Pre-download the embedding model so first run is instant
RUN PYTHONPATH=/app/src python -c \
    "from config import EMBEDDING_MODEL; from sentence_transformers import SentenceTransformer; SentenceTransformer(EMBEDDING_MODEL)"

RUN mkdir -p chroma_db docs

# Mount docs and chroma_db at runtime to persist data across container runs
VOLUME ["/app/docs", "/app/chroma_db"]

CMD ["python", "src/rag_app.py"]
