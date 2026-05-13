FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.3

# Copy dependency files first (layer caching)
COPY pyproject.toml poetry.lock ./

# Install dependencies (no dev deps, no virtualenv inside container)
RUN poetry config virtualenvs.create false \
    && poetry install --without dev --no-interaction --no-ansi

# Copy source code
COPY src/ ./src/
COPY docs/ ./docs/

# Create directory for ChromaDB persistence
RUN mkdir -p chroma_db

# Default command
CMD ["python", "src/rag_app.py"]