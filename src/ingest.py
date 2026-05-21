import os
import chromadb
from embeddings import get_embedding
from config import DATA_DIR, CHROMA_PATH

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping fixed-size chunks."""
    if not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def load_documents(data_dir: str = DATA_DIR) -> list[dict[str, str]]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Documents directory not found: {data_dir}")
    docs = []
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".txt"):
            path = os.path.join(data_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                docs.append({"id": filename, "text": f.read()})
    return docs


def main():
    client = chromadb.PersistentClient(CHROMA_PATH)
    collection = client.get_or_create_collection("docs")

    docs = load_documents()
    if not docs:
        print(f"No .txt files found in '{DATA_DIR}'. Add documents and re-run.")
        return

    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            emb = get_embedding(chunk)
            collection.upsert(
                documents=[chunk],
                embeddings=[emb],
                ids=[f"{doc['id']}-chunk-{i}"],
                metadatas=[{"source": doc["id"], "chunk_index": i}],
            )
        print(f"  {doc['id']} — {len(chunks)} chunk(s)")

    print(f"Ingestion complete. {len(docs)} document(s) processed.")


if __name__ == "__main__":  # pragma: no cover
    main()
