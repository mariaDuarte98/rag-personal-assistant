import os
import chromadb
from embeddings import get_embedding
from typing import List, Dict

DATA_DIR = "docs" # folder where your text files are stored

def embed_text(text) -> list[float]:
    return get_embedding(text)

def load_documents() -> List[Dict[str, str]]:
    docs = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(DATA_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                docs.append({"id": filename, "text": f.read()})
    return docs

def main():
    client = chromadb.PersistentClient("chroma_db")
    collection = client.get_or_create_collection("docs")

    docs = load_documents()
    for doc in docs:
        emb = embed_text(doc["text"])
        collection.add(
            documents=[doc["text"]],
            embeddings=[emb],
            ids=[doc["id"]]
        )

    print("Ingestion complete! Documents embedded.")

if __name__ == "__main__":
    main()
