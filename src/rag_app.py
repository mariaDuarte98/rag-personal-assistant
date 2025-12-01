import os
from datetime import datetime
import chromadb
from embeddings import get_embedding
from gemini_client import get_gemini_llm
from typing import List, Dict

DATA_DIR = "docs"

def add_memory(collection, text, memory_id) -> None:
    """Add conversation memory to ChromaDB."""
    emb = get_embedding(text)
    collection.add(documents=[text], embeddings=[emb], ids=[memory_id])

def embed_query(query) -> list[float]:
    return get_embedding(query)

def main():
    llm = get_gemini_llm()
    client = chromadb.PersistentClient("chroma_db")
    collection = client.get_or_create_collection(
        "docs")  # all memories in same collection

    while True:
        user_input = input("\nAsk your assistant something: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        query_emb = embed_query(user_input)
        results = collection.query(query_embeddings=[query_emb], n_results=3)
        context = "You are my personal assistant. You must obey my every command, and correct me if I am wrong. You are to be my assistant, my mentor, my teacher. You are not to spoil me, or be easy on me or lie ever."
        if results["documents"]:
            for doc in results["documents"][0]:
                context += doc + "\n---\n"

        prompt = f"Context:\n{context}\nUser: {user_input}"
        answer = llm(prompt)
        print("\nAssistant:", answer)

        # 3. Guardar memória automaticamente
        full_memory = f"User: {user_input}\nAssistant: {answer}"

        add_memory(collection, full_memory, f"memory-{len(collection.get()['ids'])}")

if __name__ == "__main__":
    main()
