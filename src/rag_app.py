import uuid
import chromadb
from embeddings import get_embedding
from gemini_client import get_gemini_llm
from config import CHROMA_PATH


def embed_query(query: str) -> list[float]:
    return get_embedding(query)


def add_memory(memory_collection, text: str, memory_id: str | None = None) -> str:
    """Persist a conversation turn. Returns the id used."""
    if memory_id is None:
        memory_id = f"memory-{uuid.uuid4().hex}"
    emb = get_embedding(text)
    memory_collection.add(documents=[text], embeddings=[emb], ids=[memory_id])
    return memory_id


def retrieve_context(docs_collection, memory_collection, query_emb: list[float]) -> str:
    context = ""

    n_docs = len(docs_collection.get()["ids"])
    if n_docs > 0:
        results = docs_collection.query(
            query_embeddings=[query_emb],
            n_results=min(3, n_docs),
            include=["documents", "metadatas"],
        )
        context += "### From your documents:\n"
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            source = meta.get("source", "unknown") if meta else "unknown"
            context += f"[{source}]\n{doc}\n---\n"

    n_mem = len(memory_collection.get()["ids"])
    if n_mem > 0:
        results = memory_collection.query(
            query_embeddings=[query_emb],
            n_results=min(2, n_mem),
        )
        context += "\n### From past conversations:\n"
        for mem in results["documents"][0]:
            context += mem + "\n---\n"

    return context


def main():
    llm = get_gemini_llm()
    client = chromadb.PersistentClient(CHROMA_PATH)

    docs_collection = client.get_or_create_collection("docs")
    memory_collection = client.get_or_create_collection("memory")

    while True:
        user_input = input("\nAsk your assistant something: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        query_emb = embed_query(user_input)
        context = retrieve_context(docs_collection, memory_collection, query_emb)

        prompt = f"You are a personal assistant.\n\nContext:\n{context}\nUser: {user_input}"
        answer = llm(prompt)
        print("\nAssistant:", answer)

        add_memory(memory_collection, f"User: {user_input}\nAssistant: {answer}")


if __name__ == "__main__":
    main()
