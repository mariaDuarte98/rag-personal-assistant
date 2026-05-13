import chromadb
from embeddings import get_embedding
from gemini_client import get_gemini_llm

CHROMA_PATH = "chroma_db"


def embed_query(query: str) -> list[float]:
    return get_embedding(query)


def add_memory(memory_collection, text: str, memory_id: str) -> None:
    """Persist a conversation turn to the memory collection."""
    emb = get_embedding(text)
    memory_collection.add(documents=[text], embeddings=[emb], ids=[memory_id])


def retrieve_context(docs_collection, memory_collection, query_emb: list[float]) -> str:
    """Fetch relevant chunks from both docs and memory collections."""
    context = ""

    docs_results = docs_collection.query(query_embeddings=[query_emb], n_results=3)
    if docs_results["documents"] and docs_results["documents"][0]:
        context += "### From your documents:\n"
        for doc in docs_results["documents"][0]:
            context += doc + "\n---\n"

    memory_results = memory_collection.query(query_embeddings=[query_emb], n_results=2)
    if memory_results["documents"] and memory_results["documents"][0]:
        context += "\n### From past conversations:\n"
        for mem in memory_results["documents"][0]:
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

        full_memory = f"User: {user_input}\nAssistant: {answer}"
        memory_id = f"memory-{len(memory_collection.get()['ids'])}"
        add_memory(memory_collection, full_memory, memory_id)


if __name__ == "__main__":
    main()