import uuid
import chromadb
from embeddings import get_embedding
from gemini_client import get_gemini_llm
from config import CHROMA_PATH

MAX_HISTORY = 5

_SYSTEM_PROMPT = """You are a knowledgeable personal assistant with access to a curated document library and memory of past conversations.

When answering:
- Use the provided context to give accurate, grounded answers
- If the context does not contain the information needed, say so clearly — do not guess
- Keep answers concise and directly relevant to the question
- When referencing specific information, mention the source document"""


def add_memory(memory_collection, text: str, memory_id: str | None = None) -> str:
    """Persist a conversation turn. Returns the id used."""
    if memory_id is None:
        memory_id = f"memory-{uuid.uuid4().hex}"
    emb = get_embedding(text)
    memory_collection.add(documents=[text], embeddings=[emb], ids=[memory_id])
    return memory_id


def retrieve_context(docs_collection, memory_collection, query_emb: list[float]) -> str:
    context = ""

    n_docs = docs_collection.count()
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

    n_mem = memory_collection.count()
    if n_mem > 0:
        results = memory_collection.query(
            query_embeddings=[query_emb],
            n_results=min(2, n_mem),
        )
        context += "\n### From past conversations:\n"
        for mem in results["documents"][0]:
            context += mem + "\n---\n"

    return context


def build_prompt(query: str, context: str, history: list[dict[str, str]]) -> str:
    history_text = "\n\n".join(
        f"User: {turn['user']}\nAssistant: {turn['assistant']}"
        for turn in history
    ) if history else "No previous conversation."

    context_text = context.strip() if context.strip() else "No relevant context found."

    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"### Relevant context\n{context_text}\n\n"
        f"### Recent conversation\n{history_text}\n\n"
        f"### Current question\n{query}"
    )


def main():
    llm = get_gemini_llm()
    client = chromadb.PersistentClient(CHROMA_PATH)

    docs_collection = client.get_or_create_collection("docs")
    memory_collection = client.get_or_create_collection("memory")

    history: list[dict[str, str]] = []

    while True:
        user_input = input("\nAsk your assistant something: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        query_emb = get_embedding(user_input)
        context = retrieve_context(docs_collection, memory_collection, query_emb)
        prompt = build_prompt(user_input, context, history[-MAX_HISTORY:])

        try:
            answer = llm(prompt)
        except Exception as e:
            print(f"\n[Error contacting Gemini: {e}. Please try again.]")
            continue

        print("\nAssistant:", answer)

        history.append({"user": user_input, "assistant": answer})
        add_memory(memory_collection, f"User: {user_input}\nAssistant: {answer}")


if __name__ == "__main__":
    main()
