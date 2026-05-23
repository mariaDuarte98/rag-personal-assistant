"""Tests for src/rag_app.py"""
from unittest.mock import MagicMock, patch


def make_mock_collection(
    docs: list[str] | None = None,
    ids: list[str] | None = None,
    metadatas: list[dict] | None = None,
):
    collection = MagicMock()
    _docs = docs or []
    _ids = ids or []
    _metas = metadatas or [{}] * len(_docs)
    collection.count.return_value = len(_docs)
    collection.query.return_value = {
        "documents": [_docs],
        "metadatas": [_metas],
        "distances": [[0.1] * len(_docs)],
        "ids": [[f"id_{i}" for i in range(len(_docs))]],
    }
    return collection


class TestAddMemory:

    @patch("rag_app.get_embedding")
    def test_adds_to_memory_collection(self, mock_get_embedding):
        mock_get_embedding.return_value = [0.5, 0.6]
        mock_collection = MagicMock()
        from rag_app import add_memory

        add_memory(mock_collection, "User: hi\nAssistant: hello", "memory-0")

        mock_collection.add.assert_called_once_with(
            documents=["User: hi\nAssistant: hello"],
            embeddings=[[0.5, 0.6]],
            ids=["memory-0"],
        )

    @patch("rag_app.get_embedding")
    def test_uses_provided_memory_id(self, mock_get_embedding):
        mock_get_embedding.return_value = [0.1]
        mock_collection = MagicMock()
        from rag_app import add_memory

        add_memory(mock_collection, "some text", "memory-42")

        call_kwargs = mock_collection.add.call_args.kwargs
        assert call_kwargs["ids"] == ["memory-42"]

    @patch("rag_app.get_embedding")
    def test_auto_generates_unique_memory_id(self, mock_get_embedding):
        mock_get_embedding.return_value = [0.1]
        mock_collection = MagicMock()
        from rag_app import add_memory

        id1 = add_memory(mock_collection, "first turn")
        id2 = add_memory(mock_collection, "second turn")

        assert id1.startswith("memory-")
        assert id2.startswith("memory-")
        assert id1 != id2

    @patch("rag_app.get_embedding")
    def test_returns_memory_id(self, mock_get_embedding):
        mock_get_embedding.return_value = [0.1]
        mock_collection = MagicMock()
        from rag_app import add_memory

        returned_id = add_memory(mock_collection, "some text", "memory-99")

        assert returned_id == "memory-99"


class TestBuildPrompt:

    def test_includes_query_in_prompt(self):
        from rag_app import build_prompt
        result = build_prompt("What is RAG?", "", [])
        assert "What is RAG?" in result

    def test_includes_context_in_prompt(self):
        from rag_app import build_prompt
        result = build_prompt("query", "Some document context.", [])
        assert "Some document context." in result

    def test_includes_history_in_prompt(self):
        from rag_app import build_prompt
        history = [{"user": "hello", "assistant": "hi there"}]
        result = build_prompt("next question", "context", history)
        assert "hello" in result
        assert "hi there" in result

    def test_empty_context_shows_fallback(self):
        from rag_app import build_prompt
        result = build_prompt("query", "", [])
        assert "No relevant context found." in result

    def test_empty_history_shows_fallback(self):
        from rag_app import build_prompt
        result = build_prompt("query", "some context", [])
        assert "No previous conversation." in result

    def test_multiple_history_turns_all_included(self):
        from rag_app import build_prompt
        history = [
            {"user": "first", "assistant": "answer one"},
            {"user": "second", "assistant": "answer two"},
        ]
        result = build_prompt("third", "ctx", history)
        assert "first" in result
        assert "second" in result
        assert "answer one" in result
        assert "answer two" in result

    def test_system_prompt_encourages_own_knowledge(self):
        from rag_app import _SYSTEM_PROMPT
        assert "own knowledge" in _SYSTEM_PROMPT.lower()

    def test_system_prompt_does_not_refuse_guessing(self):
        from rag_app import _SYSTEM_PROMPT
        assert "do not guess" not in _SYSTEM_PROMPT.lower()

    def test_system_prompt_does_not_demand_explicit_admission(self):
        from rag_app import _SYSTEM_PROMPT
        assert "say so clearly" not in _SYSTEM_PROMPT.lower()


class TestRetrieveContext:

    def test_includes_docs_in_context(self):
        docs_collection = make_mock_collection(
            docs=["AI is transforming industries."],
            ids=["id_0"],
            metadatas=[{"source": "ai.txt", "chunk_index": 0}],
        )
        memory_collection = make_mock_collection()
        from rag_app import retrieve_context

        context = retrieve_context(docs_collection, memory_collection, [0.1, 0.2])

        assert "AI is transforming industries." in context
        assert "From your documents" in context

    def test_includes_source_in_context(self):
        docs_collection = make_mock_collection(
            docs=["Some content."],
            ids=["id_0"],
            metadatas=[{"source": "notes.txt", "chunk_index": 0}],
        )
        memory_collection = make_mock_collection()
        from rag_app import retrieve_context

        context = retrieve_context(docs_collection, memory_collection, [0.1, 0.2])

        assert "notes.txt" in context

    def test_includes_memory_in_context(self):
        docs_collection = make_mock_collection()
        memory_collection = make_mock_collection(
            docs=["User: hello\nAssistant: hi!"], ids=["id_0"]
        )
        from rag_app import retrieve_context

        context = retrieve_context(docs_collection, memory_collection, [0.1, 0.2])

        assert "User: hello" in context
        assert "From past conversations" in context

    def test_combines_docs_and_memory(self):
        docs_collection = make_mock_collection(
            docs=["Document content."],
            ids=["id_0"],
            metadatas=[{"source": "doc.txt", "chunk_index": 0}],
        )
        memory_collection = make_mock_collection(
            docs=["Past conversation."], ids=["id_0"]
        )
        from rag_app import retrieve_context

        context = retrieve_context(docs_collection, memory_collection, [0.1, 0.2])

        assert "Document content." in context
        assert "Past conversation." in context

    def test_returns_empty_string_when_both_collections_empty(self):
        docs_collection = make_mock_collection()
        memory_collection = make_mock_collection()
        from rag_app import retrieve_context

        context = retrieve_context(docs_collection, memory_collection, [0.1, 0.2])

        assert context == ""

    def test_skips_query_when_collections_empty(self):
        docs_collection = make_mock_collection()
        memory_collection = make_mock_collection()
        from rag_app import retrieve_context

        retrieve_context(docs_collection, memory_collection, [0.1, 0.2])

        docs_collection.count.assert_called_once()
        docs_collection.query.assert_not_called()
        memory_collection.count.assert_called_once()
        memory_collection.query.assert_not_called()


class TestMain:

    @patch("rag_app.get_embedding", return_value=[0.1, 0.2])
    @patch("rag_app.get_gemini_llm")
    @patch("rag_app.chromadb")
    @patch("builtins.input", side_effect=["how many bones in the body?", "exit"])
    def test_main_runs_conversation_and_prints_answer(self, mock_input, mock_chromadb, mock_get_llm, mock_embed, capsys):
        mock_llm = MagicMock(return_value="The human body has 206 bones.")
        mock_get_llm.return_value = mock_llm
        mock_collection = make_mock_collection()
        mock_chromadb.PersistentClient.return_value.get_or_create_collection.return_value = mock_collection

        from rag_app import main
        main()

        mock_llm.assert_called_once()
        assert "206 bones" in capsys.readouterr().out

    @patch("rag_app.get_embedding", return_value=[0.1, 0.2])
    @patch("rag_app.get_gemini_llm")
    @patch("rag_app.chromadb")
    @patch("builtins.input", side_effect=["bad question", "exit"])
    def test_main_handles_llm_error_gracefully(self, mock_input, mock_chromadb, mock_get_llm, mock_embed, capsys):
        mock_llm = MagicMock(side_effect=Exception("quota exceeded"))
        mock_get_llm.return_value = mock_llm
        mock_collection = make_mock_collection()
        mock_chromadb.PersistentClient.return_value.get_or_create_collection.return_value = mock_collection

        from rag_app import main
        main()  # must not raise

        assert "Error contacting Gemini" in capsys.readouterr().out
