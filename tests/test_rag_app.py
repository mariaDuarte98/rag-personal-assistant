"""Tests for src/rag_app.py"""
from unittest.mock import MagicMock, patch


def make_mock_collection(
    docs: list[str] | None = None,
    ids: list[str] | None = None,
    metadatas: list[dict] | None = None,
):
    collection = MagicMock()
    collection.get.return_value = {"ids": ids or []}
    _docs = docs or []
    _metas = metadatas or [{}] * len(_docs)
    collection.query.return_value = {
        "documents": [_docs],
        "metadatas": [_metas],
        "distances": [[0.1] * len(_docs)],
        "ids": [[f"id_{i}" for i in range(len(_docs))]],
    }
    return collection


class TestEmbedQuery:

    @patch("rag_app.get_embedding")
    def test_returns_embedding_for_query(self, mock_get_embedding):
        mock_get_embedding.return_value = [0.1, 0.2, 0.3]
        from rag_app import embed_query

        result = embed_query("What is RAG?")

        mock_get_embedding.assert_called_once_with("What is RAG?")
        assert result == [0.1, 0.2, 0.3]


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

    def test_queries_both_collections(self):
        docs_collection = make_mock_collection()
        memory_collection = make_mock_collection()
        from rag_app import retrieve_context

        retrieve_context(docs_collection, memory_collection, [0.1, 0.2])

        docs_collection.get.assert_called_once()
        docs_collection.query.assert_not_called()
        memory_collection.get.assert_called_once()
        memory_collection.query.assert_not_called()
