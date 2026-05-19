"""Tests for src/ingest.py"""
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock


class TestChunkText:

    def test_short_text_returns_single_chunk(self):
        from ingest import chunk_text
        result = chunk_text("Hello world", chunk_size=500)
        assert result == ["Hello world"]

    def test_empty_text_returns_empty_list(self):
        from ingest import chunk_text
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_long_text_is_split(self):
        from ingest import chunk_text
        text = "a" * 1200
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) > 1

    def test_chunks_have_correct_size(self):
        from ingest import chunk_text
        text = "x" * 1000
        chunks = chunk_text(text, chunk_size=300, overlap=0)
        for chunk in chunks[:-1]:
            assert len(chunk) == 300

    def test_chunks_overlap(self):
        from ingest import chunk_text
        text = "abcdefghij"
        chunks = chunk_text(text, chunk_size=6, overlap=2)
        assert chunks[0][-2:] == chunks[1][:2]

    def test_full_text_is_covered(self):
        from ingest import chunk_text
        text = "Hello, this is a test of the chunking function."
        chunks = chunk_text(text, chunk_size=10, overlap=2)
        assert text.startswith(chunks[0])
        assert text.endswith(chunks[-1])


class TestLoadDocuments:

    def test_loads_txt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "a.txt"), "w").write("Document A")
            open(os.path.join(tmpdir, "b.txt"), "w").write("Document B")
            from ingest import load_documents
            docs = load_documents(tmpdir)
        assert len(docs) == 2
        assert {d["id"] for d in docs} == {"a.txt", "b.txt"}

    def test_ignores_non_txt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "notes.txt"), "w").write("Keep me")
            open(os.path.join(tmpdir, "image.png"), "w").write("Ignore me")
            from ingest import load_documents
            docs = load_documents(tmpdir)
        assert len(docs) == 1
        assert docs[0]["id"] == "notes.txt"

    def test_returns_correct_text_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "doc.txt"), "w").write("Hello RAG world")
            from ingest import load_documents
            docs = load_documents(tmpdir)
        assert docs[0]["text"] == "Hello RAG world"

    def test_returns_empty_list_when_no_txt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from ingest import load_documents
            docs = load_documents(tmpdir)
        assert docs == []

    def test_raises_if_directory_does_not_exist(self):
        from ingest import load_documents
        with pytest.raises(FileNotFoundError, match="Documents directory not found"):
            load_documents("/nonexistent/path/that/does/not/exist")


class TestIngestUsesUpsert:

    @patch("ingest.get_embedding")
    @patch("ingest.load_documents")
    @patch("ingest.chromadb.PersistentClient")
    def test_upsert_called_with_metadata(self, mock_client, mock_load, mock_embed):
        mock_embed.return_value = [0.1, 0.2]
        mock_load.return_value = [{"id": "notes.txt", "text": "Hello world"}]
        mock_collection = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        from ingest import main
        main()

        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args.kwargs
        assert call_kwargs["metadatas"][0]["source"] == "notes.txt"
        assert call_kwargs["metadatas"][0]["chunk_index"] == 0

    @patch("ingest.get_embedding")
    @patch("ingest.load_documents")
    @patch("ingest.chromadb.PersistentClient")
    def test_reingest_does_not_raise(self, mock_client, mock_load, mock_embed):
        mock_embed.return_value = [0.1, 0.2]
        mock_load.return_value = [{"id": "notes.txt", "text": "Hello world"}]
        mock_collection = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        from ingest import main
        main()
        main()

        assert mock_collection.upsert.call_count == 2
