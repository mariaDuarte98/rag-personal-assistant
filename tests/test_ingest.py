"""Tests for src/ingest.py"""
import pytest
from unittest.mock import patch, MagicMock


class TestChunkText:

    def test_short_text_returns_single_chunk(self):
        from ingest import chunk_text
        assert chunk_text("Hello world", chunk_size=500) == ["Hello world"]

    @pytest.mark.parametrize("text", ["", "   ", "\n", "\t"])
    def test_blank_text_returns_empty_list(self, text):
        from ingest import chunk_text
        assert chunk_text(text) == []

    def test_long_text_is_split(self):
        from ingest import chunk_text
        chunks = chunk_text("a" * 1200, chunk_size=500, overlap=50)
        assert len(chunks) > 1

    def test_chunks_have_correct_size(self):
        from ingest import chunk_text
        chunks = chunk_text("x" * 1000, chunk_size=300, overlap=0)
        for chunk in chunks[:-1]:
            assert len(chunk) == 300

    def test_chunks_overlap(self):
        from ingest import chunk_text
        chunks = chunk_text("abcdefghij", chunk_size=6, overlap=2)
        assert chunks[0][-2:] == chunks[1][:2]

    def test_full_text_is_covered(self):
        from ingest import chunk_text
        text = "Hello, this is a test of the chunking function."
        chunks = chunk_text(text, chunk_size=10, overlap=2)
        assert text.startswith(chunks[0])
        assert text.endswith(chunks[-1])


class TestLoadDocuments:

    def test_loads_txt_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("Document A")
        (tmp_path / "b.txt").write_text("Document B")
        from ingest import load_documents

        docs = load_documents(str(tmp_path))

        assert len(docs) == 2
        assert {d["id"] for d in docs} == {"a.txt", "b.txt"}

    @pytest.mark.parametrize("filename", ["image.png", "data.csv", "notes.md"])
    def test_ignores_non_txt_files(self, tmp_path, filename):
        (tmp_path / "notes.txt").write_text("Keep me")
        (tmp_path / filename).write_text("Ignore me")
        from ingest import load_documents

        docs = load_documents(str(tmp_path))

        assert len(docs) == 1
        assert docs[0]["id"] == "notes.txt"

    def test_returns_correct_text_content(self, tmp_path):
        (tmp_path / "doc.txt").write_text("Hello RAG world")
        from ingest import load_documents

        docs = load_documents(str(tmp_path))

        assert docs[0]["text"] == "Hello RAG world"

    def test_returns_empty_list_when_no_txt_files(self, tmp_path):
        from ingest import load_documents
        assert load_documents(str(tmp_path)) == []

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

    @patch("ingest.load_documents", return_value=[])
    @patch("ingest.chromadb.PersistentClient")
    def test_main_exits_early_when_no_documents(self, mock_client, mock_load, capsys):
        mock_collection = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        from ingest import main
        main()

        mock_collection.upsert.assert_not_called()
        assert "No .txt files found" in capsys.readouterr().out

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
