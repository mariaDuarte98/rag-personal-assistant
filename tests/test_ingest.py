"""Tests for src/ingest.py"""
import os
import tempfile
from unittest.mock import patch


class TestLoadDocuments:

    def test_loads_txt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "a.txt"), "w").write("Document A")
            open(os.path.join(tmpdir, "b.txt"), "w").write("Document B")

            import ingest as ingest_module
            original = ingest_module.DATA_DIR
            ingest_module.DATA_DIR = tmpdir
            try:
                docs = ingest_module.load_documents()
            finally:
                ingest_module.DATA_DIR = original

        assert len(docs) == 2
        assert {d["id"] for d in docs} == {"a.txt", "b.txt"}

    def test_ignores_non_txt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "notes.txt"), "w").write("Keep me")
            open(os.path.join(tmpdir, "image.png"), "w").write("Ignore me")

            import ingest as ingest_module
            original = ingest_module.DATA_DIR
            ingest_module.DATA_DIR = tmpdir
            try:
                docs = ingest_module.load_documents()
            finally:
                ingest_module.DATA_DIR = original

        assert len(docs) == 1
        assert docs[0]["id"] == "notes.txt"

    def test_returns_correct_text_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "doc.txt"), "w").write("Hello RAG world")

            import ingest as ingest_module
            original = ingest_module.DATA_DIR
            ingest_module.DATA_DIR = tmpdir
            try:
                docs = ingest_module.load_documents()
            finally:
                ingest_module.DATA_DIR = original

        assert docs[0]["text"] == "Hello RAG world"

    def test_returns_empty_list_when_no_txt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import ingest as ingest_module
            original = ingest_module.DATA_DIR
            ingest_module.DATA_DIR = tmpdir
            try:
                docs = ingest_module.load_documents()
            finally:
                ingest_module.DATA_DIR = original

        assert docs == []


class TestEmbedText:

    @patch("ingest.get_embedding")
    def test_delegates_to_get_embedding(self, mock_get_embedding):
        mock_get_embedding.return_value = [0.1, 0.2, 0.3]
        from ingest import embed_text

        result = embed_text("some text")

        mock_get_embedding.assert_called_once_with("some text")
        assert result == [0.1, 0.2, 0.3]