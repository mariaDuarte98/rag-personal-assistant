"""Tests for src/embeddings.py"""
from unittest.mock import patch
import numpy as np


class TestGetEmbedding:

    @patch("embeddings.embed_model")
    def test_returns_list_of_floats(self, mock_model):
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        from embeddings import get_embedding

        result = get_embedding("Hello world")

        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    @patch("embeddings.embed_model")
    def test_output_is_not_empty(self, mock_model):
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        from embeddings import get_embedding

        result = get_embedding("Some text")

        assert len(result) > 0

    @patch("embeddings.embed_model")
    def test_calls_encode_with_input_text(self, mock_model):
        mock_model.encode.return_value = np.array([0.5, 0.5])
        from embeddings import get_embedding

        get_embedding("test input")

        mock_model.encode.assert_called_once_with("test input")

    @patch("embeddings.embed_model")
    def test_different_texts_produce_different_embeddings(self, mock_model):
        mock_model.encode.side_effect = [np.array([0.1, 0.2]), np.array([0.9, 0.8])]
        from embeddings import get_embedding

        emb1 = get_embedding("cat")
        emb2 = get_embedding("dog")

        assert emb1 != emb2