"""Tests for src/embeddings.py"""
from unittest.mock import patch
import numpy as np


class TestGetModel:

    def test_lazy_loads_model_on_first_call(self):
        import embeddings
        embeddings._model = None

        model = embeddings._get_model()

        assert model is not None
        embeddings._model = None  # cleanup

    def test_caches_model_after_first_call(self):
        import embeddings
        embeddings._model = None

        model1 = embeddings._get_model()
        model2 = embeddings._get_model()

        assert model1 is model2
        embeddings._model = None  # cleanup


class TestGetEmbedding:

    @patch("embeddings._get_model")
    def test_returns_list_of_floats(self, mock_get_model):
        mock_get_model.return_value.encode.return_value = np.array([0.1, 0.2, 0.3])
        from embeddings import get_embedding

        result = get_embedding("Hello world")

        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    @patch("embeddings._get_model")
    def test_output_is_not_empty(self, mock_get_model):
        mock_get_model.return_value.encode.return_value = np.array([0.1, 0.2, 0.3])
        from embeddings import get_embedding

        result = get_embedding("Some text")

        assert len(result) > 0

    @patch("embeddings._get_model")
    def test_calls_encode_with_input_text(self, mock_get_model):
        mock_get_model.return_value.encode.return_value = np.array([0.5, 0.5])
        from embeddings import get_embedding

        get_embedding("test input")

        mock_get_model.return_value.encode.assert_called_once_with("test input")

    @patch("embeddings._get_model")
    def test_different_texts_produce_different_embeddings(self, mock_get_model):
        mock_get_model.return_value.encode.side_effect = [np.array([0.1, 0.2]), np.array([0.9, 0.8])]
        from embeddings import get_embedding

        emb1 = get_embedding("cat")
        emb2 = get_embedding("dog")

        assert emb1 != emb2
