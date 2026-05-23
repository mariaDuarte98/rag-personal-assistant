import sys
import pytest
from unittest.mock import MagicMock

sys.modules['sentence_transformers'] = MagicMock()


@pytest.fixture(autouse=True)
def reset_embedding_model():
    import embeddings
    embeddings._model = None
    yield
    embeddings._model = None


@pytest.fixture(autouse=True)
def reset_gemini_model():
    import gemini_client
    gemini_client._model = None
    yield
    gemini_client._model = None
