"""Tests for src/gemini_client.py"""
from unittest.mock import MagicMock, patch
import pytest


def make_mock_response(text: str = "Mocked response") -> MagicMock:
    mock_response = MagicMock()
    mock_response.text = text
    return mock_response


class TestGetModel:

    @patch("gemini_client.genai")
    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    def test_configures_genai_with_api_key(self, mock_genai):
        import gemini_client
        gemini_client._model = None

        gemini_client._get_model()

        mock_genai.configure.assert_called_once_with(api_key="test-key")
        gemini_client._model = None  # cleanup

    @patch("gemini_client.genai")
    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    def test_caches_model_after_first_call(self, mock_genai):
        import gemini_client
        gemini_client._model = None

        model1 = gemini_client._get_model()
        model2 = gemini_client._get_model()

        assert model1 is model2
        mock_genai.configure.assert_called_once()  # not called twice
        gemini_client._model = None  # cleanup


class TestGetGeminiLlm:

    @patch("gemini_client._get_model")
    def test_returns_callable(self, mock_get_model):
        mock_get_model.return_value.generate_content.return_value = make_mock_response()
        from gemini_client import get_gemini_llm

        llm = get_gemini_llm()

        assert callable(llm)

    @patch("gemini_client._get_model")
    def test_llm_returns_string(self, mock_get_model):
        mock_get_model.return_value.generate_content.return_value = make_mock_response("Hello!")
        from gemini_client import get_gemini_llm

        llm = get_gemini_llm()
        result = llm("Say hello.")

        assert isinstance(result, str)
        assert result == "Hello!"

    @patch("gemini_client._get_model")
    def test_llm_passes_prompt_to_api(self, mock_get_model):
        mock_get_model.return_value.generate_content.return_value = make_mock_response()
        from gemini_client import get_gemini_llm

        llm = get_gemini_llm()
        llm("What is RAG?")

        mock_get_model.return_value.generate_content.assert_called_once_with("What is RAG?")

    @patch("gemini_client._get_model")
    def test_llm_raises_on_api_error(self, mock_get_model):
        mock_get_model.return_value.generate_content.side_effect = Exception("Quota exceeded")
        from gemini_client import get_gemini_llm

        llm = get_gemini_llm()

        with pytest.raises(Exception, match="Quota exceeded"):
            llm("Will this fail?")

    @patch.dict("os.environ", {}, clear=True)
    @patch("gemini_client._model", None)
    def test_raises_when_api_key_missing(self):
        import gemini_client
        gemini_client._model = None
        from gemini_client import _get_model

        with pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
            _get_model()
