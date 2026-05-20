import os
from dotenv import load_dotenv
import google.generativeai as genai
from config import GEMINI_MODEL

load_dotenv()

_model = None


def _get_model():
    global _model
    if _model is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable is not set")
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel(GEMINI_MODEL)
    return _model


def get_gemini_llm():
    def llm(prompt: str) -> str:
        response = _get_model().generate_content(prompt)
        return response.text
    return llm
