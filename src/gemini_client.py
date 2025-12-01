import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel('gemini-2.5-flash')

def get_gemini_llm():
    """Return a function that generates text given a prompt."""
    def llm(prompt: str):
        response = model.generate_content(prompt)
        return response._result.candidates[0].content.parts[0].text
    return llm
