from pathlib import Path

_ROOT = Path(__file__).parent.parent

CHROMA_PATH = str(_ROOT / "chroma_db")
DATA_DIR = str(_ROOT / "docs")
GEMINI_MODEL = "gemini-2.5-flash"
