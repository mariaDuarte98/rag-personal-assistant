from sentence_transformers import SentenceTransformer
from typing import List

# Initialize the local embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str) -> List[float]:
    """Generate embedding for a given text."""
    return embed_model.encode(text).tolist()
