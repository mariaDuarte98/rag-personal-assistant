import sys
import os
from unittest.mock import MagicMock

sys.modules['sentence_transformers'] = MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))