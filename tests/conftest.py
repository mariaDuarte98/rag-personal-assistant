import sys
import os

# Adds src/ to the Python path so tests can import modules from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))