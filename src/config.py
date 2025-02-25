import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any, List

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Lean 4 settings
LEAN4_GITHUB_REPO = "https://github.com/leanprover/lean4"
LEAN4_DOCS_URL = "https://lean-lang.org/lean4/doc"
LEAN4_DOCS_URLS = [
    "https://lean-lang.org/lean4/doc/",
    # Add more specific documentation URLs here
]

# Model settings
DEFAULT_EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_SMALL_LLM = "qwen2.5:14b-instruct-q4_K_M"  # Smaller model
DEFAULT_REASONER_LLM = "deepseek-r1:8b"  # Reasoner model
OLLAMA_BASE_URL = "http://localhost:11434"

# RAG settings
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_RETRIEVE_K = 3

# Logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)