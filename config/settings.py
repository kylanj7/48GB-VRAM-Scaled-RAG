"""
Configuration settings for HuggingFace-based RAG System
"""
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "pdfs"
VECTOR_STORE_DIR = BASE_DIR / "vectorstore" / "chroma_db"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# LLM options:
LLM_MODEL = "nvidia/Llama-Nemotron-70B-Instruct" 
EMBEDDING_MODEL = "nvidia/NV-Embed-v2"

# Hardware settings
DEVICE = "cuda"  # or "cpu" for CPU-only inference
EMBEDDING_DEVICE = "cuda"  # Separate device for embeddings if needed
LLM_DEVICE = "cuda"
DEVICE_MAP = "auto"
MAX_MEMORY = {0: "22GB", 1: "22GB"} 

# Model loading settings
USE_4BIT_QUANTIZATION = True  # Enable for lower memory usage
USE_8BIT_QUANTIZATION = False  # Alternative to 4-bit
LOAD_IN_4BIT = True  # For LLM loading
TRUST_REMOTE_CODE = True  # Required for some models

# Vector store settings
COLLECTION_NAME = "database_docs"
TOP_K_RESULTS = 3

# LLM generation settings
MAX_NEW_TOKENS = 512  # Maximum tokens to generate
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True

# Batch processing
EMBEDDING_BATCH_SIZE = 32
LLM_BATCH_SIZE = 1

# HuggingFace cache directory (optional)
HF_CACHE_DIR = BASE_DIR / "hf_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Logging
LOG_LEVEL = "INFO"
