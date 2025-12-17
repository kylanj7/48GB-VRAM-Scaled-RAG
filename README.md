# HuggingFace RAG System

Enterprise-grade RAG system using HuggingFace models with quantization support for efficient inference.

## Features

- **Advanced Embeddings**: Uses state-of-the-art sentence-transformers models
- **Powerful LLMs**: Supports Mistral, Llama, Gemma, and other HuggingFace models
- **Quantization**: 4-bit and 8-bit quantization for reduced memory usage
- **Multi-GPU Support**: Automatic device mapping for large models
- **Incremental Indexing**: Only processes new/changed documents
- **Parallel Processing**: Multi-core PDF processing

## Requirements

- Python 3.9+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ GPU VRAM (16GB+ recommended for larger models)
- 16GB+ system RAM

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install flash-attention for better performance (requires CUDA)
# pip install flash-attn --no-build-isolation
```

## Model Selection

### Embedding Models

Edit `config/settings.py` to choose your embedding model:

```python
# High quality (1024 dim, ~2GB VRAM)
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Balanced (768 dim, ~500MB VRAM)
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Lightweight (384 dim, ~150MB VRAM)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
```

### LLM Models

```python
# Strong performance (7B params, ~4GB VRAM with 4-bit quant)
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# Efficient (3B params, ~2GB VRAM with 4-bit quant)
LLM_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# Balanced (9B params, ~5GB VRAM with 4-bit quant)
LLM_MODEL = "google/gemma-2-9b-it"

# Very efficient (3.8B params, ~2GB VRAM with 4-bit quant)
LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"
```

### Quantization Settings

```python
# Enable 4-bit quantization (recommended)
USE_4BIT_QUANTIZATION = True

# Or use 8-bit quantization
USE_8BIT_QUANTIZATION = True

# Disable for full precision (requires more VRAM)
USE_4BIT_QUANTIZATION = False
USE_8BIT_QUANTIZATION = False
```

## Usage

### 1. Index Documents

Place PDFs in `data/pdfs/` directory, then:

```bash
python main.py index
```

Or specify a custom directory:

```bash
python main.py index --pdf-dir /path/to/pdfs
```

### 2. Query the System

Interactive mode:

```bash
python main.py query
```

Single query mode:

```bash
python main.py query --non-interactive
```

### 3. View Statistics

```bash
python main.py stats
```

### 4. Reset System

```bash
python main.py reset
```

## Project Structure

```
├── config/
│   └── settings.py          # Configuration settings
├── src/
│   ├── embeddings.py        # HuggingFace embeddings
│   ├── llm.py              # HuggingFace LLM interface
│   ├── pdf_chunker.py      # PDF processing
│   ├── vector_store.py     # ChromaDB management
│   └── retriever.py        # RAG pipeline
├── data/
│   └── pdfs/               # Place PDFs here
├── vectorstore/
│   └── chroma_db/          # Vector database
├── hf_cache/               # HuggingFace model cache
├── main.py                 # Main application
└── requirements.txt        # Dependencies
```

## Parameters Configuration

### Adjust Generation Parameters

In `config/settings.py`:

```python
MAX_NEW_TOKENS = 512      # Maximum output length
TEMPERATURE = 0.7         # Randomness (0.0-1.0)
TOP_P = 0.9              # Nucleus sampling
TOP_K_RESULTS = 3        # Chunks to retrieve
```

### Chunking Strategy

```python
CHUNK_SIZE = 1000        # Characters per chunk
CHUNK_OVERLAP = 200      # Overlap between chunks
```

### Device Selection

```python
DEVICE = "cuda"          # "cuda" or "cpu"
EMBEDDING_DEVICE = "cuda:0"  # Specific GPU
LLM_DEVICE = "cuda:1"    # Use different GPU for LLM
```

## Performance Tips

1. **Use 4-bit quantization** for the best memory/quality trade-off
2. **Batch embeddings** are processed automatically for efficiency
3. **Multi-GPU setup**: Assign embeddings and LLM to different GPUs
4. **Flash Attention**: Install for 2-4x speed improvement
5. **Model selection**: Start with 3B models, scale up as needed

## Troubleshooting

### Out of Memory (OOM)

- Enable 4-bit quantization
- Use a smaller model (e.g., Phi-3-mini)
- Reduce `CHUNK_SIZE` and `TOP_K_RESULTS`
- Close other GPU applications

### Slow Generation

- Ensure CUDA is properly installed
- Check `torch.cuda.is_available()`
- Consider using flash-attention
- Verify model is on GPU: check device mapping

### Model Download Issues

- Check internet connection
- Set HuggingFace token for gated models:
  ```bash
  huggingface-cli login
  ```

## Testing Individual Components

```bash
# Test embeddings
python src/embeddings.py

# Test LLM
python src/llm.py

# Test PDF processing
python src/pdf_chunker.py

# Test retriever
python src/retriever.py
```

## License

MIT License - feel free to use for any purpose.
