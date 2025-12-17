# Quick Start Guide - HuggingFace RAG System

## Immediate Setup Steps

### 1. Install Dependencies (5-10 minutes)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install all required packages
pip install -r requirements.txt

# For CUDA support (GPU acceleration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Configure Your Models

Edit `config/settings.py` and choose your setup:

**For 8GB GPU (Recommended Starting Point):**
```python
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
LLM_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
USE_4BIT_QUANTIZATION = True
```

**For 16GB+ GPU:**
```python
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
USE_4BIT_QUANTIZATION = True
```

**For CPU Only (Slower):**
```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"
DEVICE = "cpu"
USE_4BIT_QUANTIZATION = False
```

### 3. First Run - Index Documents

```bash
# Place your PDFs in data/pdfs/ directory
mkdir -p data/pdfs
# Copy your PDFs into data/pdfs/

# Index all documents (first time will download models)
python main.py index
```

**First run will download models (~5-15 GB)**
- Models are cached in `hf_cache/` directory
- Subsequent runs are much faster

### 4. Query Your Documents

```bash
# Start interactive query mode
python main.py query
```

Example session:
```
Question: What are the key topics in these documents?
📝 Answer: [Your answer here based on documents]
📚 Sources: document1.pdf, document2.pdf
```

## Common Issues & Solutions

### 1. CUDA Out of Memory

**Solution 1 - Enable stronger quantization:**
```python
USE_4BIT_QUANTIZATION = True
```

**Solution 2 - Use smaller model:**
```python
LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"  # 3.8B params
```

**Solution 3 - Reduce context window:**
```python
TOP_K_RESULTS = 2  # Retrieve fewer chunks
```

### 2. Slow Performance

**Check GPU is being used:**
```python
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Shows your GPU
```

**If False:**
- Reinstall PyTorch with CUDA support
- Check NVIDIA drivers are installed
- Verify CUDA toolkit is installed

### 3. Model Download Fails

**For gated models (Llama, etc.):**
```bash
# Login to HuggingFace
huggingface-cli login
# Paste your token from huggingface.co/settings/tokens
```

**For network issues:**
- Check firewall settings
- Try downloading manually from huggingface.co
- Use mirror: `export HF_ENDPOINT=https://hf-mirror.com`

## Testing Individual Components

**Test embeddings only:**
```bash
python src/embeddings.py
```
Expected output: `✓ Embeddings working! Vector dimension: 1024`

**Test LLM only:**
```bash
python src/llm.py
```
Expected output: Generated response about RAG systems

**Test PDF processing:**
```bash
python src/pdf_chunker.py
```
Expected output: Chunk count from your PDFs

## Performance Benchmarks

### Expected speeds (with 4-bit quantization):

| Model Size | GPU        | Tokens/sec | Time per query |
|------------|------------|------------|----------------|
| 3B         | RTX 3060   | ~40-60     | 10-15s         |
| 7B         | RTX 3090   | ~30-40     | 15-20s         |
| 7B         | A100       | ~80-100    | 5-8s           |

## Next Steps

1. **Tune chunk size** - Experiment with `CHUNK_SIZE` (500-2000)
2. **Adjust retrieval** - Change `TOP_K_RESULTS` (1-10)
3. **Prompt engineering** - Modify prompt in `src/retriever.py`
4. **Try different models** - Test various LLMs for your use case
5. **Add more documents** - System scales to 1000s of PDFs

## Model Recommendations by Use Case

**Technical Documentation:**
- Embedding: `BAAI/bge-large-en-v1.5`
- LLM: `mistralai/Mistral-7B-Instruct-v0.2`

**General Q&A:**
- Embedding: `sentence-transformers/all-mpnet-base-v2`
- LLM: `meta-llama/Llama-3.2-3B-Instruct`

**Fast Inference (Production):**
- Embedding: `BAAI/bge-small-en-v1.5`
- LLM: `microsoft/Phi-3-mini-4k-instruct`

**Best Quality (Research):**
- Embedding: `BAAI/bge-large-en-v1.5`
- LLM: `mistralai/Mistral-7B-Instruct-v0.2` (or larger)

## Memory Usage Reference

```
RTX 3060 (12GB):  Use 3B models with 4-bit quant ✓
RTX 3090 (24GB):  Use 7B models with 4-bit quant ✓
RTX 4090 (24GB):  Use 7B models, can try 13B ✓
A100 (40GB):      Use any model up to 13B ✓
```

## Getting Help

1. Check `python main.py stats` to verify system state
2. Run individual test files to isolate issues
3. Review logs for specific error messages
4. Check VRAM usage: `nvidia-smi`
