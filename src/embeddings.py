"""
HuggingFace embeddings interface for semantic search using sentence-transformers
"""
from sentence_transformers import SentenceTransformer
import torch
from typing import List
from config.settings import (
    EMBEDDING_MODEL, 
    EMBEDDING_DEVICE, 
    HF_CACHE_DIR,
    EMBEDDING_BATCH_SIZE
)


class HuggingFaceEmbeddings:
    """Wrapper for HuggingFace sentence-transformers models"""
    
    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.device = EMBEDDING_DEVICE if torch.cuda.is_available() else "cpu"
        
        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            cache_folder=str(HF_CACHE_DIR),
            device=self.device
        )
        
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✓ Embeddings loaded on {self.device} (dimension: {self.dimension})")
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query string
            
        Returns:
            List of floats representing the embedding
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embedding.tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents with batching.
        
        Args:
            texts: List of document strings
            
        Returns:
            List of embeddings
        """
        embeddings = self.model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings.tolist()


def get_embeddings():
    """
    Initialize HuggingFace embeddings.
    
    Returns:
        HuggingFaceEmbeddings: Configured embedding function
    """
    return HuggingFaceEmbeddings()


if __name__ == "__main__":
    # Test embeddings
    embeddings = get_embeddings()
    test_text = "This is a test document for enterprise RAG system."
    
    try:
        result = embeddings.embed_query(test_text)
        print(f"✓ Embeddings working! Vector dimension: {len(result)}")
        
        # Test batch embedding
        test_docs = [
            "First test document",
            "Second test document", 
            "Third test document"
        ]
        batch_results = embeddings.embed_documents(test_docs)
        print(f"✓ Batch embeddings working! Generated {len(batch_results)} vectors")
        
    except Exception as e:
        print(f"✗ Embeddings test failed: {e}")
