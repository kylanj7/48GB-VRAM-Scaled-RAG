"""
RAG retriever: combines vector search with HuggingFace LLM generation
"""
from src.llm import get_llm
from src.vector_store import VectorStoreManager
from config.settings import TOP_K_RESULTS, LLM_MODEL


class RAGRetriever:
    """Main RAG pipeline for question answering"""
    
    def __init__(self):
        print("Initializing RAG Retriever...")
        self.vs_manager = VectorStoreManager()
        self.vs_manager.create_or_load()
        
        self.llm = get_llm()
        print(f"✓ RAG Retriever initialized with {LLM_MODEL}")
    
    def retrieve_and_generate(self, query: str, k: int = None):
        """
        RAG pipeline: retrieve context and generate answer.
        
        Args:
            query: User question
            k: Number of chunks to retrieve (default: TOP_K_RESULTS)
            
        Returns:
            Dict with 'answer' and 'sources'
        """
        k = k or TOP_K_RESULTS
        print(f"\n📝 Query: {query}")
        
        # Retrieve relevant chunks
        docs = self.vs_manager.similarity_search(query, k=k)
        
        if not docs:
            return {
                "answer": "No relevant documents found in the knowledge base.",
                "sources": []
            }
        
        print(f"  Retrieved {len(docs)} relevant chunks")
        
        # Build context from retrieved chunks
        context = "\n\n".join([
            f"[Source: {doc.metadata.get('source_file', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        ])
        
        # Construct prompt with instruction format
        prompt = self._format_prompt(query, context)
        
        # Generate response
        print("  Generating answer...")
        response = self.llm.generate(prompt)
        
        sources = list(set([doc.metadata.get('source_file', 'unknown') for doc in docs]))
        
        return {
            "answer": response,
            "sources": sources
        }
    
    def _format_prompt(self, query: str, context: str) -> str:
        """
        Format prompt for the LLM based on model type.
        Adjust this for different instruction formats.
        """
        # Generic instruction format (works for most models)
        prompt = f"""[INST] You are a helpful assistant that answers questions based on provided context.

Context:
{context}

Question: {query}

Provide a clear, accurate answer based only on the context above. If the context doesn't contain enough information to answer the question, say so. [/INST]

Answer:"""
        
        return prompt


if __name__ == "__main__":
    # Test retriever
    print("Testing RAG Retriever...")
    rag = RAGRetriever()
    
    test_query = "What is this document about?"
    result = rag.retrieve_and_generate(test_query)
    
    print(f"\n✓ Answer:\n{result['answer']}")
    print(f"\n📚 Sources: {', '.join(result['sources'])}")
