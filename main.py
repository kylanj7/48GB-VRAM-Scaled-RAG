"""
Main application for HuggingFace-based RAG system
"""
import argparse
from pathlib import Path
from src.pdf_chunker import process_pdfs_in_directory_parallel
from src.vector_store import VectorStoreManager
from src.retriever import RAGRetriever
from config.settings import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def index_documents(pdf_directory: Path = None):
    """Index all PDFs in the data directory"""
    pdf_dir = pdf_directory or DATA_DIR
    
    if not pdf_dir.exists():
        pdf_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory '{pdf_dir}'. Please add your PDFs here.")
        return
    
    print("=" * 60)
    print("INDEXING DOCUMENTS")
    print("=" * 60)
    
    # Process PDFs
    chunks = process_pdfs_in_directory_parallel(
        str(pdf_dir),
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    if not chunks:
        print("No documents to index.")
        return
    
    # Add to vector store
    vs_manager = VectorStoreManager()
    vs_manager.create_or_load()
    vs_manager.add_documents(chunks)
    
    print(f"\n✓ Indexed {len(chunks)} chunks from {len(set(c.metadata['source_file'] for c in chunks))} files")


def query_system(interactive: bool = True):
    """Query the RAG system"""
    print("=" * 60)
    print("RAG QUERY SYSTEM")
    print("=" * 60)
    
    rag = RAGRetriever()
    
    if interactive:
        print("\nEnter your questions (type 'quit' to exit):\n")
        while True:
            try:
                query = input("Question: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                result = rag.retrieve_and_generate(query)
                
                print(f"\n📝 Answer:\n{result['answer']}")
                print(f"\n📚 Sources: {', '.join(result['sources'])}\n")
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    else:
        # Single query mode
        test_query = "What information is in these documents?"
        result = rag.retrieve_and_generate(test_query)
        
        print(f"\nQuery: {test_query}")
        print(f"\n📝 Answer:\n{result['answer']}")
        print(f"\n📚 Sources: {', '.join(result['sources'])}")


def show_stats():
    """Display vector store statistics"""
    print("=" * 60)
    print("VECTOR STORE STATISTICS")
    print("=" * 60)
    
    vs_manager = VectorStoreManager()
    vs_manager.create_or_load()
    
    count = vs_manager.get_collection_count()
    indexed_files = vs_manager.list_indexed_files()
    
    print(f"\nTotal chunks: {count}")
    print(f"Indexed files: {len(indexed_files)}")
    
    if indexed_files:
        print("\nFiles:")
        for filename in indexed_files:
            print(f"  - {filename}")


def reset_system():
    """Reset the vector store"""
    response = input("Are you sure you want to delete all indexed documents? (yes/no): ")
    
    if response.lower() == 'yes':
        vs_manager = VectorStoreManager()
        vs_manager.create_or_load()
        vs_manager.delete_collection()
        print("✓ System reset complete")
    else:
        print("Reset cancelled")


def main():
    parser = argparse.ArgumentParser(description="HuggingFace RAG System")
    parser.add_argument(
        'command',
        choices=['index', 'query', 'stats', 'reset'],
        help='Command to execute'
    )
    parser.add_argument(
        '--pdf-dir',
        type=str,
        help='Path to PDF directory (for index command)'
    )
    parser.add_argument(
        '--non-interactive',
        action='store_true',
        help='Run query in non-interactive mode'
    )
    
    args = parser.parse_args()
    
    if args.command == 'index':
        pdf_dir = Path(args.pdf_dir) if args.pdf_dir else None
        index_documents(pdf_dir)
    
    elif args.command == 'query':
        query_system(interactive=not args.non_interactive)
    
    elif args.command == 'stats':
        show_stats()
    
    elif args.command == 'reset':
        reset_system()


if __name__ == "__main__":
    main()
