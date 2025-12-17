"""
PDF processing and chunking using PyMuPDF (fitz)
Original working implementation
"""
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def extract_and_chunk_pdf(pdf_path, chunk_size, chunk_overlap):
    """
    Extracts text from a single PDF and chunks it. 
    This function runs in a separate process.
    """
    try:
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text() + "\n"
        
        if text:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
            chunks = text_splitter.create_documents([text])
            # Add metadata to each chunk
            for chunk in chunks:
                chunk.metadata["source_file"] = os.path.basename(pdf_path)
            return chunks
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return []


def process_pdfs_in_directory_parallel(directory_path, chunk_size=1000, chunk_overlap=200):
    """
    Processes all PDF files in a given directory using multiprocessing.
    """
    pdf_files = glob.glob(os.path.join(directory_path, '*.pdf'))
    if not pdf_files:
        print(f"No PDF files found in '{directory_path}'.")
        return []

    print(f"Found {len(pdf_files)} PDFs to process.")
    all_chunks = []
    
    # Use ProcessPoolExecutor to manage a pool of worker processes
    # max_workers can be set to the number of CPU cores for optimal performance
    max_workers = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        future_to_pdf = {
            executor.submit(extract_and_chunk_pdf, pdf_file, chunk_size, chunk_overlap): pdf_file
            for pdf_file in pdf_files
        }
        
        # Process results as they complete
        for future in as_completed(future_to_pdf):
            pdf_file = future_to_pdf[future]
            try:
                chunks = future.result()
                if chunks:
                    all_chunks.extend(chunks)
                    print(f"  ✓ Processed '{os.path.basename(pdf_file)}' → {len(chunks)} chunks")
            except Exception as e:
                print(f"  ✗ Error retrieving results for {pdf_file}: {e}")

    return all_chunks


if __name__ == "__main__":
    # Test standalone with config
    from config.settings import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
    
    pdf_directory = DATA_DIR
    
    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory)
        print(f"Created directory '{pdf_directory}'. Please add your PDFs here.")
    else:
        final_chunks_list = process_pdfs_in_directory_parallel(
            pdf_directory,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        print(f"\n✓ Total chunks generated: {len(final_chunks_list)}")
        
        # Optional: Save chunks to text file for inspection
        if final_chunks_list:
            output_file = 'chunks_output.txt'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n\n---CHUNK SEPARATOR---\n\n'.join([
                    f"Source: {chunk.metadata['source_file']}\n\nContent:\n{chunk.page_content}" 
                    for chunk in final_chunks_list
                ]))
            print(f"✓ Chunks saved to {output_file}")
