import os
import sys
from pathlib import Path
import argparse
import logging
from typing import List, Optional, Dict, Any, Union
import json
import time
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    RAW_DATA_DIR, 
    PROCESSED_DATA_DIR, 
    VECTOR_STORE_DIR,
    DEFAULT_EMBEDDINGS_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP
)
from src.data.loader import LeanLoader, LeanProcessor
from src.data.processor import LeanDataProcessor
from src.models.embeddings import EmbeddingFactory
from src.rag.retriever import LeanRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_lean_files(
    repo_dir: str, 
    output_dir: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """
    Process Lean files from repository directory.
    
    Args:
        repo_dir: Directory containing Lean files
        output_dir: Directory to save processed chunks
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of processed document chunks
    """
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR / "lean_chunks"
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Processing Lean files from {repo_dir}...")
    
    # Load Lean files
    loader = LeanLoader()
    documents = loader.load_lean_files(repo_dir)
    
    # Clean documents
    cleaned_docs = LeanProcessor.clean_lean_documents(documents)
    
    # Extract definitions
    processor = LeanDataProcessor()
    definition_docs = []
    
    logger.info(f"Extracting definitions from {len(cleaned_docs)} files...")
    
    # Use a try-except block for each document to ensure process continues even if some docs fail
    for doc in cleaned_docs:
        try:
            definitions = processor.extract_lean_definitions(doc)
            definition_docs.extend(definitions)
        except Exception as e:
            logger.warning(f"Error extracting definitions from {doc.metadata.get('source', 'unknown')}: {e}")
            continue
    
    logger.info(f"Extracted {len(definition_docs)} definitions from {len(cleaned_docs)} files")
    
    # Enrich metadata
    enriched_docs = processor.enrich_document_metadata(definition_docs)
    
    # Split into chunks
    chunks = LeanProcessor.split_documents(enriched_docs, chunk_size, chunk_overlap)
    
    # Save processed chunks
    processor.save_processed_documents(chunks, output_dir, prefix="lean_chunk_")
    
    logger.info(f"Saved {len(chunks)} chunks to {output_dir}")
    
    return chunks

def process_documentation(
    docs_dir: str, 
    output_dir: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """
    Process Lean documentation.
    
    Args:
        docs_dir: Directory containing scraped documentation
        output_dir: Directory to save processed chunks
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of processed document chunks
    """
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR / "doc_chunks"
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Processing documentation from {docs_dir}...")
    
    # Check if directory exists
    if not os.path.exists(docs_dir):
        logger.warning(f"Documentation directory {docs_dir} does not exist. Skipping documentation processing.")
        logger.warning("You may need to run the scrape_docs.py script first.")
        return []
    
    # Load documentation
    processor = LeanDataProcessor()
    documents = processor.load_processed_documents(docs_dir)
    
    if not documents:
        logger.warning(f"No documents found in {docs_dir}. Skipping documentation processing.")
        return []
    
    # Split into chunks
    chunks = LeanProcessor.split_documents(documents, chunk_size, chunk_overlap)
    
    # Save processed chunks
    processor.save_processed_documents(chunks, output_dir, prefix="doc_chunk_")
    
    logger.info(f"Saved {len(chunks)} chunks to {output_dir}")
    
    return chunks

def build_vectorstore(
    documents: List[Dict[str, Any]],
    output_dir: Optional[Union[str, Path]] = None,
    embedding_model: str = DEFAULT_EMBEDDINGS_MODEL,
    collection_name: str = "lean4_collection"
):
    """
    Build vector store from documents.
    
    Args:
        documents: List of documents
        output_dir: Directory to save vector store
        embedding_model: Name of embedding model
        collection_name: Name of collection
    """
    if output_dir is None:
        output_dir = VECTOR_STORE_DIR
    
    # Convert to string if it's a Path object
    output_dir_str = str(output_dir) if isinstance(output_dir, Path) else output_dir
    
    os.makedirs(output_dir_str, exist_ok=True)
    
    logger.info(f"Building vector store with {len(documents)} documents...")
    
    # Get embeddings
    embeddings = EmbeddingFactory.get_embedding_model(embedding_model)
    
    # Create vector store
    start_time = time.time()
    
    try:
        # Clean document metadata but don't modify the original documents
        logger.info("Filtering complex metadata from documents...")
        
        # Make sure we're dealing with Document objects
        if documents and not isinstance(documents[0], Document):
            logger.warning("Documents are not Document objects. Converting...")
            documents = [
                Document(page_content=doc.get("page_content", ""), 
                         metadata=doc.get("metadata", {})) 
                if isinstance(doc, dict) else doc 
                for doc in documents
            ]
        
        # Create retriever with original documents - the metadata cleaning will happen inside
        retriever = LeanRetriever.create_from_documents(
            documents=documents,
            embedding_function=embeddings,
            persist_directory=output_dir_str,
            collection_name=collection_name
        )
        end_time = time.time()
        
        logger.info(f"Vector store built in {end_time - start_time:.2f} seconds")
        
        # Log stats
        stats = retriever.get_collection_stats()
        logger.info(f"Vector store stats: {stats}")
        
    except Exception as e:
        logger.error(f"Error building vector store: {e}")
        logger.error("Try updating chromadb or checking your disk space.")
        
        # Provide specific advice based on the error
        if "metadata value" in str(e):
            logger.error("This error is related to complex metadata values that ChromaDB cannot handle.")
            logger.error("If the filter_complex_metadata utility didn't fix the issue, you may need to manually clean your data.")
            logger.error("Try running with a small subset of documents first to identify problematic entries.")
        
        raise

def main():
    parser = argparse.ArgumentParser(description="Build vector store for Lean 4")
    parser.add_argument("--lean-dir", default=str(RAW_DATA_DIR / "lean4"), help="Directory containing Lean files")
    parser.add_argument("--docs-dir", default=str(RAW_DATA_DIR / "docs"), help="Directory containing documentation")
    parser.add_argument("--output", default=None, help="Output directory for vector store")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDINGS_MODEL, help="Embedding model to use")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap")
    parser.add_argument("--collection-name", default="lean4_collection", help="Collection name")
    parser.add_argument("--skip-docs", action="store_true", help="Skip documentation processing")
    
    args = parser.parse_args()
    
    all_chunks = []
    
    # Process Lean files
    if os.path.exists(args.lean_dir):
        lean_chunks = process_lean_files(
            args.lean_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        all_chunks.extend(lean_chunks)
    else:
        logger.warning(f"Lean directory {args.lean_dir} does not exist. Skipping Lean file processing.")
        logger.warning("You may need to run the download_lean.py script first.")
    
    # Process documentation if not skipped
    if not args.skip_docs:
        doc_chunks = process_documentation(
            args.docs_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        all_chunks.extend(doc_chunks)
    
    if not all_chunks:
        logger.error("No chunks were processed. Vector store cannot be built.")
        logger.error("Please ensure lean_dir and/or docs_dir contain valid files.")
        return
    
    # Build vector store
    build_vectorstore(
        all_chunks,
        args.output,
        args.embedding_model,
        args.collection_name
    )

if __name__ == "__main__":
    main()