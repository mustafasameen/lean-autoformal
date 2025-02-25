from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

class LeanRetriever:
    """Retriever for Lean 4 documents."""
    
    def __init__(
        self, 
        embedding_function: Embeddings,
        persist_directory: Union[str, Path],
        collection_name: str = "lean4_collection"
    ):
        """Initialize the retriever."""
        self.embedding_function = embedding_function
        # Convert Path to string if needed
        self.persist_directory = str(persist_directory) if isinstance(persist_directory, Path) else persist_directory
        self.collection_name = collection_name
        
        # Create vector store directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        logger.info(f"Initializing Chroma vector store at {self.persist_directory}")
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embedding_function,
            collection_name=collection_name
        )
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        logger.info(f"Adding {len(documents)} documents to vector store")
        self.vectorstore.add_documents(documents)
        self.vectorstore.persist()
        logger.info("Documents added and vector store persisted")
    
    def retrieve(self, query: str, k: int = 3, filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Retrieve documents based on query."""
        logger.info(f"Retrieving documents for query: '{query}' (k={k})")
        return self.vectorstore.similarity_search(query, k=k, filter=filter)
    
    def retrieve_with_relevance(self, query: str, k: int = 3, filter: Optional[Dict[str, Any]] = None) -> List[tuple]:
        """Retrieve documents with relevance scores."""
        logger.info(f"Retrieving documents with scores for query: '{query}' (k={k})")
        return self.vectorstore.similarity_search_with_relevance_scores(query, k=k, filter=filter)
    
    def retrieve_by_type(self, query: str, definition_type: str, k: int = 3) -> List[Document]:
        """Retrieve documents of a specific definition type."""
        filter_dict = {"definition_type": definition_type}
        logger.info(f"Retrieving {definition_type} documents for query: '{query}' (k={k})")
        return self.retrieve(query, k=k, filter=filter_dict)
    
    @classmethod
    def create_from_documents(
        cls,
        documents: List[Document],
        embedding_function: Embeddings,
        persist_directory: Union[str, Path],
        collection_name: str = "lean4_collection"
    ):
        """Create a new vector store from documents."""
        # Convert Path to string if needed
        persist_dir_str = str(persist_directory) if isinstance(persist_directory, Path) else persist_directory
        
        # Create vector store directory if it doesn't exist
        os.makedirs(persist_dir_str, exist_ok=True)
        
        logger.info(f"Creating new Chroma vector store from {len(documents)} documents")
        
        try:
            # Use filter_complex_metadata to ensure ChromaDB compatibility
            from langchain_community.vectorstores.utils import filter_complex_metadata
            
            # Clean documents metadata - careful to handle different document types
            cleaned_docs = []
            for doc in documents:
                if hasattr(doc, 'metadata') and doc.metadata:
                    # Create a copy with cleaned metadata
                    try:
                        # Clean metadata using our custom method first
                        clean_metadata = {}
                        for k, v in doc.metadata.items():
                            if isinstance(v, (list, dict)) or not isinstance(v, (str, int, float, bool, type(None))):
                                clean_metadata[k] = str(v)
                            else:
                                clean_metadata[k] = v
                        
                        # Create new document with cleaned metadata
                        cleaned_docs.append(
                            Document(
                                page_content=doc.page_content,
                                metadata=clean_metadata
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Error cleaning document metadata: {e}. Using minimal metadata.")
                        # Fallback to minimal metadata
                        cleaned_docs.append(
                            Document(
                                page_content=doc.page_content,
                                metadata={"source": doc.metadata.get("source", "unknown") if isinstance(doc.metadata, dict) else "unknown"}
                            )
                        )
                else:
                    # No metadata to clean
                    cleaned_docs.append(doc)
            
            logger.info(f"Cleaned metadata for {len(cleaned_docs)} documents")
            
            vectorstore = Chroma.from_documents(
                documents=cleaned_docs,
                embedding=embedding_function,
                persist_directory=persist_dir_str,
                collection_name=collection_name
            )
            
            retriever = cls(
                embedding_function=embedding_function,
                persist_directory=persist_dir_str,
                collection_name=collection_name
            )
            retriever.vectorstore = vectorstore
            logger.info(f"Vector store created at {persist_dir_str}")
            
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            if "metadata value" in str(e):
                logger.error("The error is related to incompatible metadata values. Trying with minimal metadata...")
                
                # Most aggressive approach - use only minimal metadata
                minimal_docs = []
                for doc in documents:
                    try:
                        minimal_docs.append(
                            Document(
                                page_content=doc.page_content if hasattr(doc, 'page_content') else str(doc),
                                metadata={"source": "unknown"}
                            )
                        )
                    except Exception:
                        # Last resort - convert to string if needed
                        minimal_docs.append(
                            Document(
                                page_content=str(doc),
                                metadata={"source": "unknown"}
                            )
                        )
                
                # Try one more time with absolute minimal documents
                logger.info(f"Retrying with {len(minimal_docs)} minimal documents...")
                try:
                    vectorstore = Chroma.from_documents(
                        documents=minimal_docs,
                        embedding=embedding_function,
                        persist_directory=persist_dir_str,
                        collection_name=collection_name
                    )
                    
                    retriever = cls(
                        embedding_function=embedding_function,
                        persist_directory=persist_dir_str,
                        collection_name=collection_name
                    )
                    retriever.vectorstore = vectorstore
                    logger.info(f"Vector store created with minimal metadata at {persist_dir_str}")
                    
                    return retriever
                except Exception as e2:
                    logger.error(f"Final attempt failed: {e2}")
                    raise
            # Re-raise if it's not a metadata-related error or all attempts failed
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            
            # Get sample of metadata to determine available fields
            if count > 0:
                sample = collection.get(limit=1)
                metadata_fields = list(sample['metadatas'][0].keys()) if sample['metadatas'] else []
            else:
                metadata_fields = []
            
            return {
                "document_count": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "metadata_fields": metadata_fields
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                "error": str(e),
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }