from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
import re
import json
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class LeanDataProcessor:
    """Advanced processor for Lean 4 data preparation and management."""

    @staticmethod
    def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean metadata by ensuring all values are simple types.
        ChromaDB requires all metadata values to be str, int, float, or bool.
        """
        cleaned = {}
        for key, value in metadata.items():
            # Handle nested dictionaries
            if isinstance(value, dict):
                cleaned[key] = json.dumps(value)
            # Handle lists by converting to string
            elif isinstance(value, list):
                cleaned[key] = json.dumps(value)
            # Handle non-serializable objects
            elif not isinstance(value, (str, int, float, bool, type(None))):
                cleaned[key] = str(value)
            else:
                cleaned[key] = value
        return cleaned
    
    @staticmethod
    def extract_lean_definitions(document: Document) -> List[Document]:
        """
        Extract individual Lean 4 definitions, theorems, and structures from a document.
        Returns a list of documents, each containing a single Lean 4 definition.
        """
        content = document.page_content
        file_path = document.metadata.get("source", "unknown")
        
        # Pattern to match Lean 4 definitions, theorems, structures, etc.
        patterns = [
            r'(theorem\s+\w+[^{]+\{[^}]+\})',
            r'(lemma\s+\w+[^{]+\{[^}]+\})',
            r'(definition\s+\w+[^:=]+:=[^;]+)',
            r'(structure\s+\w+[^{]+\{[^}]+\})',
            r'(inductive\s+\w+[^{]+\{[^}]+\})',
            r'(class\s+\w+[^{]+\{[^}]+\})',
            r'(instance\s+\w+[^{]+\{[^}]+\})',
            r'(def\s+\w+[^:=]+:=[^;]+)',
        ]
        
        definitions = []
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                # Create a new document for each definition
                definitions.append(
                    Document(
                        page_content=match.strip(),
                        metadata={
                            **document.metadata,
                            "definition_type": pattern.split(r'\s+')[0].replace('(', ''),
                            "parent_file": file_path
                        }
                    )
                )
        
        logger.info(f"Extracted {len(definitions)} definitions from {file_path}")
        return definitions
    
    @staticmethod
    def save_processed_documents(documents: List[Document], output_dir: str, prefix: str = "chunk_") -> None:
        """Save processed documents to JSON files."""
        os.makedirs(output_dir, exist_ok=True)
        
        for i, doc in enumerate(documents):
            file_path = os.path.join(output_dir, f"{prefix}{i:05d}.json")
            with open(file_path, 'w') as f:
                json.dump({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }, f, indent=2)
        
        logger.info(f"Saved {len(documents)} processed documents to {output_dir}")
    
    @staticmethod
    def load_processed_documents(input_dir: str) -> List[Document]:
        """Load processed documents from JSON files."""
        documents = []
        dir_path = Path(input_dir)
        
        # Check if directory exists
        if not dir_path.exists():
            logger.warning(f"Directory {input_dir} does not exist.")
            return documents
            
        # Find all JSON files
        json_files = list(dir_path.glob("*.json"))
        if not json_files:
            logger.warning(f"No JSON files found in {input_dir}.")
            return documents
            
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Handle different JSON formats
                if isinstance(data, dict):
                    if "content" in data and "metadata" in data:
                        # Standard format
                        documents.append(
                            Document(
                                page_content=data["content"],
                                metadata=data["metadata"]
                            )
                        )
                    elif "url" in data and "content" in data:
                        # Format from scrape_docs.py
                        documents.append(
                            Document(
                                page_content=data["content"],
                                metadata={"source": data.get("url", "unknown"), 
                                         "title": data.get("title", "Untitled")}
                            )
                        )
                    else:
                        # Unknown format, use the entire content
                        documents.append(
                            Document(
                                page_content=str(data),
                                metadata={"source": str(file_path)}
                            )
                        )
                elif isinstance(data, list):
                    # If it's a list of documents
                    for item in data:
                        if isinstance(item, dict) and "content" in item:
                            documents.append(
                                Document(
                                    page_content=item["content"],
                                    metadata=item.get("metadata", {"source": str(file_path)})
                                )
                            )
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
                # Add a placeholder document to avoid breaking the pipeline
                documents.append(
                    Document(
                        page_content=f"Error loading document from {file_path.name}",
                        metadata={"source": str(file_path), "error": str(e)}
                    )
                )
        
        logger.info(f"Loaded {len(documents)} processed documents from {input_dir}")
        return documents
    
    @staticmethod
    def enrich_document_metadata(documents: List[Document]) -> List[Document]:
        """
        Enrich document metadata with additional information extracted from content.
        This helps with better retrieval and filtering.
        """
        enriched_docs = []
        
        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata.copy()
            
            # Extract theorem/definition name
            name_match = re.search(r'(theorem|lemma|definition|def|structure|inductive|class|instance)\s+(\w+)', content)
            if name_match:
                metadata["definition_name"] = name_match.group(2)
                metadata["definition_type"] = name_match.group(1)
            
            # Extract imports
            import_matches = re.findall(r'import\s+([^\n]+)', content)
            if import_matches:
                metadata["imports"] = ", ".join([imp.strip() for imp in import_matches])
            
            # Determine if the document contains a proof
            if "theorem" in content or "lemma" in content:
                metadata["has_proof"] = "proof" in content or "by" in content
            
            # Clean metadata to ensure compatibility with ChromaDB
            cleaned_metadata = LeanDataProcessor.clean_metadata(metadata)
            
            # Add to enriched documents
            enriched_docs.append(Document(page_content=content, metadata=cleaned_metadata))
        
        logger.info(f"Enriched metadata for {len(enriched_docs)} documents")
        return enriched_docs