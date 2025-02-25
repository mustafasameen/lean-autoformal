from langchain_community.document_loaders import WebBaseLoader
from src.data.custom_loader import EncodingFriendlyDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Optional
from pathlib import Path
import bs4
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class LeanLoader:
    """Loader for Lean 4 source files and documentation."""
    
    @staticmethod
    def load_lean_files(repo_dir: str) -> List[Document]:
        """Load Lean 4 source files from a directory."""
        logger.info(f"Loading Lean 4 files from {repo_dir}")
        loader = EncodingFriendlyDirectoryLoader(
            path=repo_dir, 
            glob="**/*.lean",
            silent_errors=True,
            encoding="utf-8",
            errors="ignore"
        )
        documents = loader.load()
        
        # Filter out empty documents resulting from errors
        valid_documents = [doc for doc in documents if doc.page_content.strip()]
        logger.info(f"Loaded {len(valid_documents)} Lean 4 files (skipped {len(documents) - len(valid_documents)} files with errors)")
        return valid_documents
    
    @staticmethod
    def load_lean_docs(urls: List[str]) -> List[Document]:
        """Load Lean 4 documentation from URLs."""
        logger.info(f"Loading Lean 4 documentation from {len(urls)} URLs")
        all_docs = []
        for url in urls:
            try:
                loader = WebBaseLoader(
                    web_paths=(url,),
                    bs_kwargs=dict(parse_only=bs4.SoupStrainer(["article", "main", "section"])),
                )
                docs = loader.load()
                all_docs.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {url}")
            except Exception as e:
                logger.error(f"Error loading {url}: {e}")
        
        logger.info(f"Total documentation pages loaded: {len(all_docs)}")
        return all_docs


class LeanProcessor:
    """Processor for Lean 4 documents."""
    
    @staticmethod
    def split_documents(documents: List[Document], 
                        chunk_size: int = 1000, 
                        chunk_overlap: int = 200) -> List[Document]:
        """Split documents into chunks."""
        logger.info(f"Splitting {len(documents)} documents into chunks (size={chunk_size}, overlap={chunk_overlap})")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    @staticmethod
    def clean_lean_documents(documents: List[Document]) -> List[Document]:
        """Clean Lean 4 documents by removing comments and unnecessary whitespace."""
        logger.info(f"Cleaning {len(documents)} Lean documents")
        cleaned_docs = []
        
        for doc in documents:
            content = doc.page_content
            
            # Process content line by line
            lines = []
            in_comment_block = False
            
            for line in content.split('\n'):
                # Handle multi-line comments
                if '/-' in line and not in_comment_block:
                    in_comment_block = True
                    line = line.split('/-')[0].strip()
                    if line:
                        lines.append(line)
                    continue
                
                if '-/' in line and in_comment_block:
                    in_comment_block = False
                    line = line.split('-/')[1].strip()
                    if line:
                        lines.append(line)
                    continue
                
                if in_comment_block:
                    continue
                
                # Handle single-line comments
                if '--' in line:
                    line = line.split('--')[0].strip()
                
                # Add non-empty lines
                if line.strip():
                    lines.append(line)
            
            # Create new document with cleaned content
            cleaned_content = '\n'.join(lines)
            if cleaned_content.strip():
                cleaned_docs.append(
                    Document(
                        page_content=cleaned_content,
                        metadata=doc.metadata
                    )
                )
        
        logger.info(f"Cleaned documents: {len(cleaned_docs)}")
        return cleaned_docs