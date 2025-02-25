from langchain_core.documents import Document
from typing import List, Iterator, Union, Dict
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class EncodingFriendlyTextLoader:
    """A text loader that handles encoding errors gracefully."""
    
    def __init__(self, file_path: Union[str, Path], encoding: str = "utf-8", errors: str = "ignore"):
        """
        Initialize the loader.
        
        Args:
            file_path: Path to the file
            encoding: Encoding to use
            errors: How to handle encoding errors ('ignore', 'replace', etc.)
        """
        self.file_path = Path(file_path)
        self.encoding = encoding
        self.errors = errors
    
    def load(self) -> List[Document]:
        """Load and return documents from the file."""
        return list(self.lazy_load())
    
    def lazy_load(self) -> Iterator[Document]:
        """Lazy load documents from the file."""
        try:
            with open(self.file_path, encoding=self.encoding, errors=self.errors) as f:
                text = f.read()
            
            metadata = {"source": str(self.file_path)}
            yield Document(page_content=text, metadata=metadata)
        except Exception as e:
            logger.warning(f"Error loading {self.file_path}: {e}")
            # Return an empty document with metadata about the error
            metadata = {
                "source": str(self.file_path),
                "error": str(e),
                "loader_error": True
            }
            yield Document(page_content="", metadata=metadata)


class EncodingFriendlyDirectoryLoader:
    """A directory loader that handles encoding errors gracefully."""
    
    def __init__(
        self,
        path: Union[str, Path],
        glob: str = "**/*",
        silent_errors: bool = True,
        encoding: str = "utf-8",
        errors: str = "ignore"
    ):
        """
        Initialize the loader.
        
        Args:
            path: Path to the directory
            glob: Glob pattern to match files
            silent_errors: Whether to silently ignore errors
            encoding: Encoding to use
            errors: How to handle encoding errors
        """
        self.path = Path(path)
        self.glob = glob
        self.silent_errors = silent_errors
        self.encoding = encoding
        self.errors = errors
    
    def load(self) -> List[Document]:
        """Load and return documents from the directory."""
        return list(self.lazy_load())
    
    def lazy_load(self) -> Iterator[Document]:
        """Lazy load documents from the directory."""
        paths = list(self.path.glob(self.glob))
        
        for path in paths:
            if path.is_file():
                try:
                    loader = EncodingFriendlyTextLoader(
                        file_path=path,
                        encoding=self.encoding,
                        errors=self.errors
                    )
                    yield from loader.lazy_load()
                except Exception as e:
                    if self.silent_errors:
                        logger.warning(f"Error loading {path}: {e}")
                    else:
                        raise