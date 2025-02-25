from langchain_huggingface import HuggingFaceEmbeddings
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def get_embeddings(
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    device: str = "cpu",
    model_kwargs: Optional[Dict[str, Any]] = None
):
    """Get embeddings model."""
    if model_kwargs is None:
        model_kwargs = {"device": device}
    else:
        model_kwargs["device"] = device
    
    logger.info(f"Initializing embeddings model: {model_name}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs
    )

class EmbeddingFactory:
    """Factory for creating and managing embedding models."""
    
    _instances = {}
    
    @classmethod
    def get_embedding_model(
        cls,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cpu",
        model_kwargs: Optional[Dict[str, Any]] = None,
        cache: bool = True
    ):
        """Get embedding model, with caching option."""
        if cache and model_name in cls._instances:
            logger.info(f"Using cached embeddings model: {model_name}")
            return cls._instances[model_name]
        
        logger.info(f"Creating new embeddings model: {model_name}")
        embeddings = get_embeddings(model_name, device, model_kwargs)
        
        if cache:
            cls._instances[model_name] = embeddings
        
        return embeddings
    
    @classmethod
    def list_available_models(cls):
        """List recommended embedding models for mathematical text."""
        return [
            {
                "name": "sentence-transformers/all-mpnet-base-v2",
                "description": "General purpose embeddings, good baseline",
                "dimensions": 768
            },
            {
                "name": "intfloat/e5-large-v2",
                "description": "Better for mathematical and technical content",
                "dimensions": 1024 
            },
            {
                "name": "BAAI/bge-large-en-v1.5",
                "description": "Strong performance on retrieval tasks",
                "dimensions": 1024
            },
            {
                "name": "thenlper/gte-large",
                "description": "Good balance of quality and efficiency",
                "dimensions": 1024
            }
        ]