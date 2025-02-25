from langchain_ollama import OllamaLLM  # Updated import
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.language_models.base import BaseLanguageModel
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

def get_ollama_llm(
    model_name: str,
    base_url: str = "http://localhost:11434",
    temperature: float = 0.2,
    streaming: bool = False,
    model_kwargs: Optional[Dict[str, Any]] = None
):
    """Get Ollama LLM."""
    if model_kwargs is None:
        model_kwargs = {}
    
    kwargs = {
        "model": model_name,
        "base_url": base_url,
        "temperature": temperature,
    }
    
    # Add streaming if requested
    if streaming:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        kwargs["callback_manager"] = callback_manager
    
    # Add any additional model kwargs
    kwargs.update(model_kwargs)
    
    logger.info(f"Initializing Ollama LLM: {model_name}")
    return OllamaLLM(**kwargs)

class LLMFactory:
    """Factory for creating and managing LLM models."""
    
    _instances = {}
    
    @classmethod
    def get_llm(
        cls,
        model_name: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        streaming: bool = False,
        model_kwargs: Optional[Dict[str, Any]] = None,
        cache: bool = True
    ):
        """Get LLM, with caching option."""
        cache_key = f"{model_name}_{temperature}_{streaming}"
        
        if cache and cache_key in cls._instances:
            logger.info(f"Using cached LLM: {model_name}")
            return cls._instances[cache_key]
        
        logger.info(f"Creating new LLM: {model_name}")
        llm = get_ollama_llm(model_name, base_url, temperature, streaming, model_kwargs)
        
        if cache:
            cls._instances[cache_key] = llm
        
        return llm
    
    @classmethod
    def get_generator_llm(cls, base_url: str = "http://localhost:11434"):
        """Get the standard generator LLM (smaller model)."""
        from src.config import DEFAULT_SMALL_LLM
        return cls.get_llm(
            model_name=DEFAULT_SMALL_LLM,
            base_url=base_url,
            temperature=0.3
        )
    
    @classmethod
    def get_reasoner_llm(cls, base_url: str = "http://localhost:11434"):
        """Get the reasoner LLM (for validation and complex reasoning)."""
        from src.config import DEFAULT_REASONER_LLM
        return cls.get_llm(
            model_name=DEFAULT_REASONER_LLM,
            base_url=base_url,
            temperature=0.1  # Lower temperature for more deterministic reasoning
        )