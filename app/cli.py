import sys
from pathlib import Path
import argparse
import logging
import os
import json
import time
import readline  # For better CLI input experience
from typing import Dict, Any, Optional, List

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    VECTOR_STORE_DIR,
    DEFAULT_EMBEDDINGS_MODEL,
    DEFAULT_RETRIEVE_K,
    DEFAULT_SMALL_LLM,
    DEFAULT_REASONER_LLM,
    OLLAMA_BASE_URL
)
from src.models.embeddings import EmbeddingFactory
from src.models.llm import LLMFactory
from src.rag.retriever import LeanRetriever
from src.formalization.pipeline import AutoformalizationPipeline
from src.formalization.schema import NaturalLanguageInput, LeanFormalization
from src.utils.lean_validator import LeanValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_pipeline(
    vector_store_dir: str = VECTOR_STORE_DIR,
    embedding_model: str = DEFAULT_EMBEDDINGS_MODEL,
    generator_model: str = DEFAULT_SMALL_LLM,
    reasoner_model: str = DEFAULT_REASONER_LLM,
    ollama_base_url: str = OLLAMA_BASE_URL,
    k: int = DEFAULT_RETRIEVE_K
) -> AutoformalizationPipeline:
    """
    Set up the autoformalization pipeline.
    
    Args:
        vector_store_dir: Directory containing vector store
        embedding_model: Name of embedding model
        generator_model: Name of generator model
        reasoner_model: Name of reasoner model
        ollama_base_url: Base URL for Ollama
        k: Number of documents to retrieve
        
    Returns:
        Autoformalization pipeline
    """
    # Check if vector store exists
    if not os.path.exists(vector_store_dir):
        logger.error(f"Vector store not found at {vector_store_dir}")
        logger.error("Please run the build_vectorstore.py script first")
        sys.exit(1)
    
    # Set up embeddings
    embeddings = EmbeddingFactory.get_embedding_model(embedding_model)
    
    # Set up retriever
    retriever = LeanRetriever(
        embedding_function=embeddings,
        persist_directory=vector_store_dir
    )
    
    # Log retriever stats
    stats = retriever.get_collection_stats()
    logger.info(f"Retriever stats: {stats}")
    
    # Set up LLMs
    generator_llm = LLMFactory.get_llm(
        model_name=generator_model,
        base_url=ollama_base_url
    )
    
    reasoner_llm = LLMFactory.get_llm(
        model_name=reasoner_model,
        base_url=ollama_base_url
    )
    
    # Set up validator
    validator = LeanValidator()
    
    # Set up pipeline
    pipeline = AutoformalizationPipeline(
        retriever=retriever,
        generator_llm=generator_llm,
        validator_llm=reasoner_llm,
        k=k
    )
    
    return pipeline

def save_result(result: LeanFormalization, output_dir: str = "outputs"):
    """
    Save formalization result to file.
    
    Args:
        result: Formalization result
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    timestamp = int(time.time())
    filename = f"formalization_{timestamp}.json"
    file_path = os.path.join(output_dir, filename)
    
    # Convert to dict
    result_dict = result.model_dump()
    
    # Convert examples to dicts
    result_dict["examples_used"] = [example.model_dump() for example in result.examples_used]
    
    # Convert natural input to dict
    result_dict["natural_input"] = result.natural_input.model_dump()
    
    # Convert timestamp to ISO format
    result_dict["timestamp"] = result.timestamp.isoformat()
    
    # Save to file
    with open(file_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    
    logger.info(f"Result saved to {file_path}")
    
    # Also save just the Lean code to a .lean file
    lean_file_path = os.path.join(output_dir, f"formalization_{timestamp}.lean")
    with open(lean_file_path, "w") as f:
        f.write(result.lean_code)
    
    logger.info(f"Lean code saved to {lean_file_path}")
    
    return file_path

def interactive_mode(pipeline: AutoformalizationPipeline, output_dir: str = "outputs"):
    """
    Run the autoformalization pipeline in interactive mode.
    
    Args:
        pipeline: Autoformalization pipeline
        output_dir: Output directory
    """
    print("\n" + "=" * 80)
    print("Lean 4 Autoformalization Interactive Mode")
    print("=" * 80)
    print("\nEnter mathematical statements to formalize in Lean 4.")
    print("Type 'exit' or 'quit' to exit.")
    print("Type 'help' for help.")
    
    while True:
        try:
            # Get input
            statement = input("\n> ").strip()
            
            # Check for exit command
            if statement.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            
            # Check for help command
            if statement.lower() == "help":
                print_help()
                continue
            
            # Skip empty input
            if not statement:
                continue
            
            # Ask for additional context
            context = input("Additional context (optional): ").strip()
            
            # Create input
            input_data = NaturalLanguageInput(
                statement=statement,
                context=context if context else None
            )
            
            try:
                # Run pipeline
                print("\nGenerating Lean 4 formalization...\n")
                result = pipeline.run(input_data)
                
                # Save result
                save_result(result, output_dir)
                
                # Display result
                print("\n" + "=" * 80)
                print("LEAN 4 FORMALIZATION")
                print("=" * 80)
                print("\n" + result.lean_code + "\n")
                
                if result.is_valid is not None:
                    print(f"Validation: {'✓ Valid' if result.is_valid else '✗ Invalid'}")
                    if not result.is_valid and result.validation_message:
                        print(result.validation_message)
                
                # Display examples used for generation
                if result.examples_used and len(result.examples_used) > 0:
                    print("\n" + "=" * 80)
                    print("RETRIEVED EXAMPLES")
                    print("=" * 80)
                    for i, example in enumerate(result.examples_used):
                        print(f"\nExample {i+1} (from {example.source}):")
                        print("-" * 40)
                        print(example.content)
                        print("-" * 40)
                
                if result.explanation:
                    print("\n" + "=" * 80)
                    print("EXPLANATION")
                    print("=" * 80)
                    print("\n" + result.explanation + "\n")
                else:
                    print("\n" + "=" * 80)
                    print("EXPLANATION")
                    print("=" * 80)
                    print("\nNo explanation was generated. This could be because:")
                    print("1. The code validation failed")
                    print("2. The explanation generation encountered an error")
                    print("3. The LLM didn't provide an explanation")
            
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"Error: {e}")
                print("\nThere was an error generating the formalization. Please try again with a different statement.")
                print("If the error persists, check the logs for more details.")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"Error: {e}")

def print_help():
    """Print help message."""
    print("\n" + "=" * 80)
    print("HELP")
    print("=" * 80)
    print("\nEnter a mathematical statement to formalize in Lean 4.")
    print("Example: 'Prove that the sum of two even numbers is even.'")
    print("\nCommands:")
    print("  exit, quit: Exit the program")
    print("  help: Display this help message")
    print("\nAfter entering a statement, you can provide additional context if needed.")
    print("The system will retrieve relevant Lean 4 examples and generate a formalization.")

def main():
    parser = argparse.ArgumentParser(description="Lean 4 Autoformalization CLI")
    parser.add_argument("--vector-store", default=VECTOR_STORE_DIR, help="Directory containing vector store")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDINGS_MODEL, help="Embedding model to use")
    parser.add_argument("--generator-model", default=DEFAULT_SMALL_LLM, help="Generator model to use")
    parser.add_argument("--reasoner-model", default=DEFAULT_REASONER_LLM, help="Reasoner model to use")
    parser.add_argument("--ollama-base-url", default=OLLAMA_BASE_URL, help="Base URL for Ollama")
    parser.add_argument("--k", type=int, default=DEFAULT_RETRIEVE_K, help="Number of documents to retrieve")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--statement", help="Mathematical statement to formalize")
    parser.add_argument("--context", help="Additional context for the statement")
    
    args = parser.parse_args()
    
    # Set up pipeline
    pipeline = setup_pipeline(
        vector_store_dir=args.vector_store,
        embedding_model=args.embedding_model,
        generator_model=args.generator_model,
        reasoner_model=args.reasoner_model,
        ollama_base_url=args.ollama_base_url,
        k=args.k
    )
    
    if args.statement:
        # Single statement mode
        input_data = NaturalLanguageInput(
            statement=args.statement,
            context=args.context
        )
        
        result = pipeline.run(input_data)
        save_result(result, args.output)
        
        print("\n" + "=" * 80)
        print("LEAN 4 FORMALIZATION")
        print("=" * 80)
        print("\n" + result.lean_code + "\n")
        
        if result.explanation:
            print("\n" + "=" * 80)
            print("EXPLANATION")
            print("=" * 80)
            print("\n" + result.explanation + "\n")
    else:
        # Interactive mode
        interactive_mode(pipeline, args.output)

if __name__ == "__main__":
    main()