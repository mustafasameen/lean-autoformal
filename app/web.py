import sys
from pathlib import Path
import os
import logging
import gradio as gr
from typing import Dict, Any, List, Tuple, Optional

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
pipeline = None
examples = [
    ["Prove that the sum of two even numbers is even.", "Consider using induction on natural numbers."],
    ["Define a group structure and prove that identity is unique.", ""],
    ["Show that if a < b and c < d, then a + c < b + d for real numbers.", ""],
    ["Define what it means for a function to be continuous.", ""],
    ["Prove that the composition of injective functions is injective.", ""]
]

def setup_pipeline(
    vector_store_dir: str,
    embedding_model: str,
    generator_model: str,
    reasoner_model: str,
    ollama_base_url: str,
    k: int
) -> AutoformalizationPipeline:
    """Set up the autoformalization pipeline."""
    # Set up embeddings
    embeddings = EmbeddingFactory.get_embedding_model(embedding_model)
    
    # Set up retriever
    retriever = LeanRetriever(
        embedding_function=embeddings,
        persist_directory=vector_store_dir
    )
    
    # Set up LLMs
    generator_llm = LLMFactory.get_llm(
        model_name=generator_model,
        base_url=ollama_base_url
    )
    
    reasoner_llm = LLMFactory.get_llm(
        model_name=reasoner_model,
        base_url=ollama_base_url
    )
    
    # Set up pipeline
    return AutoformalizationPipeline(
        retriever=retriever,
        generator_llm=generator_llm,
        validator_llm=reasoner_llm,
        k=k
    )

def initialize(
    vector_store_dir: str = VECTOR_STORE_DIR,
    embedding_model: str = DEFAULT_EMBEDDINGS_MODEL,
    generator_model: str = DEFAULT_SMALL_LLM,
    reasoner_model: str = DEFAULT_REASONER_LLM,
    ollama_base_url: str = OLLAMA_BASE_URL,
    k: int = DEFAULT_RETRIEVE_K
):
    """Initialize the application."""
    global pipeline
    
    # Check if vector store exists
    if not os.path.exists(vector_store_dir):
        raise ValueError(f"Vector store not found at {vector_store_dir}. Please run the build_vectorstore.py script first.")
    
    # Set up pipeline
    pipeline = setup_pipeline(
        vector_store_dir=vector_store_dir,
        embedding_model=embedding_model,
        generator_model=generator_model,
        reasoner_model=reasoner_model,
        ollama_base_url=ollama_base_url,
        k=k
    )
    
    logger.info("Application initialized")

def formalize(
    statement: str,
    context: str = "",
    number_examples: int = 3
) -> Tuple[str, str, str]:
    """Formalize a mathematical statement in Lean 4."""
    global pipeline
    
    if pipeline is None:
        try:
            initialize()
        except Exception as e:
            return f"Error initializing pipeline: {str(e)}", "", ""
    
    try:
        # Create input
        input_data = NaturalLanguageInput(
            statement=statement,
            context=context if context else None
        )
        
        # Update k value if needed
        pipeline.k = number_examples
        
        # Run pipeline
        result = pipeline.run(input_data)
        
        # Format retrieved examples
        if result.examples_used and len(result.examples_used) > 0:
            examples_text = "## Retrieved Examples Used for Generation\n\n"
            for i, example in enumerate(result.examples_used):
                examples_text += f"### Example {i+1}\n"
                examples_text += f"**Source:** {example.source}\n\n"
                examples_text += f"```lean\n{example.content}\n```\n\n"
                examples_text += "---\n\n"
        else:
            examples_text = "## No Examples Retrieved\n\n"
            examples_text += "No examples were found in the database that match this query. "
            examples_text += "The formalization was generated without specific examples as context.\n\n"
            examples_text += "Consider running the `build_vectorstore.py` script to populate the database with more examples."
        
        # Format explanation
        explanation = result.explanation if result.explanation else "No detailed explanation was generated."
        
        return result.lean_code, explanation, examples_text
    
    except Exception as e:
        logger.error(f"Error: {e}")
        error_message = f"Error: {str(e)}"
        return error_message, "An error occurred during formalization.", "No examples could be retrieved due to an error."

def create_ui():
    """Create the Gradio UI."""
    with gr.Blocks(title="Lean 4 Autoformalization", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Lean 4 Autoformalization")
        gr.Markdown("""
        Translate natural language mathematical statements into formal Lean 4 code using RAG (Retrieval-Augmented Generation).
        
        This tool retrieves relevant examples from a database of Lean 4 code and documentation, then uses an LLM to generate a formalization.
        """)
        
        with gr.Row():
            with gr.Column():
                statement = gr.Textbox(
                    label="Mathematical Statement",
                    placeholder="Enter a mathematical statement to formalize...",
                    lines=3
                )
                context = gr.Textbox(
                    label="Additional Context (Optional)",
                    placeholder="Provide any additional context or constraints...",
                    lines=2
                )
                num_examples = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Number of Examples to Retrieve"
                )
                submit_btn = gr.Button("Formalize")
                
                gr.Examples(
                    examples=examples,
                    inputs=[statement, context],
                    label="Example Statements"
                )
            
            with gr.Column():
                with gr.Tabs():
                    with gr.TabItem("Lean 4 Code"):
                        code_output = gr.TextArea(
                            label="Generated Lean 4 Code",
                            lines=15,
                            max_lines=30,
                            show_copy_button=True
                        )
                    with gr.TabItem("Explanation"):
                        explanation_output = gr.Markdown(
                            label="Explanation"
                        )
                    with gr.TabItem("Retrieved Examples"):
                        examples_output = gr.Markdown(
                            label="Retrieved Examples"
                        )
        
        # Set up event handlers
        submit_btn.click(
            fn=formalize,
            inputs=[statement, context, num_examples],
            outputs=[code_output, explanation_output, examples_output]
        )
    
    return app

def main():
    # Initialize the application
    try:
        initialize()
    except Exception as e:
        logger.error(f"Error initializing application: {e}")
        print(f"Error initializing application: {e}")
        print("Please make sure the vector store has been built first.")
    
    # Create and launch the UI
    app = create_ui()
    app.launch(share=False)

if __name__ == "__main__":
    main()