import re
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.language_models.base import BaseLanguageModel
from typing import List, Dict, Any, Optional, Tuple, Callable, TypeVar
import langgraph
from langgraph.graph import StateGraph
from pydantic import BaseModel
import uuid
import logging
import datetime

from src.formalization.schema import NaturalLanguageInput, LeanFormalization, LeanExample
from src.formalization.prompts import (
    FORMALIZATION_PROMPT, 
    VALIDATION_PROMPT, 
    EXPLANATION_PROMPT,
    QUERY_IMPROVEMENT_PROMPT
)
from src.rag.retriever import LeanRetriever

logger = logging.getLogger(__name__)

class GraphState(BaseModel):
    """State for the LangGraph."""
    input: NaturalLanguageInput
    query: Optional[str] = None
    docs: Optional[List[Document]] = None
    examples: Optional[List[LeanExample]] = None
    lean_code: Optional[str] = None
    is_valid: Optional[bool] = None
    validation_message: Optional[str] = None
    explanation: Optional[str] = None
    result: Optional[LeanFormalization] = None
    
    def update(self, **kwargs):
        """Update the state safely without duplication."""
        current_data = self.model_dump()
        current_data.update(kwargs)
        return GraphState(**current_data)

class AutoformalizationPipeline:
    """Pipeline for autoformalization of natural language to Lean 4 code."""
    
    def __init__(
        self,
        retriever: LeanRetriever,
        generator_llm: BaseLanguageModel,
        validator_llm: Optional[BaseLanguageModel] = None,
        k: int = 3
    ):
        """Initialize the pipeline."""
        self.retriever = retriever
        self.generator_llm = generator_llm
        self.validator_llm = validator_llm if validator_llm is not None else generator_llm
        self.k = k
        
        logger.info("Initializing AutoformalizationPipeline")
        
        # Build the LangGraph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the StateGraph for autoformalization."""
        
        # Define functions for each node
        def refine_query(state: GraphState) -> GraphState:
            """Improve the query for better retrieval."""
            input_data = state.input
            
            # Use the statement as the default query
            query = input_data.statement
            
            # For more advanced cases, we could use an LLM to refine the query
            # based on the input statement and context
            
            logger.info(f"Query refined: '{query}'")
            return state.update(query=query)
        
        def retrieve(state: GraphState) -> GraphState:
            """Retrieve relevant Lean 4 examples."""
            query = state.query
            docs = self.retriever.retrieve(query, k=self.k)
            
            logger.info(f"Retrieved {len(docs)} documents for query: '{query}'")
            return state.update(docs=docs)
        
        def generate(state: GraphState) -> GraphState:
            """Generate Lean 4 code from natural language."""
            docs = state.docs
            input_data = state.input
            
            # Format examples for prompt
            examples_text = "\n\n".join(doc.page_content for doc in docs)
            
            # Generate Lean 4 code
            prompt = FORMALIZATION_PROMPT.format(
                examples=examples_text,
                statement=input_data.statement,
                context=input_data.context if input_data.context else "No additional context provided.",
                constraints="\n".join(input_data.additional_constraints) if input_data.additional_constraints else "No additional constraints."
            )
            
            logger.info(f"Generating Lean 4 code for statement: '{input_data.statement}'")
            lean_code = self.generator_llm.invoke(prompt)
            
            # Create LeanExample objects
            examples = [
                LeanExample(
                    source=doc.metadata.get("source", "Unknown"),
                    content=doc.page_content,
                    metadata=doc.metadata
                )
                for doc in docs
            ]
            
            return state.update(lean_code=lean_code, examples=examples)
        
        def validate(state: GraphState) -> GraphState:
            """Validate the generated Lean 4 code."""
            if self.validator_llm is None:
                logger.info("Skipping validation (no validator LLM provided)")
                return state.update(
                    is_valid=None,
                    validation_message="Validation skipped (no validator LLM provided)."
                )
            
            lean_code = state.lean_code
            
            # Validate Lean 4 code (simulated through LLM)
            prompt = VALIDATION_PROMPT.format(code=lean_code)
            
            logger.info("Validating generated Lean 4 code")
            validation_result = self.validator_llm.invoke(prompt)
            
            # Parse validation result
            is_valid = "valid" in validation_result.lower() and "invalid" not in validation_result.lower()
            
            logger.info(f"Validation result: {'Valid' if is_valid else 'Invalid'}")
            return state.update(
                is_valid=is_valid,
                validation_message=validation_result
            )
        
        def explain(state: GraphState) -> GraphState:
            """Generate an explanation for the Lean 4 code."""
            if not state.is_valid and state.is_valid is not None:
                logger.info("Skipping explanation for invalid code")
                return state.update(explanation="Explanation skipped because the code was marked as invalid.")
            
            prompt = EXPLANATION_PROMPT.format(
                statement=state.input.statement,
                code=state.lean_code
            )
            
            logger.info("Generating explanation for Lean 4 code")
            try:
                explanation = self.generator_llm.invoke(prompt)
                
                # Check if explanation is empty or very short
                if not explanation or len(explanation.strip()) < 50:
                    logger.warning("Generated explanation is too short, providing fallback explanation")
                    explanation = f"# Explanation of the Generated Lean 4 Code\n\n"
                    explanation += f"This code formalizes the statement: **{state.input.statement}**\n\n"
                    explanation += "## Key Components\n\n"
                    explanation += "- The code defines necessary mathematical structures\n"
                    explanation += "- It implements the formal proof of the stated theorem\n"
                    explanation += "- It follows Lean 4 conventions for mathematical proofs\n\n"
                    explanation += "## Implementation Details\n\n"
                    explanation += "The formalization uses standard Lean 4 techniques to represent the mathematical concepts and prove the stated theorem."
                # Add markdown formatting for better display
                else:
                    # Convert the explanation to markdown with proper formatting
                    lines = explanation.split('\n')
                    formatted_explanation = ""
                    
                    # Add a title
                    formatted_explanation += "# Explanation of the Generated Lean 4 Code\n\n"
                    
                    # Process the lines to add proper markdown formatting
                    in_list = False
                    for i, line in enumerate(lines):
                        # Skip empty lines at the beginning
                        if i < 3 and not line.strip():
                            continue
                            
                        # Format section titles
                        if line.strip() and all(c == line.strip()[0] for c in line.strip()):
                            formatted_explanation += f"## {lines[i+1] if i+1 < len(lines) else 'Section'}\n\n"
                            continue
                            
                        # Skip lines that are just separators
                        if line.strip() and all(c in '-=*' for c in line.strip()):
                            continue
                            
                        # Format numbered lists
                        if re.match(r'^\s*\d+\.', line):
                            if not in_list:
                                formatted_explanation += "\n"
                                in_list = True
                        elif in_list and not line.strip():
                            in_list = False
                            formatted_explanation += "\n"
                            
                        # Add the line with basic formatting
                        if line.strip() and line.strip()[0] == '*':
                            # Convert asterisks to proper markdown bullets
                            formatted_explanation += f"{line.replace('*', '-', 1)}\n"
                        else:
                            formatted_explanation += f"{line}\n"
                    
                    explanation = formatted_explanation
                
                return state.update(explanation=explanation)
            except Exception as e:
                logger.error(f"Error generating explanation: {e}")
                fallback = f"# Error in Explanation Generation\n\n"
                fallback += f"There was an error generating the detailed explanation: {str(e)}\n\n"
                fallback += f"## Summary\n\nThis code implements the mathematical statement: '{state.input.statement}'"
                return state.update(explanation=fallback)
        
        def format_output(state: GraphState) -> Dict[str, Any]:
            """Format the output."""
            result = LeanFormalization(
                natural_input=state.input,
                lean_code=state.lean_code,
                examples_used=state.examples,
                is_valid=state.is_valid,
                validation_message=state.validation_message,
                explanation=state.explanation,
                timestamp=datetime.datetime.now()
            )
            
            logger.info("Formatting final output")
            
            # Return a dictionary with both the state and the result
            # This allows us to access the result directly in the output
            return {"state": state.update(result=result), "result": result}
        
        # Build the graph using the current LangGraph API
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("refine_query", refine_query)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate", generate)
        workflow.add_node("validate", validate)
        workflow.add_node("explain", explain)
        workflow.add_node("format_output", format_output)
        
        # Add edges
        workflow.add_edge("refine_query", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "validate")
        workflow.add_edge("validate", "explain")
        workflow.add_edge("explain", "format_output")
        
        # Set entry point
        workflow.set_entry_point("refine_query")
        
        # Set final node and output type
        workflow.set_finish_point("format_output")
        
        # Compile the graph
        return workflow.compile()
    
    def run(self, input_data: NaturalLanguageInput) -> LeanFormalization:
        """Run the autoformalization pipeline."""
        logger.info(f"Running autoformalization for: '{input_data.statement}'")
        
        # Create initial state
        initial_state = GraphState(input=input_data)
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # In the new LangGraph version, the output is a dictionary of values
        # We need to extract the result from the output
        if isinstance(final_state, dict) and 'result' in final_state:
            return final_state['result']
        elif isinstance(final_state, dict) and 'state' in final_state and hasattr(final_state['state'], 'result'):
            return final_state['state'].result
        elif hasattr(final_state, 'result'):
            return final_state.result
        else:
            # Construct a result from the final state as a fallback
            logger.warning(f"Couldn't extract result directly from graph output. Building result from state.")
            try:
                # Try to build a result object from whatever final state we have
                if isinstance(final_state, dict):
                    state_dict = final_state
                elif hasattr(final_state, 'model_dump'):
                    state_dict = final_state.model_dump()
                else:
                    state_dict = vars(final_state)
                
                return LeanFormalization(
                    natural_input=input_data,
                    lean_code=state_dict.get('lean_code', "# Error: No code was generated"),
                    examples_used=state_dict.get('examples', []),
                    is_valid=state_dict.get('is_valid', False),
                    validation_message=state_dict.get('validation_message', "Error processing validation"),
                    explanation=state_dict.get('explanation', "No explanation available due to processing error"),
                    timestamp=datetime.datetime.now()
                )
            except Exception as e:
                # If everything fails, return a minimal result
                logger.error(f"Error constructing result from graph output: {e}")
                return LeanFormalization(
                    natural_input=input_data,
                    lean_code="# Error: Failed to generate Lean code due to an internal error.",
                    examples_used=[],
                    is_valid=False,
                    validation_message=f"Error: {str(e)}",
                    explanation="An error occurred during the autoformalization process.",
                    timestamp=datetime.datetime.now()
                )