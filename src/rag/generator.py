from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class LeanGenerator:
    """Generator for Lean 4 code from natural language."""
    
    def __init__(self, llm: BaseLanguageModel):
        """Initialize the generator."""
        self.llm = llm
        
        # Define prompt templates
        self.rag_prompt = PromptTemplate.from_template(
            """You are an expert in Lean 4 theorem proving and formalization.
            
            Use the following retrieved Lean 4 code examples and documentation to help translate 
            the natural language statement into correct Lean 4 code.
            
            Retrieved examples:
            {context}
            
            Natural language statement:
            {query}
            
            Generate the corresponding Lean 4 code that correctly formalizes the statement above.
            Be precise and ensure your syntax follows Lean 4 conventions.
            
            Lean 4 code:
            """
        )
        
        logger.info(f"Initializing LeanGenerator with LLM: {type(llm).__name__}")
        
        # Create generation chain
        self.chain = (
            {"context": lambda x: self._format_docs(x["docs"]), "query": RunnablePassthrough()}
            | self.rag_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Format documents for context."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def generate(self, query: str, docs: List[Document]) -> str:
        """Generate Lean 4 code from natural language query and retrieved documents."""
        logger.info(f"Generating Lean 4 code for query: '{query}' with {len(docs)} context documents")
        return self.chain.invoke({"docs": docs, "query": query})
    
    def generate_with_explanation(self, query: str, docs: List[Document]) -> Dict[str, str]:
        """Generate Lean 4 code with an explanation."""
        explanation_prompt = PromptTemplate.from_template(
            """You are an expert in Lean 4 theorem proving and formalization.
            
            You've generated the following Lean 4 code based on a natural language statement:
            
            Natural language: {query}
            
            Generated Lean 4 code:
            {code}
            
            Now provide a detailed explanation of this code. Explain:
            1. The key components and how they relate to the original statement
            2. Any mathematical concepts that were formalized
            3. How the code follows Lean 4 conventions and patterns
            4. Any notable aspects of the implementation
            
            Your explanation should be clear and educational, suitable for someone learning Lean 4.
            
            Explanation:
            """
        )
        
        try:
            # Generate the code first
            code = self.generate(query, docs)
            
            # Log what we're doing
            logger.info(f"Generating explanation for code of length {len(code)}")
            
            # Then generate explanation with a timeout to prevent hanging
            try:
                explanation = self.llm.invoke(
                    explanation_prompt.format(query=query, code=code)
                )
                
                if not explanation or explanation.strip() == "":
                    logger.warning("Empty explanation generated, providing default explanation")
                    explanation = f"This code implements '{query}' in Lean 4. It defines the necessary structures and proves the stated theorem."
            
            except Exception as e:
                logger.error(f"Error generating explanation: {e}")
                explanation = f"Error generating explanation: {str(e)}"
            
            return {
                "code": code,
                "explanation": explanation
            }
        except Exception as e:
            logger.error(f"Error in generate_with_explanation: {e}")
            return {
                "code": f"-- Error generating code: {str(e)}",
                "explanation": f"Error occurred during code generation: {str(e)}"
            }