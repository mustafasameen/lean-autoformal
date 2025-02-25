from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import datetime

class LeanExample(BaseModel):
    """Schema for a Lean 4 code example."""
    source: str = Field(..., description="Source of the example (e.g., file path, URL)")
    content: str = Field(..., description="Content of the example")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    relevance_score: Optional[float] = Field(default=None, description="Relevance score if available")

class NaturalLanguageInput(BaseModel):
    """Schema for natural language input."""
    statement: str = Field(..., description="Natural language mathematical statement")
    context: Optional[str] = Field(default=None, description="Additional context for the statement")
    additional_constraints: Optional[List[str]] = Field(default_factory=list, description="Additional constraints or requirements")

class LeanFormalization(BaseModel):
    """Schema for Lean 4 formalization."""
    natural_input: NaturalLanguageInput = Field(..., description="Original natural language input")
    lean_code: str = Field(..., description="Generated Lean 4 code")
    examples_used: List[LeanExample] = Field(default_factory=list, description="Examples used for generation")
    is_valid: Optional[bool] = Field(default=None, description="Whether the formalization is valid Lean 4 code")
    validation_message: Optional[str] = Field(default=None, description="Validation message if validation failed")
    explanation: Optional[str] = Field(default=None, description="Explanation of the generated code")
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now, description="Timestamp of generation")

class FeedbackInput(BaseModel):
    """Schema for feedback on generated formalizations."""
    formalization_id: str = Field(..., description="ID of the formalization")
    is_correct: bool = Field(..., description="Whether the formalization is correct")
    issues: Optional[List[str]] = Field(default_factory=list, description="List of issues with the formalization")
    suggested_correction: Optional[str] = Field(default=None, description="Suggested correction for the formalization")
    comments: Optional[str] = Field(default=None, description="Additional comments")

class RetrievalResult(BaseModel):
    """Schema for retrieval results."""
    query: str = Field(..., description="Query used for retrieval")
    documents: List[LeanExample] = Field(..., description="Retrieved documents")
    filter_applied: Optional[Dict[str, Any]] = Field(default=None, description="Filter applied to retrieval")
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now, description="Timestamp of retrieval")