from langchain_core.prompts import PromptTemplate

# Core prompt for generating Lean 4 code from natural language
FORMALIZATION_PROMPT = PromptTemplate.from_template(
    """You are an expert in Lean 4 theorem proving and formalization.
    
    Your task is to translate the following natural language mathematical statement into formal Lean 4 code.
    
    Use the following retrieved Lean 4 code examples and documentation to help with the translation:
    
    Retrieved examples:
    {examples}
    
    Natural language statement:
    {statement}
    
    Additional context (if any):
    {context}
    
    Additional constraints (if any):
    {constraints}
    
    Generate the corresponding Lean 4 code that correctly formalizes the statement above.
    Be precise and ensure your syntax follows Lean 4 conventions.
    
    Lean 4 code:
    """
)

# Prompt for validating Lean 4 code
VALIDATION_PROMPT = PromptTemplate.from_template(
    """You are an expert in Lean 4 theorem proving and formalization.
    
    Examine the following Lean 4 code and determine if it is syntactically valid according to Lean 4 rules.
    
    Lean 4 code:
    {code}
    
    Is this code syntactically valid in Lean 4? Respond with:
    1. "Valid" or "Invalid"
    2. If invalid, explain why and suggest a correction
    
    Validation result:
    """
)

# Prompt for explaining Lean 4 code
EXPLANATION_PROMPT = PromptTemplate.from_template(
    """You are an expert in Lean 4 theorem proving and formalization.
    
    Explain the following Lean 4 code in terms that would be understandable to someone 
    learning Lean 4 who has a background in mathematics but is new to formal verification.
    
    Natural language statement:
    {statement}
    
    Lean 4 code:
    {code}
    
    Your explanation should cover:
    1. How the code represents the mathematical concepts in the statement
    2. Key Lean 4 constructs and syntax used
    3. The overall structure and approach of the formalization
    4. Any important details about how Lean 4 handles these mathematical concepts
    
    Explanation:
    """
)

# Prompt for improving retrieval queries
QUERY_IMPROVEMENT_PROMPT = PromptTemplate.from_template(
    """You are an expert in Lean 4 theorem proving and formalization.
    
    I want to retrieve relevant Lean 4 code examples for the following mathematical statement:
    
    Original statement: {original_query}
    
    Help me create better retrieval queries to find relevant examples. Generate 3 alternative queries that:
    1. Extract key mathematical concepts and terms
    2. Use Lean 4 specific terminology where appropriate
    3. Focus on the core structures or patterns needed for this formalization
    
    Each query should be concise and focused on different aspects of the original statement.
    
    Alternative queries:
    """
)

# Prompt for generating a formalization with step-by-step reasoning
REASONING_FORMALIZATION_PROMPT = PromptTemplate.from_template(
    """You are an expert in Lean 4 theorem proving and formalization.
    
    Your task is to translate the following natural language mathematical statement into formal Lean 4 code.
    
    Natural language statement:
    {statement}
    
    Retrieved Lean 4 examples to help with translation:
    {examples}
    
    Please approach this problem step by step:
    
    Step 1: Identify the key mathematical concepts in the statement.
    
    Step 2: Find the most relevant patterns or structures from the retrieved examples.
    
    Step 3: Determine how to represent the mathematical objects in Lean 4.
    
    Step 4: Develop the formalization incrementally, explaining your approach.
    
    Step 5: Provide the complete, correct Lean 4 code.
    
    Work through each step thoroughly before moving to the next one.
    """
)

# Prompt for comparing multiple generated formalizations
COMPARISON_PROMPT = PromptTemplate.from_template(
    """You are an expert in Lean 4 theorem proving and formalization.
    
    I have generated multiple formalizations for the same mathematical statement:
    
    Natural language statement:
    {statement}
    
    Formalization 1:
    {formalization_1}
    
    Formalization 2:
    {formalization_2}
    
    Compare these formalizations and analyze:
    1. The strengths and weaknesses of each approach
    2. Which formalization is more accurate and why
    3. Which formalization is more idiomatic Lean 4 code
    4. Any improvements that could be made to either formalization
    
    Provide a detailed comparison and a recommendation for which formalization to use:
    """
)