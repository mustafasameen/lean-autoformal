# Lean 4 Autoformalization System

A system for translating natural language mathematical statements into formal Lean 4 code using Retrieval-Augmented Generation (RAG).

## Overview

This project implements an autoformalization pipeline that bridges the gap between informal mathematical statements and their formal counterparts in Lean 4. It leverages a RAG approach to retrieve relevant Lean 4 code examples and documentation, which are then used to guide an LLM in generating accurate Lean 4 formalizations.

### Key Features

- **Data Collection**: Clones Lean 4 repositories and scrapes documentation
- **RAG-based Retrieval**: Finds relevant examples to guide formalization
- **LLM-powered Generation**: Uses local LLMs via Ollama for code generation
- **Validation**: Validates generated Lean 4 code
- **Explanation**: Provides explanations of the generated formalizations
- **CLI & Web Interface**: Multiple ways to interact with the system

## Installation

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) for running local LLMs
- Git for cloning repositories

### Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/lean4-autoformalization.git
   cd lean4-autoformalization
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Install required Ollama models:

   ```bash
   ollama pull qwen2.5:14b-instruct-q4_K_M
   ollama pull deepseek-r1:8b
   ```

4. Configure the environment:
   ```bash
   cp .env.example .env
   # Edit .env file to adjust settings if needed
   ```

## Building the Knowledge Base

Before using the system, you need to build the knowledge base:

1. Download Lean 4 source code:

   ```bash
   python scripts/download_lean.py
   ```

2. Scrape Lean 4 documentation:

   ```bash
   python scripts/scrape_docs.py
   ```

3. Build the vector store:
   ```bash
   python scripts/build_vectorstore.py
   ```

## Usage

### Command Line Interface

Run the CLI in interactive mode:

```bash
python app/cli.py
```

Or formalize a single statement:

```bash
python app/cli.py --statement "Prove that the sum of two even numbers is even."
```

### Web Interface

Launch the web interface:

```bash
python app/web.py
```

Then open your browser at `http://localhost:7860`.

## Project Structure

```
lean4-autoformalization/
├── data/                   # Data storage
│   ├── raw/                # Raw Lean 4 files
│   ├── processed/          # Processed chunks
│   └── vector_store/       # Chroma DB files
├── src/                    # Source code
│   ├── config.py           # Configuration
│   ├── data/               # Data handling
│   ├── models/             # Model definitions
│   ├── rag/                # RAG components
│   ├── formalization/      # Autoformalization components
│   └── utils/              # Utilities
├── scripts/                # Utility scripts
│   ├── download_lean.py    # Download Lean repo
│   ├── scrape_docs.py      # Scrape Lean documentation
│   └── build_vectorstore.py # Build vector database
└── app/                    # Application
    ├── cli.py              # Command-line interface
    └── web.py              # Web interface
```

## How It Works

1. **Data Collection**: The system collects Lean 4 code and documentation from various sources.

2. **Preprocessing**: The collected data is processed, cleaned, and split into chunks.

3. **Indexing**: The chunks are embedded using a sentence transformer model and stored in a Chroma vector database.

4. **Retrieval**: When a mathematical statement is provided, the system retrieves the most relevant Lean 4 examples.

5. **Generation**: An LLM uses the retrieved examples to generate a formal Lean 4 representation of the mathematical statement.

6. **Validation**: The generated code is validated for correctness.

7. **Explanation**: An explanation of the formalization is provided.

## Configuration

Edit the `.env` file to configure:

- LLM models
- Embedding model
- Retrieval parameters
- Chunk size and overlap
- Ollama base URL

## Examples

### Example 1: Natural Number Induction

**Input**: "Prove that the sum of the first n odd numbers equals n squared."

**Output**:

```lean
theorem sum_first_n_odd_equals_n_squared (n : Nat) :
  (∑ i in Finset.range n, 2 * i + 1) = n^2 := by
  induction n with
  | zero => simp
  | succ n ih =>
    simp [Finset.sum_range_succ]
    rw [ih]
    ring
```

### Example 2: Group Theory

**Input**: "Define a group structure and prove that the identity element is unique."

**Output**:

```lean
class Group (G : Type) extends Mul G, Inv G, One G where
  mul_assoc : ∀ a b c : G, a * b * c = a * (b * c)
  one_mul : ∀ a : G, 1 * a = a
  mul_one : ∀ a : G, a * 1 = a
  mul_left_inv : ∀ a : G, a⁻¹ * a = 1

theorem group_identity_unique {G : Type} [Group G] (e : G) (h : ∀ a : G, e * a = a) : e = 1 := by
  specialize h 1
  rw [← mul_one e] at h
  exact h
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Lean Community](https://leanprover-community.github.io/) for the Lean theorem prover
- [LangChain](https://python.langchain.com/) for the RAG framework
- [Ollama](https://ollama.ai/) for local LLM capabilities
