import subprocess
import tempfile
import os
from typing import Dict, Any, Tuple, Optional
import logging
import re

logger = logging.getLogger(__name__)

class LeanValidator:
    """Validator for Lean 4 code."""
    
    def __init__(self, lean_path: Optional[str] = None):
        """
        Initialize the Lean validator.
        
        Args:
            lean_path: Path to the Lean 4 executable. If None, it will be searched in PATH.
        """
        self.lean_path = lean_path or self._find_lean_executable()
        logger.info(f"Initialized LeanValidator with Lean path: {self.lean_path}")
    
    def _find_lean_executable(self) -> str:
        """Find the Lean 4 executable in PATH."""
        try:
            # Try to find lean directly
            result = subprocess.run(
                ["which", "lean"],
                check=True,
                capture_output=True,
                text=True
            )
            lean_path = result.stdout.strip()
            logger.info(f"Found Lean executable at: {lean_path}")
            return lean_path
        except subprocess.CalledProcessError:
            # If not found, try to use the elan-managed one
            try:
                result = subprocess.run(
                    ["elan", "which", "lean"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                lean_path = result.stdout.strip()
                logger.info(f"Found Lean executable via elan at: {lean_path}")
                return lean_path
            except subprocess.CalledProcessError:
                logger.warning("Lean executable not found. Validation will be simulated.")
                return "lean"  # Default to "lean" and hope it's in PATH when used
    
    def validate(self, code: str, timeout: int = 10) -> Tuple[bool, str]:
        """
        Validate Lean 4 code by running it through the Lean executable.
        
        Args:
            code: Lean 4 code to validate
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not self.lean_path or self.lean_path == "lean":
            # If Lean is not available, fall back to simulated validation
            return self._simulated_validation(code)
        
        # Create a temporary file with the code
        with tempfile.NamedTemporaryFile(suffix=".lean", mode="w", delete=False) as temp:
            temp.write(code)
            temp_path = temp.name
        
        try:
            # Run Lean on the temporary file
            result = subprocess.run(
                [self.lean_path, temp_path],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Check if there are any errors
            if result.returncode == 0 and not result.stderr:
                return True, "Code is valid Lean 4."
            else:
                return False, f"Validation errors:\n{result.stderr}"
        except subprocess.TimeoutExpired:
            return False, f"Validation timed out after {timeout} seconds."
        except Exception as e:
            return False, f"Error during validation: {str(e)}"
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def _simulated_validation(self, code: str) -> Tuple[bool, str]:
        """
        Simulate validation when the Lean executable is not available.
        This is a fallback method that performs basic syntax checks.
        
        Args:
            code: Lean 4 code to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        logger.info("Performing simulated validation")
        
        # Basic syntax checks
        errors = []
        
        # Check for balanced braces, parentheses, and brackets
        if code.count('{') != code.count('}'):
            errors.append("Unbalanced braces: '{' and '}'")
        
        if code.count('(') != code.count(')'):
            errors.append("Unbalanced parentheses: '(' and ')'")
        
        if code.count('[') != code.count(']'):
            errors.append("Unbalanced brackets: '[' and ']'")
        
        # Check for common Lean 4 keywords
        lean_keywords = ['theorem', 'lemma', 'definition', 'def', 'structure', 'inductive', 'class', 'instance']
        if not any(keyword in code for keyword in lean_keywords):
            errors.append("No Lean 4 keywords found (theorem, lemma, definition, etc.)")
        
        # Validate import statements
        import_matches = re.findall(r'import\s+([^\n]+)', code)
        for match in import_matches:
            if ',' in match and '.' not in match:
                errors.append(f"Invalid import statement: {match}. Use period for nested modules.")
        
        # Check for missing end statements
        if 'namespace' in code and 'end' not in code:
            errors.append("Missing 'end' statement for namespace")
        
        # Check for Lean-specific patterns
        if 'by {' in code and not re.search(r'by\s*\{[^}]+\}', code):
            errors.append("Incomplete 'by' tactic block")
        
        if errors:
            return False, "Simulated validation found issues:\n" + "\n".join(errors)
        else:
            return True, "Simulated validation passed. Note: This is not a full Lean 4 validation."