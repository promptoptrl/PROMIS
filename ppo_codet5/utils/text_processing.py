"""
Text processing utilities for PPO-CodeT5.
"""

import re
import ast
import astunparse
from typing import List, Optional, Tuple, Dict, Any
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


class TextProcessor:
    """Text processing utilities for prompt manipulation."""
    
    PREFIX = "Here is the rewritten prompt:"
    
    def __init__(self):
        """Initialize text processor."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def is_single_sentence(self, text: str) -> bool:
        """Check if text is a single sentence."""
        if "\n" in text:
            return False
        term_count = sum(text.count(x) for x in ".!?")
        return term_count <= 1
    
    def looks_like_code(self, text: str) -> bool:
        """Check if text contains code markers."""
        code_markers = (
            "```", "def ", "class ", "import ", "from ", 
            "#include", ";", "{", "}", "[PYTHON]", "[/PYTHON]"
        )
        return any(marker in text for marker in code_markers)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = re.sub(r"\s+", " ", text).strip().strip("`\"' ")
        return text
    
    def extract_rewrite(self, text: str) -> str:
        """
        Extract rewritten prompt from text.
        
        Args:
            text: Input text containing rewritten prompt
            
        Returns:
            Cleaned rewritten prompt
        """
        i = text.lower().find(self.PREFIX.lower())
        if i == -1:
            return self.clean_text(text)
        
        sliced = text[i:]
        j = sliced.lower().find(self.PREFIX.lower(), len(self.PREFIX))
        if j != -1:
            sliced = sliced[:j]
        
        after = sliced[len(self.PREFIX):].strip()
        after = after.split("\n", 1)[0]
        
        # Find first sentence
        p = after.find(".")
        if p != -1:
            after = after[:p+1]
        
        return self.clean_text(f"{self.PREFIX} {after}")
    
    def validate_rewrite(self, text: str, function_name: str) -> Tuple[bool, List[str]]:
        """
        Validate rewritten prompt.
        
        Args:
            text: Rewritten prompt text
            function_name: Expected function name
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not text.startswith(self.PREFIX):
            errors.append("Missing required prefix.")
        
        if text.lower().count(self.PREFIX.lower()) != 1:
            errors.append("Multiple rewritten prompts detected.")
        
        body = text[len(self.PREFIX):].strip()
        if not body:
            errors.append("Empty body.")
        
        if not self.is_single_sentence(body):
            errors.append("Not a single sentence.")
        
        if self.looks_like_code(body):
            errors.append("Contains code/examples.")
        
        if function_name not in body:
            errors.append("Missing required function_name.")
        
        if "Python function" not in body:
            errors.append("Must explicitly request a Python function.")
        
        return len(errors) == 0, errors
    
    def repair_rewrite(self, original_prompt: str, function_name: str) -> str:
        """
        Repair invalid rewritten prompt.
        
        Args:
            original_prompt: Original prompt text
            function_name: Function name
            
        Returns:
            Repaired prompt
        """
        task = re.sub(r"\s+", " ", original_prompt).strip().rstrip(".")
        return f"Write a Python function named {function_name} that performs the intended task described as: {task}."
    
    def summarize_feedback(self, feedback: str, max_chars: int = 500) -> str:
        """
        Summarize feedback text.
        
        Args:
            feedback: Feedback text
            max_chars: Maximum characters
            
        Returns:
            Summarized feedback
        """
        # Remove code blocks
        feedback = re.sub(r"```.*?```", "", feedback, flags=re.DOTALL)
        feedback = re.sub(r"\[PYTHON\].*?\[/PYTHON\]", "", feedback, flags=re.DOTALL)
        
        # Truncate if too long
        if len(feedback) > max_chars:
            feedback = feedback[:max_chars] + "..."
        
        return feedback.strip()


class CodeProcessor:
    """Code processing utilities."""
    
    def __init__(self):
        """Initialize code processor."""
        pass
    
    def parse_code_block(self, text: str, language: str = "python") -> Optional[str]:
        """
        Parse code block from text.
        
        Args:
            text: Text containing code block
            language: Programming language
            
        Returns:
            Extracted code or None
        """
        pattern = fr"```{language}\n(.*?)\n```"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1) if match else None
    
    def parse_first_function(self, code: str) -> Optional[str]:
        """
        Parse first function from code.
        
        Args:
            code: Python code
            
        Returns:
            First function definition or None
        """
        lines = code.split("\n")
        def_start = -1
        def_end = 0
        got_return = False
        
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                if def_start == -1:
                    def_start = i
                else:
                    break
            elif "return" in line and def_start != -1:
                got_return = True
            if not line.strip() and def_start != -1 and got_return:
                def_end = i
                break
        
        if def_end == 0:
            def_end = len(lines) - 1
        
        if def_start == -1:
            return None
        
        return "\n".join(lines[def_start:def_end+1]).rstrip("[/PYTHON]")
    
    def extract_function_name(self, code: str) -> Optional[str]:
        """
        Extract function name from code.
        
        Args:
            code: Python code
            
        Returns:
            Function name or None
        """
        match = re.search(r"def\s+(\w+)\s*\(", code)
        return match.group(1) if match else None
    
    def validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python syntax.
        
        Args:
            code: Python code
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)
    
    def format_code(self, code: str) -> str:
        """
        Format Python code.
        
        Args:
            code: Python code
            
        Returns:
            Formatted code
        """
        try:
            tree = ast.parse(code)
            return astunparse.unparse(tree).strip()
        except SyntaxError:
            return code
