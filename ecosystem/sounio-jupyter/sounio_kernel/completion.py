"""Autocompletion support for Sounio kernel."""

from typing import List, Dict, Any, Optional


# Sounio language keywords
SOUNIO_KEYWORDS = [
    # Control flow
    "if",
    "else",
    "match",
    "for",
    "while",
    "in",
    "return",
    "break",
    "continue",
    # Declarations
    "let",
    "var",
    "fn",
    "struct",
    "linear struct",
    "enum",
    "type",
    "module",
    "import",
    # Types & effects
    "i32",
    "i64",
    "f32",
    "f64",
    "bool",
    "String",
    "with",
    "IO",
    "Pure",
    "Alloc",
    "Net",
    # Epistemic types
    "Knowledge",
    "Uncertainty",
    "Measurement",
    # Operators & literals
    "true",
    "false",
    "ref",
    "deref",
    "as",
    # Special
    "_",
]

# Sounio stdlib functions (basic set)
SOUNIO_STDLIB_FUNCTIONS = [
    # I/O
    "print",
    "println",
    "read_line",
    "write",
    # Math
    "sqrt",
    "sin",
    "cos",
    "tan",
    "abs",
    "floor",
    "ceil",
    "round",
    "min",
    "max",
    "pow",
    # String
    "len",
    "concat",
    "substring",
    "trim",
    # Array/collection operations
    "push",
    "pop",
    "get",
    "set",
    "map",
    "filter",
    "fold",
    # Epistemic
    "measure",
    "knowledge",
    "uncertainty",
    "combine_knowledge",
    # Type conversion
    "to_i32",
    "to_i64",
    "to_f64",
    "to_string",
]

# Snippet templates for common patterns
SOUNIO_SNIPPETS = {
    "fn": "fn ${name}() {\n    ${0}\n}",
    "fn_with_effect": "fn ${name}() with ${effect} {\n    ${0}\n}",
    "let": "let ${name} = ${value}",
    "var": "var ${name} = ${value}",
    "struct": "struct ${name} {\n    ${field}: ${type}\n}",
    "linear_struct": "linear struct ${name} {\n    ${field}: ${type}\n}",
    "enum": "enum ${name} {\n    ${variant}\n}",
    "match": "match ${value} {\n    ${pattern} => ${result}\n}",
    "if": "if ${cond} {\n    ${then}\n} else {\n    ${else}\n}",
    "knowledge": "Knowledge { value: ${value}, epsilon: ${epsilon}, prov: \"${provenance}\" }",
    "type_alias": "type ${name} = ${base_type}",
    "for_loop": "for ${var} in ${collection} {\n    ${0}\n}",
    "while_loop": "while ${cond} {\n    ${0}\n}",
}


class SounioCompleter:
    """Handles autocompletion for Sounio code."""

    def __init__(self) -> None:
        """Initialize the completer."""
        self.user_variables: Dict[str, Any] = {}
        self.user_functions: Dict[str, Any] = {}

    def complete(self, code: str, cursor_pos: int) -> List[str]:
        """
        Generate completions for the given code at cursor position.

        Args:
            code: The full code string
            cursor_pos: Position of cursor in code

        Returns:
            List of completion strings
        """
        # Get text to cursor
        text_to_cursor = code[:cursor_pos]

        # Extract the current word being completed
        current_word = self._extract_current_word(text_to_cursor)

        if not current_word:
            return []

        matches = []

        # Check keywords
        for kw in SOUNIO_KEYWORDS:
            if kw.startswith(current_word):
                matches.append(kw)

        # Check stdlib functions
        for func in SOUNIO_STDLIB_FUNCTIONS:
            if func.startswith(current_word):
                matches.append(func)

        # Check user-defined variables
        for var in self.user_variables:
            if var.startswith(current_word):
                matches.append(var)

        # Check user-defined functions
        for func in self.user_functions:
            if func.startswith(current_word):
                matches.append(func)

        # Remove duplicates and sort
        return sorted(set(matches))

    def extract_definitions(self, code: str) -> None:
        """
        Extract variable and function definitions from code.

        Args:
            code: The Sounio code to analyze
        """
        import re

        # Extract variable declarations: let/var name = ...
        var_pattern = r"(?:let|var)\s+(\w+)"
        for match in re.finditer(var_pattern, code):
            self.user_variables[match.group(1)] = True

        # Extract function declarations: fn name(...) ...
        fn_pattern = r"fn\s+(\w+)\s*\("
        for match in re.finditer(fn_pattern, code):
            self.user_functions[match.group(1)] = True

    def _extract_current_word(self, text: str) -> str:
        """
        Extract the word currently being completed from text.

        Args:
            text: Text up to cursor position

        Returns:
            The current word, or empty string if at word boundary
        """
        # Strip trailing whitespace and other non-word chars
        if not text or text[-1].isspace():
            return ""

        # Find the start of current word (non-word chars before current word)
        word_chars = []
        for char in reversed(text):
            if char.isalnum() or char == "_":
                word_chars.append(char)
            else:
                break

        # Reverse to get proper order
        current_word = "".join(reversed(word_chars))
        return current_word

    def get_snippet(self, name: str) -> Optional[str]:
        """
        Get a snippet template by name.

        Args:
            name: The snippet name

        Returns:
            The snippet template or None
        """
        return SNIPPETS.get(name)


# Global snippet templates for display
SNIPPETS = SOUNIO_SNIPPETS
