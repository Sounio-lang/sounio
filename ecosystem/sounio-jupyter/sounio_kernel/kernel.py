"""Core Sounio Kernel - IPykernel integration."""

import os
import sys
import re
from typing import Any, Dict, Optional
from ipykernel.kernelbase import Kernel

from sounio_kernel.executor import CellExecutor
from sounio_kernel.display import format_knowledge_html
from sounio_kernel.magics import SounioMagics


class SounioKernel(Kernel):
    """Jupyter kernel for the Sounio programming language."""

    implementation = "sounio_kernel"
    implementation_version = "0.2.0"
    language = "sounio"
    language_version = "0.1.0"
    language_info = {
        "name": "sounio",
        "mimetype": "text/x-sounio",
        "file_extension": ".sio",
        "pygments_lexer": "sounio",
        "codemirror_mode": "sounio",
        "nbconvert_exporter": "sounio",
    }
    banner = """
    Sounio Jupyter Kernel v0.2.0
    Epistemic computing and uncertainty quantification
    Session-persistent declarations across cells
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Sounio kernel."""
        super().__init__(**kwargs)

        # Initialize the executor for running Sounio code
        self.executor = CellExecutor()

        # Initialize completion support
        from sounio_kernel.completion import SounioCompleter
        self.completer = SounioCompleter()

        # Initialize magic commands
        self.magics = SounioMagics(self)

        # Track execution state
        self.execution_count = 0
        self.variables: Dict[str, Any] = {}
        # Last parsed Knowledge values — made available to %python magic
        self._last_knowledge_values: list = []

    def do_execute(
        self,
        code: str,
        silent: bool = False,
        store_history: bool = True,
        user_expressions: Optional[Dict[str, str]] = None,
        allow_stdin: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute a code cell."""
        if not code.strip():
            return {
                "status": "ok",
                "execution_count": self.execution_count,
                "payload": [],
                "user_expressions": {},
            }

        if store_history:
            self.execution_count += 1

        # Extract definitions for completion support
        self.completer.extract_definitions(code)

        # Check for magic commands first
        stripped = code.strip()
        if stripped.startswith("%"):
            return self._handle_magic(stripped, silent)

        # Execute the code
        success = False
        try:
            output, errors, exitcode = self.executor.run_cell(code)
            success = exitcode == 0

            # Parse and store Knowledge values for %python magic
            if output:
                self._last_knowledge_values = self._extract_knowledge_values(output)

            if not silent:
                if output:
                    # Post the output to the frontend
                    self._post_output(output)

                if errors and exitcode != 0:
                    # Post error output
                    self._post_error(errors)

        except Exception as e:
            if not silent:
                self._post_error(f"Kernel error: {str(e)}")
            success = False

        return {
            "status": "ok" if success else "error",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": user_expressions or {},
        }

    def _handle_magic(self, code: str, silent: bool) -> Dict[str, Any]:
        """Dispatch a magic command and return the execute reply."""
        handled, result = self.magics.handle_magic(code)
        if not handled:
            if not silent:
                self._post_error(f"Unknown magic command: {code.split()[0]}")
            return {
                "status": "error",
                "execution_count": self.execution_count,
                "payload": [],
                "user_expressions": {},
            }

        if not silent:
            self._post_output(result)

        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }

    def do_complete(
        self,
        code: str,
        cursor_pos: int,
    ) -> Dict[str, Any]:
        """Handle tab completion."""
        # Get completions from the completer
        matches = self.completer.complete(code, cursor_pos)

        # Compute cursor boundaries for replacement
        text_to_cursor = code[:cursor_pos]

        # Find start of word
        cursor_start = cursor_pos
        for i in range(len(text_to_cursor) - 1, -1, -1):
            if text_to_cursor[i].isalnum() or text_to_cursor[i] == "_":
                cursor_start = i
            else:
                break

        return {
            "matches": matches,
            "cursor_start": cursor_start,
            "cursor_end": cursor_pos,
            "metadata": {},
            "status": "ok",
        }

    def do_inspect(
        self,
        code: str,
        cursor_pos: int,
        detail_level: int = 0,
    ) -> Dict[str, Any]:
        """Handle introspection (Shift+Tab)."""
        # Extract the symbol at cursor position
        text_to_cursor = code[:cursor_pos]

        # Find the identifier at cursor
        word_chars = []
        for char in reversed(text_to_cursor):
            if char.isalnum() or char == "_":
                word_chars.append(char)
            else:
                break

        symbol = "".join(reversed(word_chars))

        if not symbol:
            return {
                "status": "ok",
                "found": False,
                "data": {},
                "metadata": {},
            }

        # Provide basic documentation for keywords
        keyword_docs = {
            "let": "Declare an immutable variable: let x = 5",
            "var": "Declare a mutable variable: var y = 10",
            "fn": "Declare a function: fn name() { ... }",
            "struct": "Define a structure: struct Name { field: Type }",
            "match": "Pattern matching: match value { pattern => result }",
            "Knowledge": "Epistemic type for uncertain values with provenance",
            "print": "Print value to stdout",
            "println": "Print value with newline to stdout",
            "sqrt": "Compute square root of a number",
        }

        if symbol in keyword_docs:
            return {
                "status": "ok",
                "found": True,
                "data": {
                    "text/plain": keyword_docs[symbol],
                },
                "metadata": {},
            }

        return {
            "status": "ok",
            "found": False,
            "data": {},
            "metadata": {},
        }

    def do_is_complete(self, code: str) -> Dict[str, Any]:
        """Check if code is syntactically complete."""
        code = code.strip()
        if not code:
            return {"status": "complete"}

        # Check for balanced braces/brackets/parens
        opens = code.count("{") + code.count("(") + code.count("[")
        closes = code.count("}") + code.count(")") + code.count("]")

        # Handle multi-line strings (simple check)
        triple_quotes = code.count('"""')
        if triple_quotes % 2 == 1:
            return {"status": "incomplete", "indent": "    "}

        if opens > closes:
            return {"status": "incomplete", "indent": "    "}

        return {"status": "complete"}

    def do_shutdown(self, restart: bool) -> None:
        """Handle kernel shutdown."""
        self.executor.cleanup()
        super().do_shutdown(restart)

    def _post_output(self, output: str) -> None:
        """Post execution output to the frontend."""
        # Check if output contains Knowledge values and format them
        formatted_output = self._format_epistemic_output(output)

        stream_content = {
            "name": "stdout",
            "text": formatted_output,
        }
        self.send_response(self.iopub_socket, "stream", stream_content)

    def _post_error(self, error: str) -> None:
        """Post error output to the frontend."""
        stream_content = {
            "name": "stderr",
            "text": error,
        }
        self.send_response(self.iopub_socket, "stream", stream_content)

    def _extract_knowledge_values(self, output: str) -> list:
        """Parse Knowledge values from souc stdout for %python magic injection."""
        import re
        pattern = (
            r'Knowledge\s*\{\s*value:\s*([\d.e+-]+)\s+'
            r'epsilon:\s*([\d.e+-]+)\s+'
            r'prov:\s*"([^"]*)"\s*\}'
        )
        results = []
        for v, e, prov in re.findall(pattern, output):
            try:
                # Use a simple namespace object so kernel.py doesn't depend on sounio-py
                class _K:
                    pass
                k = _K()
                k.value = float(v)
                k.epsilon = float(e)
                k.provenance = prov
                results.append(k)
            except ValueError:
                pass
        return results

    def _format_epistemic_output(self, output: str) -> str:
        """Format Knowledge values in output with HTML coloring."""
        # Pattern: Knowledge { value: <num> epsilon: <num> prov: "<str>" }
        pattern = r"Knowledge\s*\{\s*value:\s*([\d.e+-]+)\s*epsilon:\s*([\d.e+-]+)\s*prov:\s*\"([^\"]+)\"\s*\}"

        def replace_knowledge(match: Any) -> str:
            try:
                value = float(match.group(1))
                epsilon = float(match.group(2))
                prov = match.group(3)
                return format_knowledge_html(value, epsilon, prov)
            except (ValueError, IndexError):
                return match.group(0)

        return re.sub(pattern, replace_knowledge, output)
