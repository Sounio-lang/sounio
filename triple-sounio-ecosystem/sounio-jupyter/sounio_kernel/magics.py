"""Jupyter magic commands for Sounio kernel."""

import os
import re
import time
import tempfile
from typing import Any, Dict, Optional, Tuple


class SounioMagics:
    """IPython-style magic commands for Sounio kernel."""

    def __init__(self, kernel: Any) -> None:
        """Initialize magics with kernel reference."""
        self.kernel = kernel
        self.executor = kernel.executor

    def handle_magic(self, code: str) -> Tuple[bool, str]:
        """
        Handle a magic command.

        Args:
            code: The magic command line

        Returns:
            Tuple of (handled, result_message)
        """
        parts = code.strip().split(None, 1)
        magic_name = parts[0]
        magic_args = parts[1] if len(parts) > 1 else ""

        handlers = {
            "%time": self.magic_time,
            "%timeit": self.magic_timeit,
            "%writefile": self.magic_writefile,
            "%sounio": self.magic_sounio,
            "%check": self.magic_check,
            "%ast": self.magic_ast,
            "%types": self.magic_types,
        }

        if magic_name not in handlers:
            return False, ""

        handler = handlers[magic_name]
        try:
            result = handler(magic_args)
            return True, result
        except Exception as e:
            return True, f"Error executing {magic_name}: {str(e)}"

    def magic_time(self, line: str) -> str:
        """
        Time a single statement.

        Usage:
            %time <code>
        """
        if not line.strip():
            return "Usage: %time <sounio code>"

        code = line.strip()
        start = time.time()
        stdout, stderr, exitcode = self.executor.run_cell(code)
        elapsed = time.time() - start

        result = f"CPU time: {elapsed:.4f}s\n"
        if stdout:
            result += f"Output:\n{stdout}"
        if stderr and exitcode != 0:
            result += f"Error:\n{stderr}"

        return result

    def magic_timeit(self, line: str) -> str:
        """
        Time statement multiple times.

        Usage:
            %timeit <code>
            %timeit -n 100 <code>
        """
        # Parse number of iterations
        match = re.match(r"-n\s+(\d+)\s+(.*)", line.strip())
        if match:
            number = int(match.group(1))
            code = match.group(2)
        else:
            number = 3
            code = line.strip()

        if not code:
            return "Usage: %timeit [-n <number>] <sounio code>"

        times = []
        for _ in range(number):
            start = time.time()
            stdout, stderr, exitcode = self.executor.run_cell(code)
            times.append(time.time() - start)

        min_time = min(times)
        max_time = max(times)
        avg_time = sum(times) / len(times)

        result = f"Timeit results ({number} runs):\n"
        result += f"  Min: {min_time:.4f}s\n"
        result += f"  Max: {max_time:.4f}s\n"
        result += f"  Avg: {avg_time:.4f}s\n"

        return result

    def magic_writefile(self, line: str, cell: Optional[str] = None) -> str:
        """
        Write cell content to a file.

        Usage:
            %%writefile filename.sio
            <code>
        """
        parts = line.strip().split()
        if not parts:
            return "Usage: %%writefile <filename>"

        filename = parts[0]

        # Get cell content from parent kernel
        # Note: This is called from the kernel's cell magic handler
        # which provides the cell content separately
        if cell:
            try:
                with open(filename, "w") as f:
                    f.write(cell)
                return f"Wrote {len(cell)} chars to {filename}"
            except Exception as e:
                return f"Error writing file: {str(e)}"

        return "Usage: %%writefile <filename> (use as cell magic)"

    def magic_sounio(self, line: str) -> str:
        """
        Sounio kernel information and settings.

        Usage:
            %sounio info       - Show kernel version and config
            %sounio stdlib     - Show stdlib path
            %sounio souc       - Show souc binary path
        """
        subcmd = line.strip().lower() if line.strip() else "info"

        if subcmd == "info":
            return f"""Sounio Kernel v0.1.0
Language: Sounio 1.0.0-beta.4
Stdlib: {self.executor.stdlib_path or 'not found'}
Souc binary: {self.executor.souc_binary or 'not found'}
Features: Epistemic programming, uncertainty quantification, provenance tracking"""

        elif subcmd == "stdlib":
            if self.executor.stdlib_path:
                return f"Stdlib path: {self.executor.stdlib_path}"
            else:
                return "Stdlib path not found. Set SOUNIO_STDLIB_PATH environment variable."

        elif subcmd == "souc":
            if self.executor.souc_binary:
                return f"Souc binary: {self.executor.souc_binary}"
            else:
                return "Souc binary not found. Set SOUC environment variable."

        else:
            return """Available %sounio subcommands:
  %sounio info    - Show kernel version and config
  %sounio stdlib  - Show stdlib path
  %sounio souc    - Show souc binary path"""

    def magic_check(self, line: str) -> str:
        """
        Type-check code without executing.

        Usage:
            %check <code>
        """
        if not line.strip():
            return "Usage: %check <sounio code>"

        code = line.strip()

        # Wrap in main if needed
        if not code.startswith("fn "):
            code = f"fn main() with IO {{\n    {code}\n}}"

        # Write to temp file and check
        with tempfile.NamedTemporaryFile(suffix=".sio", mode="w", delete=False) as f:
            f.write(code)
            f.flush()
            temp_file = f.name

        try:
            # Use souc check command
            import subprocess

            env = os.environ.copy()
            if self.executor.stdlib_path:
                env["SOUNIO_STDLIB_PATH"] = self.executor.stdlib_path

            result = subprocess.run(
                [self.executor.souc_binary, "check", temp_file],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )

            if result.returncode == 0:
                return "✓ Type check passed"
            else:
                return f"✗ Type check failed:\n{result.stderr}"
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def magic_ast(self, line: str) -> str:
        """
        Show AST for code.

        Usage:
            %ast <code>
        """
        if not line.strip():
            return "Usage: %ast <sounio code>"

        code = line.strip()

        # Wrap in main if needed
        if not code.startswith("fn "):
            code = f"fn main() with IO {{\n    {code}\n}}"

        # Write to temp file and show AST
        with tempfile.NamedTemporaryFile(suffix=".sio", mode="w", delete=False) as f:
            f.write(code)
            f.flush()
            temp_file = f.name

        try:
            import subprocess

            env = os.environ.copy()
            if self.executor.stdlib_path:
                env["SOUNIO_STDLIB_PATH"] = self.executor.stdlib_path

            result = subprocess.run(
                [self.executor.souc_binary, "check", "--show-ast", temp_file],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )

            return result.stdout + result.stderr
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def magic_types(self, line: str) -> str:
        """
        Show inferred types.

        Usage:
            %types <code>
        """
        if not line.strip():
            return "Usage: %types <sounio code>"

        code = line.strip()

        # Wrap in main if needed
        if not code.startswith("fn "):
            code = f"fn main() with IO {{\n    {code}\n}}"

        # Write to temp file and show types
        with tempfile.NamedTemporaryFile(suffix=".sio", mode="w", delete=False) as f:
            f.write(code)
            f.flush()
            temp_file = f.name

        try:
            import subprocess

            env = os.environ.copy()
            if self.executor.stdlib_path:
                env["SOUNIO_STDLIB_PATH"] = self.executor.stdlib_path

            result = subprocess.run(
                [self.executor.souc_binary, "check", "--show-types", temp_file],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )

            return result.stdout + result.stderr
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
