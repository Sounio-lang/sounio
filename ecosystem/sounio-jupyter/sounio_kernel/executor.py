"""Cell executor - runs Sounio code via subprocess.

This module wraps the sounio.SounioExecutor to provide a Jupyter kernel-compatible
interface for executing Sounio code cells. The CellExecutor maintains backward
compatibility with the kernel.py interface while delegating to SounioExecutor
for the actual code execution.
"""

import os
import re
import tempfile
import subprocess
from typing import Tuple, Optional
from pathlib import Path

# Try to import SounioExecutor, fall back to subprocess if not available
try:
    from sounio import SounioExecutor as _SounioExecutor
    HAS_SOUNIO_EXECUTOR = True
except ImportError:
    HAS_SOUNIO_EXECUTOR = False
    _SounioExecutor = None


class CellExecutor:
    """Executes Sounio code cells in subprocess via SounioExecutor.

    This class wraps the sounio.SounioExecutor library to provide
    Jupyter kernel compatibility. It maintains the legacy interface
    (run_cell) while delegating execution to SounioExecutor.
    """

    def __init__(self) -> None:
        """Initialize the executor.

        Sets up either a SounioExecutor instance (preferred) or falls back
        to manual subprocess management if sounio-py is not installed.
        """
        self.souc_binary = self._find_souc_binary()
        self.stdlib_path = self._find_stdlib_path()
        self.temp_dir = tempfile.mkdtemp(prefix="sounio_kernel_")

        # Initialize SounioExecutor if available
        if HAS_SOUNIO_EXECUTOR:
            self._sounio_executor = _SounioExecutor(
                souc_path=self.souc_binary if self.souc_binary else None,
                stdlib_path=self.stdlib_path if self.stdlib_path else None,
            )
        else:
            self._sounio_executor = None

    def run_cell(self, code: str) -> Tuple[str, str, int]:
        """
        Run a Sounio code cell.

        Args:
            code: Sounio source code to execute

        Returns:
            Tuple of (stdout, stderr, exitcode)
        """
        # Wrap code in main function if not already a function definition
        wrapped_code = self._wrap_code(code)

        if HAS_SOUNIO_EXECUTOR and self._sounio_executor:
            # Use SounioExecutor for execution
            try:
                result = self._sounio_executor.run_code(wrapped_code, timeout=30)
                return result.stdout, result.stderr, result.exit_code
            except Exception as e:
                # Fallback to subprocess on error
                return "", f"SounioExecutor error: {str(e)}", 1
        else:
            # Fallback to direct subprocess execution
            temp_file = self._create_temp_file(wrapped_code)
            try:
                output, errors, exitcode = self._execute_souc(temp_file)
                return output, errors, exitcode
            finally:
                # Cleanup temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)

    def cleanup(self) -> None:
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _wrap_code(self, code: str) -> str:
        """
        Wrap code in main function if needed.

        Day 2 TODO: Auto-detect and wrap expressions for REPL-like experience.
        """
        code = code.strip()

        # If already starts with fn definition, don't wrap
        if code.startswith("fn "):
            return code

        # If starts with a type/struct definition at top-level, don't wrap
        if any(
            code.startswith(kw)
            for kw in ["type ", "struct ", "linear struct ", "enum "]
        ):
            return code

        # Otherwise, wrap in main function with IO effect
        # This includes let, var, and bare expressions
        wrapped = f"""fn main() with IO {{
{code}
}}
"""
        return wrapped

    def _create_temp_file(self, code: str) -> str:
        """Create a temporary Sounio source file."""
        temp_file = os.path.join(self.temp_dir, "cell.sio")
        with open(temp_file, "w") as f:
            f.write(code)
        return temp_file

    def _execute_souc(self, sio_file: str) -> Tuple[str, str, int]:
        """
        Execute Sounio code via souc binary.

        Returns:
            Tuple of (stdout, stderr, exitcode)
        """
        if not self.souc_binary:
            return (
                "",
                "souc binary not found. Install Sounio or set SOUC env var.",
                1,
            )

        env = os.environ.copy()
        if self.stdlib_path:
            env["SOUNIO_STDLIB_PATH"] = self.stdlib_path

        try:
            result = subprocess.run(
                [self.souc_binary, "run", sio_file],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "Execution timeout (30s)", 124
        except FileNotFoundError:
            return "", f"souc binary not found: {self.souc_binary}", 1
        except Exception as e:
            return "", f"Execution error: {str(e)}", 1

    def _find_souc_binary(self) -> Optional[str]:
        """
        Find the souc binary.

        Priority:
        1. SOUC environment variable
        2. Default location in sounio repo
        3. souc in PATH
        """
        # Check SOUC env var
        if "SOUC" in os.environ:
            souc = os.environ["SOUC"]
            if os.path.isfile(souc) and os.access(souc, os.X_OK):
                return souc

        # Check default location (assumes kernel runs from sounio repo)
        default_souc = "./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
        if os.path.isfile(default_souc) and os.access(default_souc, os.X_OK):
            return os.path.abspath(default_souc)

        # Check in /home/demetrios/RustroverProjects/sounio
        repo_souc = "/home/demetrios/RustroverProjects/sounio/artifacts/omega/souc-bin/souc-linux-x86_64-jit"
        if os.path.isfile(repo_souc) and os.access(repo_souc, os.X_OK):
            return repo_souc

        # Check PATH
        try:
            result = subprocess.run(
                ["which", "souc"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        return None

    def _find_stdlib_path(self) -> Optional[str]:
        """
        Find Sounio stdlib path.

        Priority:
        1. SOUNIO_STDLIB_PATH environment variable
        2. Default location in sounio repo
        """
        # Check env var
        if "SOUNIO_STDLIB_PATH" in os.environ:
            return os.environ["SOUNIO_STDLIB_PATH"]

        # Check default location
        default_stdlib = "./stdlib"
        if os.path.isdir(default_stdlib):
            return os.path.abspath(default_stdlib)

        # Check in /home/demetrios/RustroverProjects/sounio
        repo_stdlib = "/home/demetrios/RustroverProjects/sounio/stdlib"
        if os.path.isdir(repo_stdlib):
            return repo_stdlib

        return None
