"""Cell executor - runs Sounio code via subprocess.

This module wraps the sounio.SounioExecutor to provide a Jupyter kernel-compatible
interface for executing Sounio code cells. The CellExecutor maintains backward
compatibility with the kernel.py interface while delegating to SounioExecutor
for the actual code execution.

Session persistence:
  Declarations (fn, let, var, struct, use, pub, type) are accumulated across
  cell executions. Expressions are evaluated in the context of the accumulated
  session, giving a REPL-like experience inside Jupyter notebooks.
"""

import os
import re
import shutil
import tempfile
import subprocess
from typing import Tuple, Optional, List
from pathlib import Path

# Try to import SounioExecutor, fall back to subprocess if not available
try:
    from sounio import SounioExecutor as _SounioExecutor
    HAS_SOUNIO_EXECUTOR = True
except ImportError:
    HAS_SOUNIO_EXECUTOR = False
    _SounioExecutor = None


def _is_declaration(code: str) -> bool:
    """Return True if the code is a top-level declaration."""
    trimmed = code.strip()
    return bool(
        trimmed.startswith("fn ")
        or trimmed.startswith("let ")
        or trimmed.startswith("var ")
        or trimmed.startswith("use ")
        or trimmed.startswith("struct ")
        or trimmed.startswith("pub ")
        or trimmed.startswith("type ")
        or trimmed.startswith("module ")
        or trimmed.startswith("import ")
        or trimmed.startswith("enum ")
    )


def _filter_compiler_output(output: str) -> str:
    """Strip verbose compiler noise; keep only errors and warnings."""
    lines = []
    for line in output.splitlines():
        stripped = line.strip()
        if (
            stripped.startswith("error:")
            or stripped.startswith("warning:")
            or stripped.startswith("typecheck: failed")
            or (
                len(stripped) > 4
                and stripped[0] == "E"
                and stripped[1:4].isdigit()
                and stripped[4] == " "
            )
        ):
            lines.append(stripped)
    if lines:
        return "\n".join(lines)
    # No structured errors — return a trimmed version of the raw output
    trimmed = output.strip()
    if len(trimmed) > 500:
        return trimmed[:500] + "\n... (truncated)"
    return trimmed


class CellExecutor:
    """Executes Sounio code cells in subprocess via SounioExecutor.

    This class wraps the sounio.SounioExecutor library to provide
    Jupyter kernel compatibility. It maintains the legacy interface
    (run_cell) while delegating execution to SounioExecutor.

    Session persistence:
      The executor accumulates declarations across calls so that multi-cell
      notebooks behave like a REPL session.
    """

    def __init__(self) -> None:
        """Initialize the executor.

        Sets up either a SounioExecutor instance (preferred) or falls back
        to manual subprocess management if sounio-py is not installed.
        """
        self.souc_binary = self._find_souc_binary()
        self.stdlib_path = self._find_stdlib_path()
        self.temp_dir = tempfile.mkdtemp(prefix="sounio_kernel_")

        # Session state — accumulated declarations survive across cell execs
        self._declarations: List[str] = []

        # Initialize SounioExecutor if available
        if HAS_SOUNIO_EXECUTOR:
            self._sounio_executor = _SounioExecutor(
                souc_path=self.souc_binary if self.souc_binary else None,
                stdlib_path=self.stdlib_path if self.stdlib_path else None,
            )
        else:
            self._sounio_executor = None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset_session(self) -> None:
        """Clear all accumulated declarations."""
        self._declarations.clear()

    def get_session_source(self) -> str:
        """Return the accumulated session source code."""
        return "\n".join(self._declarations)

    def _validate_declaration(self, decl: str) -> Tuple[bool, str]:
        """Check that ``decl`` compiles together with the current session.

        Returns (ok, error_message).
        """
        src = f"{self.get_session_source()}\n{decl}\nfn main() -> i64 {{ 0 }}\n"
        ok, err = self._compile_source_str(src)
        return ok, err

    def _compile_source_str(self, src: str) -> Tuple[bool, str]:
        """Compile a source string, returning (success, stderr_or_error)."""
        temp_file = self._create_temp_file(src, suffix="check.sio")
        out_file = os.path.join(self.temp_dir, "check.out")
        try:
            output, errors, exitcode = self._raw_compile(temp_file, out_file)
            if exitcode != 0:
                filtered = _filter_compiler_output(output + "\n" + errors)
                return False, filtered
            return True, ""
        finally:
            for p in (temp_file, out_file):
                if os.path.exists(p):
                    os.remove(p)

    def _run_expression(self, expr: str) -> Tuple[str, str, int]:
        """Compile and run an expression in the context of the current session.

        Tries ``print_int`` first; falls back to ``print_f64`` on type mismatch.
        """
        session = self.get_session_source()
        src_int = (
            f"{session}\n"
            f"fn main() -> i64 with IO, Mut, Panic, Div, Alloc {{\n"
            f"    print_int({expr});\n"
            f"    print_char(10);\n"
            f"    0\n"
            f"}}\n"
        )
        result = self._compile_and_run(src_int)
        if result[2] == 0:
            return result

        # Type-mismatch fallback to f64 printer
        err_text = result[1]
        if "type mismatch" in err_text or "typecheck: failed" in err_text or "E200" in err_text:
            src_f64 = (
                f"{session}\n"
                f"fn main() -> i64 with IO, Mut, Panic, Div, Alloc {{\n"
                f"    print_f64({expr});\n"
                f"    print_char(10);\n"
                f"    0\n"
                f"}}\n"
            )
            result2 = self._compile_and_run(src_f64)
            if result2[2] == 0:
                return result2

        return result

    def _compile_and_run(self, src: str) -> Tuple[str, str, int]:
        """Compile source to ELF and execute it."""
        src_file = self._create_temp_file(src, suffix="run.sio")
        out_file = os.path.join(self.temp_dir, "run.out")
        try:
            output, errors, exitcode = self._raw_compile(src_file, out_file)
            if exitcode != 0:
                filtered = _filter_compiler_output(output + "\n" + errors)
                return "", filtered, exitcode

            # Make executable and run
            os.chmod(out_file, os.stat(out_file).st_mode | 0o755)
            result = subprocess.run(
                [out_file],
                capture_output=True,
                text=True,
                timeout=30,
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            if result.returncode != 0:
                msg = stdout if stdout else stderr
                return "", f"Program exited with code {result.returncode}\n{msg}", result.returncode
            return stdout, stderr, 0
        except subprocess.TimeoutExpired:
            return "", "Execution timeout (30s)", 124
        except Exception as e:
            return "", f"Execution error: {str(e)}", 1
        finally:
            for p in (src_file, out_file):
                if os.path.exists(p):
                    os.remove(p)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_cell(self, code: str) -> Tuple[str, str, int]:
        """
        Run a Sounio code cell.

        If the code is a declaration it is validated and added to the
        persistent session.  If it is an expression it is evaluated in
        the context of the accumulated session.

        Args:
            code: Sounio source code to execute

        Returns:
            Tuple of (stdout, stderr, exitcode)
        """
        code = code.strip()
        if not code:
            return "", "", 0

        if _is_declaration(code):
            ok, err = self._validate_declaration(code)
            if ok:
                self._declarations.append(code)
                return "", "", 0
            return "", err, 1

        return self._run_expression(code)

    def cleanup(self) -> None:
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    # ------------------------------------------------------------------
    # Code wrapping (legacy compatibility)
    # ------------------------------------------------------------------

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

    def _create_temp_file(self, code: str, suffix: str = "cell.sio") -> str:
        """Create a temporary Sounio source file."""
        fd, path = tempfile.mkstemp(suffix=suffix, dir=self.temp_dir)
        with os.fdopen(fd, "w") as f:
            f.write(code)
        return path

    # ------------------------------------------------------------------
    # Raw compilation (direct native compiler invocation)
    # ------------------------------------------------------------------

    def _raw_compile(self, sio_file: str, out_file: str) -> Tuple[str, str, int]:
        """Compile a .sio file to an ELF using the native compiler directly.

        Returns (stdout, stderr, exitcode).  stdout here means the compiler's
        stdout; the executable is written to *out_file*.
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
                [self.souc_binary, sio_file, out_file],
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "Compilation timeout (60s)", 124
        except FileNotFoundError:
            return "", f"souc binary not found: {self.souc_binary}", 1
        except Exception as e:
            return "", f"Compilation error: {str(e)}", 1

    def _find_souc_binary(self) -> Optional[str]:
        """
        Find the native souc binary (souc-linux-x86_64).

        Priority:
        1. SOUC environment variable (may point to wrapper or native binary)
        2. SOUNIO_COMPILER environment variable
        3. Repository-root bin/souc-linux-x86_64
        4. bin/souc wrapper → resolve to its native binary
        5. PATH lookup for souc-linux-x86_64
        6. PATH lookup for souc wrapper → resolve to its native binary
        """
        # Determine repo root robustly
        repo_root = self._repo_root()

        # 1. SOUC env var
        for env_var in ("SOUC", "SOUNIO_COMPILER"):
            if env_var in os.environ:
                path = Path(os.environ[env_var])
                # If it points to the wrapper, try to resolve the native binary
                native = self._resolve_native_from_wrapper(path)
                if native and native.is_file() and os.access(native, os.X_OK):
                    return str(native)
                if path.is_file() and os.access(path, os.X_OK):
                    return str(path)

        # 2. Repo root native binary
        repo_native = repo_root / "bin" / "souc-linux-x86_64"
        if repo_native.is_file() and os.access(repo_native, os.X_OK):
            return str(repo_native)

        # 3. Wrapper in repo root
        repo_wrapper = repo_root / "bin" / "souc"
        native = self._resolve_native_from_wrapper(repo_wrapper)
        if native and native.is_file() and os.access(native, os.X_OK):
            return str(native)

        # 4. PATH lookup for native binary
        for name in ("souc-linux-x86_64", "souc"):
            try:
                result = subprocess.run(
                    ["which", name],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    path = Path(result.stdout.strip())
                    if name == "souc":
                        native = self._resolve_native_from_wrapper(path)
                        if native and native.is_file() and os.access(native, os.X_OK):
                            return str(native)
                    if path.is_file() and os.access(path, os.X_OK):
                        return str(path)
            except Exception:
                pass

        return None

    @staticmethod
    def _repo_root() -> Path:
        """Return the repository root path.

        Works regardless of the current working directory by walking up from
        this file's location.
        """
        here = Path(__file__).resolve()
        # Walk up until we find a marker (bin/, stdlib/, .git/)
        for parent in here.parents:
            if any((parent / marker).exists() for marker in ("bin", "stdlib", ".git")):
                return parent
        # Fallback: go up 4 levels from sounio_kernel/executor.py
        return here.parents[4]

    @staticmethod
    def _resolve_native_from_wrapper(wrapper: Path) -> Optional[Path]:
        """If *wrapper* is the bash wrapper, return the native binary it delegates to."""
        if not wrapper.is_file():
            return None
        # The wrapper typically lives next to souc-linux-x86_64 in the same dir
        native = wrapper.parent / "souc-linux-x86_64"
        if native.is_file() and os.access(native, os.X_OK):
            return native
        return None

    def _find_stdlib_path(self) -> Optional[str]:
        """
        Find Sounio stdlib path.

        Priority:
        1. SOUNIO_STDLIB_PATH environment variable
        2. Repository-root stdlib path
        3. Current working directory stdlib path
        """
        repo_root = self._repo_root()

        # Check env var
        if "SOUNIO_STDLIB_PATH" in os.environ:
            return os.environ["SOUNIO_STDLIB_PATH"]

        # Check repository-root location independent of caller cwd.
        repo_stdlib = repo_root / "stdlib"
        if repo_stdlib.is_dir():
            return str(repo_stdlib)

        # Check current working directory (legacy behavior).
        default_stdlib = Path("stdlib")
        if default_stdlib.is_dir():
            return str(default_stdlib.resolve())

        return None
