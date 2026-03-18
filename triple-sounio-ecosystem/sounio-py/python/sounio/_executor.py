"""SounioExecutor — subprocess wrapper around the souc JIT binary.

Exposes run_file, run_code, and check_file with structured return types.
Knowledge values printed by the running program are parsed from stdout
using the canonical souc format:

    Knowledge { value: 42.000 epsilon: 0.100 prov: "name" }
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from .knowledge import Knowledge


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class ExecutionResult:
    """Outcome of running a .sio file or inline code.

    Attributes
    ----------
    stdout : str
        Combined standard output of the souc process.
    stderr : str
        Standard error output (warnings, diagnostics).
    exit_code : int
        Process exit status (0 = success).
    knowledge_values : list[Knowledge]
        All Knowledge values parsed from stdout.
    """

    def __init__(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        knowledge_values: Optional[List[Knowledge]] = None,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.knowledge_values = knowledge_values or []

    @property
    def ok(self) -> bool:
        """True when exit_code is 0."""
        return self.exit_code == 0

    def __repr__(self) -> str:
        return (
            f"ExecutionResult(exit_code={self.exit_code}, "
            f"knowledge_values={len(self.knowledge_values)})"
        )


class CheckResult:
    """Outcome of type-checking a .sio file.

    Attributes
    ----------
    success : bool
        True when the file passed type-checking.
    errors : str
        Raw stderr content; empty when success is True.
    ast : str or None
        AST dump when ``--show-ast`` was requested.
    types : str or None
        Type dump when ``--show-types`` was requested.
    """

    def __init__(
        self,
        success: bool,
        errors: str,
        ast: Optional[str] = None,
        types: Optional[str] = None,
    ) -> None:
        self.success = success
        self.errors = errors
        self.ast = ast
        self.types = types

    def __repr__(self) -> str:
        return f"CheckResult(success={self.success})"


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

# Regex matching the canonical souc Knowledge output format.
_KNOWLEDGE_RE = re.compile(
    r'Knowledge\s*\{\s*value:\s*([0-9eE+\-.]+)\s+'
    r'epsilon:\s*([0-9eE+\-.]+)\s+'
    r'prov:\s*"([^"]*)"\s*\}'
)


class SounioExecutor:
    """Execute Sounio code via the souc JIT binary.

    Parameters
    ----------
    souc_path : str, optional
        Explicit path to the souc binary. When omitted the constructor looks up
        (in order):
        1. ``SOUC`` environment variable.
        2. ``SOUNIO_SOUC_PATH`` environment variable.
        3. Repo-relative default path.
    stdlib_path : str, optional
        Path to the Sounio stdlib.  Defaults to ``SOUNIO_STDLIB_PATH`` env var
        or the repo-relative ``stdlib/`` directory.
    """

    # Repo-relative default (works when cwd is the repo root or sounio-py/).
    _DEFAULT_SOUC_CANDIDATES = [
        # Running from sounio-py/ inside the repo
        "../../../../artifacts/omega/souc-bin/souc-linux-x86_64-jit",
        # Running from repo root
        "artifacts/omega/souc-bin/souc-linux-x86_64-jit",
    ]
    _DEFAULT_STDLIB_CANDIDATES = [
        "../../../../stdlib",
        "stdlib",
    ]

    def __init__(
        self,
        souc_path: Optional[str] = None,
        stdlib_path: Optional[str] = None,
    ) -> None:
        self.souc_path = souc_path or self._resolve_souc()
        self.stdlib_path = stdlib_path or self._resolve_stdlib()

    # ---- Resolution helpers -----------------------------------------------

    @classmethod
    def _resolve_souc(cls) -> str:
        for var in ("SOUC", "SOUNIO_SOUC_PATH"):
            val = os.environ.get(var)
            if val and Path(val).exists():
                return val
        for rel in cls._DEFAULT_SOUC_CANDIDATES:
            p = Path(rel)
            if p.exists():
                return str(p.resolve())
        # Return the first candidate even if missing; error surfaces at run time.
        return cls._DEFAULT_SOUC_CANDIDATES[-1]

    @classmethod
    def _resolve_stdlib(cls) -> str:
        val = os.environ.get("SOUNIO_STDLIB_PATH")
        if val:
            return val
        for rel in cls._DEFAULT_STDLIB_CANDIDATES:
            p = Path(rel)
            if p.exists():
                return str(p.resolve())
        return cls._DEFAULT_STDLIB_CANDIDATES[-1]

    # ---- Core I/O ---------------------------------------------------------

    def _build_env(self) -> dict:
        env = dict(os.environ)
        env["SOUNIO_STDLIB_PATH"] = self.stdlib_path
        return env

    @staticmethod
    def _parse_knowledge(stdout: str) -> List[Knowledge]:
        """Extract all Knowledge values from souc stdout."""
        return [
            Knowledge(float(v), float(e), prov)
            for v, e, prov in _KNOWLEDGE_RE.findall(stdout)
        ]

    # ---- Public API -------------------------------------------------------

    def run_file(self, path: str, timeout: int = 30) -> ExecutionResult:
        """JIT-run a .sio file and return a structured result.

        Parameters
        ----------
        path : str
            Absolute or relative path to the ``.sio`` source file.
        timeout : int
            Maximum seconds to wait for the process (default 30).
        """
        proc = subprocess.run(
            [self.souc_path, "run", path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=self._build_env(),
        )
        return ExecutionResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
            knowledge_values=self._parse_knowledge(proc.stdout),
        )

    def run_code(self, code: str, timeout: int = 30) -> ExecutionResult:
        """Run inline Sounio source code.

        The code is written to a temporary ``.sio`` file and then executed via
        ``souc run``.  The file is deleted afterwards.

        Parameters
        ----------
        code : str
            Sounio source code to run.
        timeout : int
            Maximum seconds to wait for the process.
        """
        with tempfile.NamedTemporaryFile(
            suffix=".sio", mode="w", delete=False
        ) as fh:
            fh.write(code)
            tmp_path = fh.name
        try:
            return self.run_file(tmp_path, timeout=timeout)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def check_file(
        self,
        path: str,
        show_ast: bool = False,
        show_types: bool = False,
    ) -> CheckResult:
        """Type-check a .sio file without executing it.

        Parameters
        ----------
        path : str
            Path to the ``.sio`` file.
        show_ast : bool
            If True, pass ``--show-ast`` and capture the AST dump.
        show_types : bool
            If True, pass ``--show-types`` and capture the types dump.
        """
        cmd = [self.souc_path, "check", path]
        if show_ast:
            cmd.append("--show-ast")
        if show_types:
            cmd.append("--show-types")

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            env=self._build_env(),
        )
        ast_dump = proc.stdout if show_ast else None
        types_dump = proc.stdout if show_types else None
        return CheckResult(
            success=proc.returncode == 0,
            errors=proc.stderr,
            ast=ast_dump,
            types=types_dump,
        )

    # ---- Convenience repr ------------------------------------------------

    def __repr__(self) -> str:
        return f"SounioExecutor(souc={self.souc_path!r})"
