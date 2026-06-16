"""
sounio._compile — JIT compilation bridge for Sounio code from Python.

Allows running .sio files and inline Sounio code directly from Python.
This is a stub implementation — full JIT via the native souc binary
will be wired once the C ABI / cdylib bridge is available.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union


SOUC_BIN_ENV = "SOUC_BIN"
SOUC_STDLIB_ENV = "SOUNIO_STDLIB_PATH"

_default_souc = Path(__file__).parent.parent.parent.parent.parent / "bin" / "souc"


def _find_souc() -> str:
    """Locate the souc compiler binary."""
    if env_path := os.environ.get(SOUC_BIN_ENV):
        return env_path
    if _default_souc.exists():
        return str(_default_souc)
    raise RuntimeError(
        "souc compiler not found. Set SOUC_BIN environment variable "
        "or install via: pip install sounio[native]"
    )


class CompiledModule:
    """
    A compiled Sounio module runnable from Python.

    Created by sounio.compile(). Provides a .run() method to execute
    the compiled binary and capture output.
    """

    def __init__(self, source_path: str, binary_path: Optional[str] = None):
        self._source_path = source_path
        self._binary_path = binary_path
        self._compiled = binary_path is not None

    def run(self, *args: str, capture: bool = True) -> "RunResult":
        """
        Execute the compiled module.

        Args:
            *args: command-line arguments to pass to the binary
            capture: if True, capture stdout/stderr

        Returns:
            RunResult with stdout, stderr, and exit_code
        """
        if not self._compiled or self._binary_path is None:
            raise RuntimeError(
                "Module is not compiled. Use sounio.compile() to compile first."
            )
        cmd = [self._binary_path] + list(args)
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
        )
        return RunResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
        )

    def __repr__(self) -> str:
        status = "compiled" if self._compiled else "stub"
        return f"CompiledModule(source={self._source_path!r}, status={status})"


class RunResult:
    """Result of running a Sounio binary."""

    def __init__(self, stdout: str, stderr: str, exit_code: int):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.ok = exit_code == 0

    def __repr__(self) -> str:
        return f"RunResult(exit_code={self.exit_code}, ok={self.ok})"


def compile(
    source: Union[str, Path],
    output: Optional[Union[str, Path]] = None,
    optimize: bool = False,
) -> CompiledModule:
    """
    Compile a Sounio source file to a native binary.

    Args:
        source:   path to .sio file or inline Sounio code string
        output:   path for compiled binary (defaults to temp file)
        optimize: enable compiler optimizations

    Returns:
        CompiledModule that can be .run()

    Example::

        model = sounio.compile("models/pbpk_epistemic.sio")
        result = model.run()
        print(result.stdout)
    """
    source_path = str(source)
    is_inline = not source_path.endswith(".sio") and "\n" in source_path

    if is_inline:
        tmp_src = tempfile.NamedTemporaryFile(suffix=".sio", delete=False, mode="w")
        tmp_src.write(source_path)
        tmp_src.close()
        source_path = tmp_src.name

    try:
        souc = _find_souc()
    except RuntimeError as e:
        print(f"[sounio.compile] Warning: {e}")
        return CompiledModule(source_path, binary_path=None)

    if output is None:
        tmp_bin = tempfile.NamedTemporaryFile(delete=False, suffix=".out")
        tmp_bin.close()
        output_path = tmp_bin.name
    else:
        output_path = str(output)

    cmd = [souc, "run", source_path, "-o", output_path]
    if optimize:
        cmd.append("-O")

    env = os.environ.copy()
    if stdlib := os.environ.get(SOUC_STDLIB_ENV):
        env[SOUC_STDLIB_ENV] = stdlib

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"Compilation failed:\n{result.stderr}"
        )

    return CompiledModule(source_path, binary_path=output_path)


def run(
    source: Union[str, Path],
    *args: str,
    optimize: bool = False,
    capture: bool = True,
) -> RunResult:
    """
    Compile and immediately run a Sounio source file.

    Convenience wrapper around compile() + CompiledModule.run().

    Example::

        result = sounio.run("models/pbpk.sio")
        print(result.stdout)
    """
    module = compile(source, optimize=optimize)
    return module.run(*args, capture=capture)
