"""kernel_client — launch and communicate with a live Sounio Jupyter kernel.

Requires jupyter_client and the sounio-jupyter kernel to be installed:

    pip install jupyter_client
    python -m sounio_kernel install

Usage
-----
>>> import sounio
>>> with sounio.launch_jupyter_kernel() as k:
...     result = k.execute("let x: f64 = 42.0\\nprintln(x)")
...     print(result.stdout)
42.0
...     print(result.knowledge_values)
[]

>>> # Get Knowledge values from Sounio epistemic output
>>> with sounio.launch_jupyter_kernel() as k:
...     values = k.get_knowledge_values(
...         'let k: Knowledge<f64> = measure(500.0, uncertainty: 2.5, prov: "sensor")'
...     )
...     print(values[0])
Knowledge(500.000 ± 2.500, prov='sensor')
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .knowledge import Knowledge


# Regex matching the canonical souc Knowledge output format.
_KNOWLEDGE_RE = re.compile(
    r'Knowledge\s*\{\s*value:\s*([0-9eE+\-.]+)\s+'
    r'epsilon:\s*([0-9eE+\-.]+)\s+'
    r'prov:\s*"([^"]*)"\s*\}'
)


@dataclass
class KernelResult:
    """Result of executing a code cell in the Sounio Jupyter kernel.

    Attributes
    ----------
    stdout : str
        Standard output produced by the execution.
    stderr : str
        Standard error output.
    status : str
        "ok" or "error".
    knowledge_values : list[Knowledge]
        Epistemic values parsed from stdout.
    """

    stdout: str = ""
    stderr: str = ""
    status: str = "ok"
    knowledge_values: List[Knowledge] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when status is "ok"."""
        return self.status == "ok"

    def __repr__(self) -> str:
        return (
            f"KernelResult(status={self.status!r}, "
            f"knowledge_values={len(self.knowledge_values)})"
        )


class KernelConnection:
    """A live connection to a running Sounio Jupyter kernel.

    Obtain via :func:`launch_jupyter_kernel`. Use as a context manager
    to ensure the kernel is shut down when done.

    Examples
    --------
    >>> with launch_jupyter_kernel() as k:
    ...     result = k.execute("let x = 1 + 2")
    ...     print(result.stdout)
    """

    def __init__(self, km, kc) -> None:
        self._km = km
        self._kc = kc

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------

    def execute(self, code: str, timeout: float = 30.0) -> KernelResult:
        """Send Sounio code to the kernel and collect all output.

        Parameters
        ----------
        code : str
            Sounio source code to run (functions, expressions, etc.).
        timeout : float
            Seconds to wait for the kernel to become idle after execution.

        Returns
        -------
        KernelResult
            Contains stdout, stderr, status, and parsed Knowledge values.
        """
        self._kc.execute(code)

        stdout_parts: List[str] = []
        stderr_parts: List[str] = []
        status = "ok"

        while True:
            try:
                msg = self._kc.get_iopub_msg(timeout=timeout)
            except Exception:
                # Timeout or channel error — collect whatever arrived
                break

            msg_type = msg.get("msg_type", "")
            content = msg.get("content", {})

            if msg_type == "stream":
                if content.get("name") == "stdout":
                    stdout_parts.append(content.get("text", ""))
                elif content.get("name") == "stderr":
                    stderr_parts.append(content.get("text", ""))
            elif msg_type == "error":
                status = "error"
                tb = content.get("traceback", [])
                # Strip ANSI colour codes for clean text output
                clean = [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in tb]
                stderr_parts.append("\n".join(clean))
            elif msg_type == "execute_reply":
                if content.get("status") == "error":
                    status = "error"
            elif msg_type == "status":
                if content.get("execution_state") == "idle":
                    break

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)
        knowledge_values = [
            Knowledge(float(v), float(e), prov)
            for v, e, prov in _KNOWLEDGE_RE.findall(stdout)
        ]

        return KernelResult(
            stdout=stdout,
            stderr=stderr,
            status=status,
            knowledge_values=knowledge_values,
        )

    def get_knowledge_values(
        self, code: str, timeout: float = 30.0
    ) -> List[Knowledge]:
        """Execute code and return only the parsed Knowledge values."""
        return self.execute(code, timeout=timeout).knowledge_values

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Shut down the kernel process."""
        try:
            self._kc.stop_channels()
        except Exception:
            pass
        try:
            self._km.shutdown_kernel(now=True)
        except Exception:
            pass

    def __enter__(self) -> "KernelConnection":
        return self

    def __exit__(self, *_) -> None:
        self.shutdown()

    def __repr__(self) -> str:
        return f"KernelConnection(kernel_name={self._km.kernel_name!r})"


# ------------------------------------------------------------------
# Public factory
# ------------------------------------------------------------------


def launch_jupyter_kernel(timeout: float = 30.0) -> KernelConnection:
    """Start a Sounio Jupyter kernel and return a live connection.

    Requires ``jupyter_client`` (``pip install jupyter_client``) and the
    sounio kernel spec (``python -m sounio_kernel install``).

    Parameters
    ----------
    timeout : float
        Seconds to wait for the kernel to become ready (default 30).

    Returns
    -------
    KernelConnection
        Use as a context manager or call ``.shutdown()`` explicitly.

    Raises
    ------
    ImportError
        If ``jupyter_client`` is not installed.
    RuntimeError
        If the sounio kernel spec is not installed or fails to start.

    Examples
    --------
    >>> with launch_jupyter_kernel() as k:
    ...     r = k.execute('fn main() with IO { println("hello from sounio") }')
    ...     print(r.stdout)
    hello from sounio
    """
    try:
        from jupyter_client import KernelManager
    except ImportError as exc:
        raise ImportError(
            "jupyter_client is required for launch_jupyter_kernel(). "
            "Install with: pip install jupyter_client"
        ) from exc

    km = KernelManager(kernel_name="sounio")
    try:
        km.start_kernel()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to start Sounio Jupyter kernel: {exc}\n"
            "Make sure the kernel is installed: python -m sounio_kernel install"
        ) from exc

    kc = km.client()
    kc.start_channels()

    try:
        kc.wait_for_ready(timeout=timeout)
    except RuntimeError as exc:
        try:
            kc.stop_channels()
            km.shutdown_kernel(now=True)
        except Exception:
            pass
        raise RuntimeError(
            f"Sounio kernel did not become ready within {timeout}s: {exc}"
        ) from exc

    return KernelConnection(km, kc)
