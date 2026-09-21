"""
Madaros compiler bridge.

Auto-detects environment:
- If MADAROS_BIN exists locally → runs madaros directly (remote workspace)
- Otherwise → tunnels via SSH to sounio-workspace-control (local Mac)
"""

import asyncio
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field

_DEFAULT_BIN = "/workspace/sounio/bin/madaros"
MADAROS_BIN  = (os.environ.get("SOUNIO_MADAROS_BIN") or
                os.environ.get("MADAROS_RAW_BIN") or
                _DEFAULT_BIN)
SSH_HOST    = "sounio-workspace-control"
SSH_SOCKET  = os.path.expanduser("~/.ssh/sockets/sounio-workspace-control.sock")

# True when running on a host that has the madaros binary locally
_LOCAL = os.path.isfile(MADAROS_BIN)


@dataclass
class CheckResult:
    ok: bool
    time_ms: float
    raw: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _parse_output(raw: str, exit_code: int) -> tuple[bool, list[str], list[str]]:
    errors, warnings = [], []
    for line in raw.splitlines():
        l = line.strip()
        if not l:
            continue
        low = l.lower()
        if "error" in low or "failed" in low or "unexpected" in low:
            errors.append(l)
        elif "warning" in low or "warn" in low:
            warnings.append(l)
    ok = exit_code == 0 and "check: OK" in raw
    return ok, errors, warnings


def _tmp_path() -> str:
    tid = threading.get_ident() % 99991
    return f"/tmp/mc_{os.getpid()}_{tid}.sio"


class MadarosCompiler:
    def __init__(self, madaros_bin: str = MADAROS_BIN,
                 ssh_host: str = SSH_HOST, timeout: float = 15.0):
        self.madaros_bin = madaros_bin
        self.ssh_host    = ssh_host
        self.timeout     = timeout

    def check(self, code: str) -> CheckResult:
        t0 = time.monotonic()
        if _LOCAL:
            return self._check_local(code, t0)
        return self._check_ssh(code, t0)

    def _check_local(self, code: str, t0: float) -> CheckResult:
        path = _tmp_path()
        try:
            with open(path, "w") as f:
                f.write(code)
            proc = subprocess.run(
                [self.madaros_bin, "check", path],
                capture_output=True, timeout=self.timeout,
            )
            elapsed = (time.monotonic() - t0) * 1000
        except subprocess.TimeoutExpired:
            return CheckResult(ok=False, time_ms=self.timeout * 1000,
                               raw="timeout", errors=["Timeout"])
        except Exception as e:
            return CheckResult(ok=False, time_ms=0, raw=str(e), errors=[str(e)])
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        raw = proc.stdout.decode() + proc.stderr.decode()
        ok, errors, warnings = _parse_output(raw, proc.returncode)
        return CheckResult(ok=ok, time_ms=elapsed, raw=raw,
                           errors=errors, warnings=warnings)

    def _check_ssh(self, code: str, t0: float) -> CheckResult:
        import shlex
        path    = _tmp_path()
        inline  = shlex.quote(code)
        cmd_str = (f"printf %s {inline} > {path} && "
                   f"{self.madaros_bin} check {path}; rm -f {path}")
        ssh_cmd = ["ssh", "-S", SSH_SOCKET, self.ssh_host, cmd_str]
        try:
            proc = subprocess.run(ssh_cmd, capture_output=True, timeout=self.timeout)
            elapsed = (time.monotonic() - t0) * 1000
        except subprocess.TimeoutExpired:
            return CheckResult(ok=False, time_ms=self.timeout * 1000,
                               raw="timeout", errors=["Timeout"])
        except Exception as e:
            return CheckResult(ok=False, time_ms=0, raw=str(e), errors=[str(e)])
        raw = proc.stdout.decode() + proc.stderr.decode()
        ok, errors, warnings = _parse_output(raw, proc.returncode)
        return CheckResult(ok=ok, time_ms=elapsed, raw=raw,
                           errors=errors, warnings=warnings)


class AsyncMadarosCompiler:
    def __init__(self, madaros_bin: str = MADAROS_BIN,
                 ssh_host: str = SSH_HOST,
                 max_concurrent: int = 20, timeout: float = 15.0):
        # Local: 20 concurrent fine (no SSH overhead)
        # Remote via SSH: cap at 6 to avoid ControlMaster overload
        concurrency     = max_concurrent if _LOCAL else min(max_concurrent, 6)
        self.semaphore  = asyncio.Semaphore(concurrency)
        self.madaros_bin = madaros_bin
        self.ssh_host   = ssh_host
        self.timeout    = timeout

    async def check(self, code: str) -> CheckResult:
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            c    = MadarosCompiler(self.madaros_bin, self.ssh_host, self.timeout)
            return await loop.run_in_executor(None, c.check, code)

    async def check_batch(self, codes: list[str]) -> list[CheckResult]:
        return await asyncio.gather(*[self.check(c) for c in codes])


def smoke_test():
    mode = "LOCAL" if _LOCAL else f"SSH→{SSH_HOST}"
    print(f"mode: {mode}")
    c = MadarosCompiler()
    for name, code in [
        ("valid",   "fn main() -> i32 { return 42 }"),
        ("invalid", "fn broken { ??? }"),
    ]:
        r = c.check(code)
        print(f"[{name}] ok={r.ok} time={r.time_ms:.0f}ms")
        for e in r.errors:
            print(f"  ERR: {e}")


if __name__ == "__main__":
    smoke_test()
