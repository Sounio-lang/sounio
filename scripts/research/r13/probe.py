#!/usr/bin/env python3
"""Run one contract with a source-level patch installed after a named function.

The patch text arrives in PROBE_PATCH; it is inserted immediately after the
target function's module-level definition, so tables built at import time are
also affected (a post-import monkey-patch would be too late).

Writes JSON to PROBE_OUT and leaves via os._exit(0): the file is already
flushed and fsynced, so there is no interpreter shutdown left to fail on a
descriptor the probed code may have closed (R11 SS3, hazards 3-5).
"""
from __future__ import annotations

import ast, contextlib, io, json, os, re, resource, sys

VERDICT_RE = re.compile(r"^([A-Z0-9_]*_VERDICT) (\S+)", re.M)
MEM_CAP = 8 * 1024 ** 3          # RLIMIT_AS, so runaway allocation raises
                                 # inside our try instead of killing the process


def inject(src: str, fn: str, patch: str) -> str | None:
    if not patch:
        return src
    tree = ast.parse(src)
    target = None
    for node in tree.body:                       # module level only
        if isinstance(node, ast.FunctionDef) and node.name == fn:
            target = node
    if target is None:
        return None
    lines = src.splitlines(keepends=True)
    return "".join(lines[:target.end_lineno]) + patch + "\n" + \
           "".join(lines[target.end_lineno:])


def main() -> None:
    path, fn = sys.argv[1], sys.argv[2]
    out = {"contract": os.path.basename(path), "fn": fn,
           "verdict": None, "error": None}
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEM_CAP, MEM_CAP))
    except Exception:                                            # noqa: BLE001
        pass
    try:
        src = open(path, encoding="utf8", errors="replace").read()
        patched = inject(src, fn, os.environ.get("PROBE_PATCH", ""))
        if patched is None:
            out["error"] = f"no module-level def {fn}"
        else:
            g = {"__name__": "__main__", "__file__": os.path.abspath(path)}
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    exec(compile(patched, path, "exec"), g)
            except SystemExit:
                pass
            m = VERDICT_RE.search(buf.getvalue())
            if m:
                out["verdict"] = m.group(2)
            else:
                out["error"] = "no verdict token"
    except BaseException as exc:                                 # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]

    fd = os.open(os.environ["PROBE_OUT"], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.write(fd, json.dumps(out).encode())
    os.fsync(fd)
    os.close(fd)
    os._exit(0)


if __name__ == "__main__":
    main()
