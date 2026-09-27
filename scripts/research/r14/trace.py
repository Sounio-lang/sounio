#!/usr/bin/env python3
"""Record which (a, b, bits) each contract actually queries of the CD sign table.

The battery measured which perturbations move a contract's verdict. This
measures which parts of the shared object it CONSULTS. The two together give a
question nobody has asked of a research corpus:

    of everything this contract computes, how much of it is load-bearing for
    what it concludes?

A contract that queries thousands of basis products while its verdict depends on
three of them rests on a far narrower base than its runtime suggests. A contract
that queries a whole level where nothing is load-bearing is computing decoration.

Writes {contract: {"bits": {L: n_distinct_pairs}, "pairs": [[a,b,bits], ...]}}.
Child process, JSON out, os._exit (R11 SS3).
"""
from __future__ import annotations

import ast, contextlib, io, json, os, re, resource, sys

VERDICT_RE = re.compile(r"^([A-Z0-9_]*_VERDICT) (\S+)", re.M)
MEM_CAP = 8 * 1024 ** 3
CAP = 400_000                      # distinct-tuple cap; beyond this we stop
                                   # recording and say so rather than OOM

TRACER = """
_orig_{n} = {n}
import builtins as _b
_b._CD_SEEN = set()
_b._CD_OVER = [False]
def _bits_of_{n}(r, k):
    if 'bits' in k:
        return k['bits']
    if r:
        return r[0]
    d = _orig_{n}.__defaults__
    return d[-1] if d else None
def {n}(a, b, *r, **k):
    if not _b._CD_OVER[0]:
        if len(_b._CD_SEEN) < {cap}:
            _b._CD_SEEN.add((a, b, _bits_of_{n}(r, k)))
        else:
            _b._CD_OVER[0] = True
    return _orig_{n}(a, b, *r, **k)
"""


def inject(src: str, fn: str, patch: str) -> str | None:
    tree = ast.parse(src)
    target = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn:
            target = node
    if target is None:
        return None
    lines = src.splitlines(keepends=True)
    return "".join(lines[:target.end_lineno]) + patch + "\n" + \
           "".join(lines[target.end_lineno:])


def main() -> None:
    path, fn = sys.argv[1], sys.argv[2]
    out = {"contract": os.path.basename(path), "fn": fn, "verdict": None,
           "error": None, "overflow": False, "by_bits": {}, "pairs": []}
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEM_CAP, MEM_CAP))
    except Exception:
        pass
    try:
        src = open(path, encoding="utf8", errors="replace").read()
        patched = inject(src, fn, TRACER.format(n=fn, cap=CAP))
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
            out["verdict"] = m.group(2) if m else None
            import builtins as _b
            seen = getattr(_b, "_CD_SEEN", set())
            out["overflow"] = bool(getattr(_b, "_CD_OVER", [False])[0])
            by = {}
            for a, b, bits in seen:
                by.setdefault(str(bits), 0)
                by[str(bits)] += 1
            out["by_bits"] = by
            out["pairs"] = sorted([list(t) for t in seen])[:CAP]
    except BaseException as exc:                                # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]

    fd = os.open(os.environ["PROBE_OUT"], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.write(fd, json.dumps(out).encode())
    os.fsync(fd)
    os.close(fd)
    os._exit(0)


if __name__ == "__main__":
    main()
