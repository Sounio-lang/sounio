#!/usr/bin/env python3
"""Sounio verification MCP — "no sorry, no mathlib" enforcement bridge.

Pure-stdlib JSON-RPC 2.0 over stdio (newline-delimited). No external deps.
Tools expose ground-truth verification for the Erdős/dissertation work:

  lean_check   — type-check a Lean file (no-mathlib build) + static sorry/Mathlib audit
  lean_axioms  — RIGOROUS doctrine gate: `#print axioms <thm>` → detects sorryAx
                 (hidden sorry) and Mathlib axioms; the trusted base, not a grep
  souc_check   — run `souc check|run` on a .sio (E170 unwrap-safety, epistemic, proofs)

All file paths are confined to the dissertation root. Subprocesses are timed out.
"""
import json, os, re, subprocess, sys, tempfile

DISS   = os.environ.get("SOUNIO_DISS", "/workspace/sounio-dissertation")
LEAN4  = os.path.join(DISS, "formal", "lean4")
ELAN   = os.environ.get("ELAN_BIN", "/workspace/.home/openvscode-server/.elan/bin")
SOUC   = os.path.join(DISS, "bin", "souc")
ENV    = {**os.environ, "PATH": ELAN + ":" + os.environ.get("PATH", ""),
          "SOUNIO_STDLIB_PATH": os.path.join(DISS, "stdlib")}

def _confine(path, base):
    p = path if os.path.isabs(path) else os.path.join(base, path)
    p = os.path.realpath(p)
    if not p.startswith(os.path.realpath(DISS)):
        raise ValueError(f"path escapes dissertation root: {path}")
    return p

def _run(cmd, cwd, timeout=180):
    try:
        r = subprocess.run(cmd, cwd=cwd, env=ENV, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s"
    except FileNotFoundError as e:
        return 127, f"not found: {e}"

# real sorry = `sorry` token not inside a comment/docstring/string "no sorry" prose
_SORRY = re.compile(r'(^|[^\w`"])sorry\b')
def _static_audit(text):
    real_sorry = 0
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith(("--", "/-", "#", "*", "-/")):  # comment/doc line
            continue
        # strip trailing line comment
        code = ln.split("--", 1)[0]
        if _SORRY.search(code) and "`sorry`" not in ln:
            real_sorry += 1
    mathlib = len(re.findall(r'^\s*import\s+Mathlib', text, re.M))
    return real_sorry, mathlib

# ---- tools -----------------------------------------------------------------
def lean_check(args):
    f = _confine(args["file"], LEAN4)
    text = open(f, encoding="utf-8", errors="replace").read()
    real_sorry, mathlib = _static_audit(text)
    rc, out = _run([os.path.join(ELAN, "lake"), "env", "lean", f], cwd=LEAN4,
                   timeout=int(args.get("timeout", 240)))
    ok = rc == 0
    return {
        "file": os.path.relpath(f, DISS),
        "builds": ok, "exit": rc,
        "sorry_free": real_sorry == 0, "real_sorry_count": real_sorry,
        "mathlib_free": mathlib == 0, "mathlib_imports": mathlib,
        "diagnostics": out.strip()[-4000:],
        "doctrine_ok": ok and real_sorry == 0 and mathlib == 0,
    }

def lean_axioms(args):
    f = _confine(args["file"], LEAN4)
    names = args["names"] if isinstance(args.get("names"), list) else [args["names"]]
    src = open(f, encoding="utf-8", errors="replace").read()
    # open all declared namespaces so unqualified theorem names resolve
    namespaces = list(dict.fromkeys(re.findall(r'^\s*namespace\s+(\S+)', src, re.M)))
    opens = "".join(f"open {ns}\n" for ns in namespaces)
    probe = src + "\n\n-- verify_mcp axiom audit --\n" + opens + \
            "".join(f"#print axioms {n}\n" for n in names)
    tmp = tempfile.NamedTemporaryFile("w", dir=LEAN4, suffix=".lean", delete=False, encoding="utf-8")
    tmp.write(probe); tmp.close()
    try:
        rc, out = _run([os.path.join(ELAN, "lake"), "env", "lean", tmp.name], cwd=LEAN4,
                       timeout=int(args.get("timeout", 240)))
    finally:
        os.unlink(tmp.name)
    has_sorry = "sorryAx" in out
    has_mathlib = bool(re.search(r'\bMathlib\.', out))
    return {
        "file": os.path.relpath(f, DISS), "theorems": names, "exit": rc,
        "depends_on_sorry": has_sorry, "depends_on_mathlib_axioms": has_mathlib,
        "doctrine_ok": rc == 0 and not has_sorry and not has_mathlib,
        "axiom_report": out.strip()[-4000:],
    }

def souc_check(args):
    f = _confine(args["file"], DISS)
    mode = args.get("mode", "check")
    if mode not in ("check", "run"):
        raise ValueError("mode must be 'check' or 'run'")
    rc, out = _run([SOUC, mode, f], cwd=DISS, timeout=int(args.get("timeout", 120)))
    return {"file": os.path.relpath(f, DISS), "mode": mode, "exit": rc,
            "ok": rc == 0, "output": out.strip()[-6000:]}

TOOLS = {
    "lean_check": (lean_check,
        "Type-check a Lean file via `lake env lean` (no-mathlib build) and statically audit it. "
        "Returns builds/sorry_free/mathlib_free/doctrine_ok + diagnostics.",
        {"type": "object", "properties": {
            "file": {"type": "string", "description": "Lean file, relative to formal/lean4 or absolute (confined to dissertation)"},
            "timeout": {"type": "integer", "description": "seconds (default 240)"}},
         "required": ["file"]}),
    "lean_axioms": (lean_axioms,
        "RIGOROUS doctrine gate: run `#print axioms` on named theorems. Detects sorryAx (hidden sorry) "
        "and Mathlib axiom dependencies — the trusted base, stronger than grep. Returns doctrine_ok.",
        {"type": "object", "properties": {
            "file": {"type": "string"},
            "names": {"type": "array", "items": {"type": "string"}, "description": "theorem/def names to audit"},
            "timeout": {"type": "integer"}},
         "required": ["file", "names"]}),
    "souc_check": (souc_check,
        "Run `souc check|run` on a .sio file: type/effect errors (e.g. E170 unwrap-safety), epistemic "
        "output, proof results. Returns ok + output.",
        {"type": "object", "properties": {
            "file": {"type": "string", "description": ".sio file, relative to dissertation root or absolute"},
            "mode": {"type": "string", "enum": ["check", "run"], "description": "default check"},
            "timeout": {"type": "integer"}},
         "required": ["file"]}),
}

# ---- JSON-RPC / MCP stdio loop --------------------------------------------
def _resp(id_, result=None, error=None):
    m = {"jsonrpc": "2.0", "id": id_}
    if error is not None: m["error"] = error
    else: m["result"] = result
    sys.stdout.write(json.dumps(m) + "\n"); sys.stdout.flush()

def main():
    proto = "2025-06-18"
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try: msg = json.loads(line)
        except Exception: continue
        method, id_, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
        if method == "initialize":
            proto_req = params.get("protocolVersion", proto)
            _resp(id_, {"protocolVersion": proto_req,
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "sounio-verify", "version": "1.0.0"}})
        elif method == "notifications/initialized":
            continue
        elif method == "ping":
            _resp(id_, {})
        elif method == "tools/list":
            _resp(id_, {"tools": [{"name": n, "description": d, "inputSchema": s}
                                  for n, (_fn, d, s) in TOOLS.items()]})
        elif method == "tools/call":
            name = params.get("name"); args = params.get("arguments") or {}
            if name not in TOOLS:
                _resp(id_, error={"code": -32601, "message": f"unknown tool {name}"}); continue
            try:
                out = TOOLS[name][0](args)
                _resp(id_, {"content": [{"type": "text", "text": json.dumps(out, indent=2)}]})
            except Exception as e:
                _resp(id_, {"content": [{"type": "text", "text": f"ERROR: {type(e).__name__}: {e}"}],
                            "isError": True})
        elif id_ is not None:
            _resp(id_, error={"code": -32601, "message": f"method not found: {method}"})

if __name__ == "__main__":
    main()
