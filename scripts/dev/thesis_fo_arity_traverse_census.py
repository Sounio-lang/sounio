#!/usr/bin/env python3
"""Transitive arity>2 helper traverse census for dissertation surfaces.

Counts surfaces that REACH a helper with >=3 parameters from main,
following same-file calls and resolved `use` imports. Not a fail count.

Limitations (printed at the end):
- comments stripped line-wise only (`//`), not block comments
- method calls `recv.foo(` are resolved if a unique fn/method named foo
  is in the loaded set; ambiguous names are flagged
- does not expand macros / generated code
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()

# Builtins / runtime — not FO-transfer helpers.
IGNORE_CALLS = {
    "print", "println", "print_f64", "print_int", "print_char",
    "measure", "variance_of", "uncertainty_of", "acknowledge",
    "require_confidence", "assert", "panic", "abs", "sqrt",
    "Some", "None", "Ok", "Err", "Box", "Some",
}

FN_DEF = re.compile(
    r"^(?:pub(?:\(crate\))?\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)",
    re.S | re.M,
)
IMPL_FN = re.compile(
    r"fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)",
    re.S,
)
USE_LINE = re.compile(
    r"^use\s+([A-Za-z0-9_:]+)(?:::\{([^}]+)\})?",
    re.M,
)
CALL = re.compile(r"(?<![\w.])([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def strip_line_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        if "//" in line:
            # keep //@ annotations out of code anyway
            line = line[: line.find("//")]
        out.append(line)
    return "\n".join(out)


def arity_of(params: str) -> int:
    params = params.strip()
    if not params:
        return 0
    # drop nested parens content roughly
    depth = 0
    cleaned = []
    for ch in params:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            cleaned.append(ch)
    parts = [p.strip() for p in "".join(cleaned).split(",") if p.strip()]
    return len(parts)


def extract_fns(text: str, path: str) -> dict[str, dict]:
    code = strip_line_comments(text)
    fns: dict[str, dict] = {}
    for m in FN_DEF.finditer(code):
        name, params = m.group(1), m.group(2)
        start = m.end()
        # body: from next { 
        brace = code.find("{", start)
        if brace < 0:
            body = ""
        else:
            depth = 0
            i = brace
            while i < len(code):
                if code[i] == "{":
                    depth += 1
                elif code[i] == "}":
                    depth -= 1
                    if depth == 0:
                        body = code[brace + 1 : i]
                        break
                i += 1
            else:
                body = code[brace + 1 :]
        calls = [c for c in CALL.findall(body) if c != name and c not in IGNORE_CALLS]
        fns[name] = {
            "arity": arity_of(params),
            "calls": calls,
            "path": path,
            "params": " ".join(params.split()),
        }
    return fns


def parse_uses(text: str) -> list[tuple[str, list[str]]]:
    out = []
    for m in USE_LINE.finditer(text):
        prefix, group = m.group(1), m.group(2)
        if group:
            names = [n.strip() for n in group.split(",") if n.strip()]
            out.append((prefix, names))
        else:
            # use foo::bar  → last segment is the name
            names = [prefix.split("::")[-1]]
            out.append((prefix, names))
    return out


def resolve_module(prefix: str) -> list[Path]:
    """Map use prefix to candidate files under stdlib/ and examples/."""
    parts = prefix.split("::")
    cands = []
    # stdlib/<parts>.sio or stdlib/<parts>/mod.sio or stdlib/<parts>/lib.sio
    rel = Path(*parts)
    for base in (ROOT / "stdlib", ROOT / "examples", ROOT / "tests" / "run-pass"):
        p = base / rel
        if p.with_suffix(".sio").is_file():
            cands.append(p.with_suffix(".sio"))
        if p.is_dir():
            for name in ("lib.sio", "mod.sio"):
                if (p / name).is_file():
                    cands.append(p / name)
            cands.extend(sorted(p.glob("*.sio")))
        # darwin_pbpk lives under stdlib/darwin_pbpk
        if parts[0] == "epistemic":
            q = ROOT / "stdlib" / "epistemic" / (parts[-1] + ".sio") if len(parts) > 1 else ROOT / "stdlib" / "epistemic"
            if q.is_file():
                cands.append(q)
            elif q.is_dir():
                cands.extend(sorted(q.glob("*.sio")))
    # unique
    seen = set()
    uniq = []
    for c in cands:
        if c.resolve() not in seen:
            seen.add(c.resolve())
            uniq.append(c)
    return uniq


def load_suite() -> list[tuple[str, Path]]:
    gate = (ROOT / "scripts/ci/dissertation_pbpk_suite_gate.sh").read_text()
    surfaces = []
    for arr in ("TESTS", "TESTS_SMOKE"):
        block = re.search(rf"^{arr}=\((.*?)^\)", gate, re.S | re.M)
        if not block:
            continue
        for line in block.group(1).splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # "name    path"
            m = re.match(r'"(\S+)\s+(\S+)"', line)
            if not m:
                continue
            name, rel = m.group(1), m.group(2)
            p = ROOT / rel
            if p.is_file():
                surfaces.append((name, p))
    return surfaces


def main() -> None:
    surfaces = load_suite()
    cache: dict[Path, dict[str, dict]] = {}

    def fns_of(path: Path) -> dict[str, dict]:
        path = path.resolve()
        if path not in cache:
            cache[path] = extract_fns(path.read_text(encoding="utf-8", errors="replace"), str(path.relative_to(ROOT)))
        return cache[path]

    rows = []
    for sname, spath in surfaces:
        text = spath.read_text(encoding="utf-8", errors="replace")
        local = extract_fns(text, str(spath.relative_to(ROOT)))
        cache[spath.resolve()] = local

        # import resolution: load candidate modules, merge named exports
        imported: dict[str, dict] = {}
        unresolved_use = []
        for prefix, names in parse_uses(text):
            mods = resolve_module(prefix)
            if not mods:
                unresolved_use.append(prefix)
                continue
            for n in names:
                found = None
                for mod in mods:
                    fns = fns_of(mod)
                    if n in fns:
                        found = fns[n]
                        break
                if found:
                    imported[n] = found
                else:
                    unresolved_use.append(f"{prefix}::{n}")

        # universe for lookup: local overrides import
        universe = dict(imported)
        universe.update(local)

        if "main" not in local:
            rows.append((sname, "NO_MAIN", [], unresolved_use, []))
            continue

        # BFS from main
        seen = set()
        q = deque(["main"])
        reachable = []
        ambig = []
        while q:
            cur = q.popleft()
            if cur in seen:
                continue
            seen.add(cur)
            info = universe.get(cur)
            if info is None:
                continue
            reachable.append((cur, info))
            for callee in info["calls"]:
                if callee in IGNORE_CALLS:
                    continue
                if callee in universe:
                    q.append(callee)
                else:
                    # maybe unique across loaded cache?
                    hits = []
                    for fns in cache.values():
                        if callee in fns:
                            hits.append(fns[callee])
                    if len(hits) == 1:
                        universe[callee] = hits[0]
                        q.append(callee)
                    elif len(hits) > 1:
                        ambig.append(callee)

        hot = [(n, i) for n, i in reachable if n != "main" and i["arity"] >= 3]
        max_a = max((i["arity"] for _, i in hot), default=0)
        rows.append((sname, "OK", hot, unresolved_use, ambig, max_a, [n for n, _ in reachable if n != "main"]))

    traverse = [r for r in rows if r[1] == "OK" and r[2]]
    no_trav = [r for r in rows if r[1] == "OK" and not r[2]]
    no_main = [r for r in rows if r[1] == "NO_MAIN"]

    print(f"ROOT {ROOT}")
    print(f"SURFACES {len(rows)}  TRAVERSE_ARITY_GE3 {len(traverse)}  NO_TRAVERSE {len(no_trav)}  NO_MAIN {len(no_main)}")
    print()
    print("=== TRAVERSE (from main, transitive, arity>=3) ===")
    for r in sorted(traverse, key=lambda x: -x[5]):
        sname, _, hot, uses, ambig, max_a, _ = r
        bits = [f"{n}:{i['arity']}@{i['path']}" for n, i in sorted(hot, key=lambda x: -x[1]["arity"])]
        print(f"{sname:32} max={max_a:2}  n_hot={len(hot):2}  {', '.join(bits[:8])}" + (" …" if len(bits) > 8 else ""))
        if uses:
            print(f"{'':32}  unresolved_use={uses[:6]}")
        if ambig:
            print(f"{'':32}  ambiguous_calls={sorted(set(ambig))[:6]}")

    print()
    print("=== NO TRANSITIVE ARITY>=3 FROM MAIN ===")
    for r in no_trav:
        print(f"  {r[0]}")

    print()
    print("=== NO MAIN ===")
    for r in no_main:
        print(f"  {r[0]}")

    print()
    print("LIMITATIONS: line comments only; method recv counted in arity;")
    print("import resolve is path-heuristic under stdlib/; ambiguous names flagged.")


if __name__ == "__main__":
    main()
