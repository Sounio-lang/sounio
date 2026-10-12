#!/usr/bin/env python3
"""Heuristic sweep: whole-aggregate `a = b` / `*p = b` copies in .sio sources
followed by in-place mutation of either side (Madaros handle-alias hazard).
See docs/audit/MADAROS_STRUCT_REASSIGN_ARRAY_ALIAS_DISPATCH_2026-09-26.md.
Rows it flags are candidates for reading by hand, not verdicts.

Usage: python3 docs/audit/repro/aggregate_copy_alias_sweep.py <repo_root> [subdir ...]
       (default subdir: stdlib)
Prints TSV: file:line  form  lhs  rhs  type  in_loop  mutations
"""
import os, re, subprocess, sys

root = sys.argv[1]
subdirs = sys.argv[2:] or ["stdlib"]
files = subprocess.run(["git", "-C", root, "ls-files", "-z", "--"] + [f"{d}/*.sio" for d in subdirs],
                       capture_output=True, text=True).stdout.split("\0")
files = [f for f in files if f]

IDENT = r"[a-z_][a-z0-9_]*"
ASSIGN_RE = re.compile(r"^\s*(\*?)(" + IDENT + r")\s*=\s*(" + IDENT + r")\s*(//.*)?$")
STRUCT_RE = re.compile(r"^\s*(?:pub\s+)?(?:linear\s+|affine\s+)?struct\s+([A-Z]\w*)")
FN_RE = re.compile(r"^\s*(?:pub\s+)?(?:kernel\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\((.*)")
RET_RE = re.compile(r"\)\s*->\s*([A-Za-z_\[][^{]*?)\s*(?:with\b|where\b|\{|$)")
LOOP_RE = re.compile(r"^\s*(?:while\b|for\b|loop\b)")
NONAGG = {"true", "false", "None"}

def strip_comment(s):
    i = s.find("//")
    return s if i < 0 else s[:i]

structs = set()
fn_ret = {}
texts = {}
for f in files:
    try:
        lines = open(os.path.join(root, f), encoding="utf-8", errors="replace").read().split("\n")
    except OSError:
        continue
    texts[f] = lines
    for i, l in enumerate(lines):
        m = STRUCT_RE.match(l)
        if m:
            structs.add(m.group(1))
        m = FN_RE.match(l)
        if m:
            sig = " ".join(lines[i:i + 6])
            r = RET_RE.search(sig)
            if r:
                fn_ret.setdefault(m.group(1), r.group(1).strip())

def head_type(t):
    t = t.strip()
    t = re.sub(r"^&!?\s*", "", t)
    if t.startswith("["):
        return "[array]"
    m = re.match(r"([A-Za-z_]\w*)", t)
    return m.group(1) if m else None

def is_agg(t):
    return t == "[array]" or (t in structs)

def split_fns(lines):
    """Yield (start, end) line ranges for top-level and impl-nested fns via brace depth."""
    i = 0
    n = len(lines)
    while i < n:
        if FN_RE.match(lines[i]):
            depth = 0
            opened = False
            j = i
            while j < n:
                s = strip_comment(lines[j])
                s = re.sub(r'"(?:\\.|[^"\\])*"', '""', s)
                for ch in s:
                    if ch == "{":
                        depth += 1
                        opened = True
                    elif ch == "}":
                        depth -= 1
                if opened and depth <= 0:
                    break
                j += 1
            yield i, min(j, n - 1)
            i = j + 1
        else:
            i += 1

def name_type(lines, start, upto, name, depth=0):
    """Type of `name` at line `upto` within fn starting at `start`."""
    if depth > 4:
        return None
    t = None
    sig = " ".join(lines[start:start + 6])
    sig = sig[:sig.find("{")] if "{" in sig else sig
    m = re.search(r"[(,]\s*(?:mut\s+)?" + name + r"\s*:\s*([^,)]+)", sig)
    if m:
        t = head_type(m.group(1))
    decl = re.compile(r"\b(?:let|var)\s+" + name + r"\b\s*(:\s*([^=]+?))?\s*(=\s*(.*))?$")
    for k in range(start + 1, upto):
        s = strip_comment(lines[k]).strip()
        m = decl.search(s)
        if not m:
            continue
        if m.group(2):
            t = head_type(m.group(2))
            continue
        rhs = (m.group(4) or "").strip()
        if rhs.startswith("["):
            t = "[array]"
        elif re.match(r"[A-Z]\w*\s*\{", rhs):
            t = re.match(r"([A-Z]\w*)", rhs).group(1)
        elif re.match(r"[A-Z]\w*::\w+\s*\(", rhs):
            ty, fn = re.match(r"([A-Z]\w*)::(\w+)", rhs).groups()
            rt = fn_ret.get(fn)
            t = head_type(rt) if rt and head_type(rt) != "Self" else ty
        elif re.match(r"\w+\s*\(", rhs):
            fn = re.match(r"(\w+)", rhs).group(1)
            rt = fn_ret.get(fn)
            t = head_type(rt) if rt else None
        elif re.fullmatch(IDENT, rhs):
            t = name_type(lines, start, k, rhs, depth + 1)
        else:
            t = None
    return t

def mutations(lines, a, b, name, skip):
    out = []
    store = re.compile(r"(?<![\w.])" + name + r"((?:\.\w+|\[[^\]]*\])+)\s*([+\-*/]?=)(?!=)")
    refm = re.compile(r"&!\s*" + name + r"\b(?!\s*\.)")
    refm_field = re.compile(r"&!\s*" + name + r"(\.\w+)+")
    for k in range(a, b + 1):
        if k == skip:
            continue
        s = strip_comment(lines[k])
        if store.search(s):
            out.append(f"{name}@L{k+1}")
        elif refm.search(s) or refm_field.search(s):
            out.append(f"{name}@L{k+1}(&!)")
    return out

def block_end(lines, start):
    depth = 0
    opened = False
    for j in range(start, len(lines)):
        s = re.sub(r'"(?:\\.|[^"\\])*"', '""', strip_comment(lines[j]))
        for ch in s:
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
        if opened and depth <= 0:
            return j
    return len(lines) - 1

def enclosing_loop(lines, start, line):
    """Line index of the OUTERMOST loop header enclosing `line`, or -1."""
    depth = 0
    stack = []
    for k in range(start, line):
        s = strip_comment(lines[k])
        s = re.sub(r'"(?:\\.|[^"\\])*"', '""', s)
        is_loop = k if LOOP_RE.match(s) else -1
        for ch in s:
            if ch == "{":
                depth += 1
                stack.append(is_loop)
                is_loop = -1
            elif ch == "}":
                depth -= 1
                if stack:
                    stack.pop()
    heads = [h for h in stack if h >= 0]
    return heads[0] if heads else -1

rows = []
for f, lines in texts.items():
    for (a, b) in split_fns(lines):
        for k in range(a + 1, b + 1):
            m = ASSIGN_RE.match(lines[k])
            if not m:
                continue
            star, lhs, rhs = m.group(1), m.group(2), m.group(3)
            if rhs in NONAGG or lhs == rhs:
                continue
            rt = name_type(lines, a, k, rhs)
            lt = None if star else name_type(lines, a, k, lhs)
            t = rt if is_agg(rt) else (lt if is_agg(lt) else None)
            if t is None and star:
                t = rt
            if t is None or not is_agg(t):
                continue
            lh = enclosing_loop(lines, a, k)
            lo, hi = (lh, b) if lh >= 0 else (k, b)
            muts = mutations(lines, lo, hi, rhs, k) + ([] if star else mutations(lines, lo, hi, lhs, k))
            rows.append((f"{f}:{k+1}", "*p = x" if star else "a = b", lhs, rhs, t, f"loop@L{lh+1}" if lh >= 0 else "-", " ".join(muts) or "-"))

for r in sorted(rows):
    print("\t".join(r))
print(f"# candidates={len(rows)} with_mutation={sum(1 for r in rows if r[6] != '-')} structs_known={len(structs)}", file=sys.stderr)
