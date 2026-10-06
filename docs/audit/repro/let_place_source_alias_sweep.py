#!/usr/bin/env python3
"""Sweep for `let/var v = <place>` aggregate snapshots that Madaros aliases.

Audit tooling for docs/audit/MADAROS_LET_FIELD_SOURCE_ALIAS_DISPATCH_2026-09-26.md.
Not part of any science path.

A place is `a.f(.g)*`, `(*p).f...`, `*p` or `xs[i]`. On Madaros (main
2e8b76d31) such a binding shares storage with the place whenever the place's
static type is an aggregate (struct or fixed array). The alias is observable
only if one side is written IN PLACE afterwards:

  A  the source place is written in place (`a.f.x = ..`, `a.f[i] = ..`,
     `&!a`, `&!a.f`) while the snapshot is still live: the snapshot changes.
  B  the new binding is written in place (`v.x = ..`, `v[i] = ..`, `&!v`):
     the owner changes. If the owner is a by-value parameter, the CALLER's
     value changes.

Rebinding the whole source (`a = f(a)`, `a.f = x`) is not a write into the
shared storage and is not flagged.

Heuristic and line-oriented. Types come from struct declarations across the
repository, parameter and `let` annotations, struct literals, declared return
types, and earlier field snapshots, resolved per file (see SymbolTables);
names with conflicting definitions are left unresolved, never guessed. Output
is deterministic and ends each run with reconciled coverage counts.
Multi-line statements and interprocedural mutation are missed. Every hit must
be read by hand.

Usage: let_place_source_alias_sweep.py <repo_root> <subdir> [<subdir> ...]
"""
import re
import subprocess
import sys
from collections import ChainMap

ID = r"[A-Za-z_][A-Za-z0-9_]*"
NON_COPYABLE = {"Box", "Knowledge", "Seq", "Option", "Vec", "String", "string"}


def ls_files(root, subdirs):
    out = subprocess.run(["git", "-C", root, "ls-files", "-z"] + [f"{d}/*.sio" for d in subdirs],
                         capture_output=True, check=True).stdout.decode()
    return [p for p in out.split("\0") if p]


def strip_comment(line):
    i = line.find("//")
    return line if i < 0 else line[:i]


def mask_comments(text):
    """Comments and literal contents replaced by spaces, newlines and offsets kept.

    `//` and `/* */` comments are blanked whole. String and char literals keep
    their delimiters but their contents are blanked, so `"http://..."` is not
    read as a comment and embedded source (e.g. WGSL `struct VertexIn {..}` in
    a string) never registers. Declarations and function spans are parsed from
    the masked text only.
    """
    def blank(a, b):
        for k in range(a, min(b, n)):
            if text[k] != "\n":
                out[k] = " "

    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            blank(i + 1, j)
            i = j + 1
        elif c == "'" and i + 2 < n and (text[i + 2] == "'" or (text[i + 1] == "\\" and i + 3 < n and text[i + 3] == "'")):
            w = 4 if text[i + 1] == "\\" else 3
            blank(i + 1, i + w - 1)
            i += w
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
        elif text.startswith("/*", i):
            while i < n and not text.startswith("*/", i):
                if text[i] != "\n":
                    out[i] = " "
                i += 1
            for k in range(i, min(i + 2, n)):
                out[k] = " "
            i += 2
        else:
            i += 1
    return "".join(out)


def split_top(s, sep=","):
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "[(<{":
            depth += 1
        elif ch in "])>}":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return parts


def match_brace(text, open_idx):
    depth = 0
    for i in range(open_idx, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
    return len(text) - 1


STRUCT_RE = re.compile(r"\b(?:pub\s+)?(?:linear\s+|affine\s+)?struct\s+(" + ID + r")\s*(?:<[^>{]*>)?\s*\{")
FN_RE = re.compile(r"\bfn\s+(" + ID + r")\s*(?:<[^>(]*>)?\s*\(")


def parse_structs_in(text):
    structs = {}
    for m in STRUCT_RE.finditer(text):
        name = m.group(1)
        ob = m.end() - 1
        body = text[ob + 1:match_brace(text, ob)]
        body = "\n".join(strip_comment(l) for l in body.split("\n"))
        fields = {}
        for part in split_top(body):
            part = part.strip()
            fm = re.match(r"(?:pub\s+)?(" + ID + r")\s*:\s*(.+)$", part, re.S)
            if fm:
                fields[fm.group(1)] = " ".join(fm.group(2).split())
        structs.setdefault(name, fields)
    return structs


def parse_fn_returns_in(text):
    rets = {}
    for m in FN_RE.finditer(text):
        # find matching ')' of params
        depth, i = 0, m.end() - 1
        while i < len(text):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        rest = text[i + 1:i + 300]
        rm = re.match(r"\s*->\s*([^{]+?)\s*(?:with\b[^{]*)?\{", rest, re.S)
        if rm:
            rets.setdefault(m.group(1), " ".join(rm.group(1).split()))
    return rets


def base_type(t):
    t = t.strip()
    if t.startswith("&!"):
        return t[2:].strip()
    if t.startswith("&"):
        return t[1:].strip()
    return t


def aggregate_kind(t, structs):
    """'array', 'struct' or None for a (dereferenced) type string."""
    if not t:
        return None
    t = base_type(t)
    if re.match(r"^\[.*;\s*[^\]]+\]$", t):
        return "array"
    head = re.match(r"^(" + ID + r")", t)
    if head and head.group(1) in structs and head.group(1) not in NON_COPYABLE:
        return "struct"
    return None


def field_type(struct_t, field, structs):
    t = base_type(struct_t or "")
    head = re.match(r"^(" + ID + r")", t)
    if not head or head.group(1) not in structs:
        return None
    return structs[head.group(1)].get(field)


def resolve_place(expr, env, structs):
    """Type of a place expression, or None. Returns (type, root_ident)."""
    e = expr.strip()
    m = re.match(r"^\(\s*\*\s*(" + ID + r")\s*\)((?:\." + ID + r")*)$", e)
    if m:
        t, root, chain = env.get(m.group(1)), m.group(1), m.group(2)
    else:
        m = re.match(r"^\*\s*(" + ID + r")$", e)
        if m:
            return env.get(m.group(1)), m.group(1)
        m = re.match(r"^(" + ID + r")\s*\[[^\]]*\]((?:\." + ID + r")*)$", e)
        if m:
            at = base_type(env.get(m.group(1)) or "")
            am = re.match(r"^\[\s*(.+?)\s*;.*\]$", at)
            t = am.group(1) if am else None
            root, chain = m.group(1), m.group(2)
        else:
            m = re.match(r"^(" + ID + r")((?:\." + ID + r")+)$", e)
            if not m:
                return None, None
            t, root, chain = env.get(m.group(1)), m.group(1), m.group(2)
    for f in [x for x in chain.split(".") if x]:
        if t is None:
            return None, root
        t = field_type(t, f, structs)
    return t, root


LET_RE = re.compile(r"^\s*(let|var)\s+(" + ID + r")\s*(?::\s*([^=]+?))?\s*=\s*(.+?)\s*;?\s*$")
PARAM_RE = re.compile(r"(" + ID + r")\s*:\s*([^,]+)")


def fn_spans(text):
    for m in FN_RE.finditer(text):
        depth, i = 0, m.end() - 1
        while i < len(text):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        params = text[m.end():i]
        ob = text.find("{", i)
        semi = text.find(";", i)
        if ob < 0 or (0 <= semi < ob and "\n" not in text[i:semi]):
            continue
        cb = match_brace(text, ob)
        yield m.group(1), params, ob, cb


def written_in_place(line, target):
    """True if `line` writes into the storage named by `target` (a place text)."""
    t = re.escape(target)
    l = strip_comment(line)
    if re.search(r"&!\s*\(?\s*" + t + r"\b(?!\s*\.\s*" + ID + r"\s*\()", l):
        return True
    # target.x... = / target[..]... = / compound forms, not ==, not <=, >=, !=
    if re.search(r"(?<![A-Za-z0-9_.])" + t + r"\s*(?:\.\s*" + ID + r"|\[[^\]]*\])+(?:\s*(?:\.\s*" + ID + r"|\[[^\]]*\]))*\s*(?:[-+*/%]?=)(?!=)", l):
        return True
    return False


USE_RE = re.compile(r"^\s*(?:pub\s+)?use\s+([A-Za-z_][\w:]*?)(?:::\{([^}]*)\}|::(" + ID + r"))\s*;?\s*$", re.M)


class SymbolTables:
    """Struct and fn-return tables resolved PER FILE, deterministically.

    Short names collide across the corpus (several `State` structs, dozens of
    `step` fns), so a single global first-wins table would depend on file
    order. Resolution order for a name used in file F:
      1. a definition in F itself;
      2. a definition in the module F imports that name from
         (`use a::b::{X}` -> stdlib/a/b.sio or stdlib/a/b/mod.sio);
      3. the corpus-wide definition, only if every definition of the name
         agrees; a name with conflicting definitions is left unresolved.
    """

    def __init__(self, texts):
        self.local_structs, self.local_rets = {}, {}
        all_structs, all_rets = {}, {}
        for path in sorted(texts):
            ls = parse_structs_in(texts[path])
            lr = parse_fn_returns_in(texts[path])
            self.local_structs[path], self.local_rets[path] = ls, lr
            for k, v in ls.items():
                all_structs.setdefault(k, []).append(v)
            for k, v in lr.items():
                all_rets.setdefault(k, []).append(v)
        self.global_structs = {k: v[0] for k, v in all_structs.items() if all(x == v[0] for x in v)}
        self.global_rets = {k: v[0] for k, v in all_rets.items() if all(x == v[0] for x in v)}
        self.ambiguous_structs = sorted(k for k in all_structs if k not in self.global_structs)
        self.texts = texts

    def _module_file(self, mod):
        base = "stdlib/" + mod.replace("::", "/")
        for cand in (base + ".sio", base + "/mod.sio"):
            if cand in self.texts:
                return cand
        return None

    def for_file(self, path):
        imp_s, imp_r = {}, {}
        for m in USE_RE.finditer(self.texts.get(path, "")):
            names = [n.strip().split(" as ")[0].strip() for n in (m.group(2) or m.group(3) or "").split(",")]
            mf = self._module_file(m.group(1))
            if not mf:
                continue
            for n in names:
                if n in self.local_structs.get(mf, {}):
                    imp_s[n] = self.local_structs[mf][n]
                if n in self.local_rets.get(mf, {}):
                    imp_r[n] = self.local_rets[mf][n]
        structs = ChainMap(self.local_structs.get(path, {}), imp_s, self.global_structs)
        rets = ChainMap(self.local_rets.get(path, {}), imp_r, self.global_rets)
        return structs, rets


def sweep(root, subdirs):
    files = ls_files(root, subdirs)
    all_files = ls_files(root, ["stdlib", "self-hosted", "examples", "tests"])
    texts = {}
    for p in sorted(set(all_files) | set(files)):
        try:
            with open(f"{root}/{p}", encoding="utf-8", errors="replace") as fh:
                texts[p] = mask_comments(fh.read())
        except OSError:
            # Best effort: a file listed by git ls-files but unreadable (e.g. a
            # dangling symlink) is skipped; the sweep reports on what it read.
            pass
    tables = SymbolTables(texts)
    rows = []
    # Every place binding the sweep sees, by how far its type resolved.
    counts = {"place_bindings": 0, "aggregate": 0, "scalar": 0,
              "unresolved_root": 0, "unresolved_chain": 0}
    for p in sorted(files):
        text = texts.get(p, "")
        structs, rets = tables.for_file(p)
        line_starts = [0]
        for i, c in enumerate(text):
            if c == "\n":
                line_starts.append(i + 1)

        def line_no(off):
            lo, hi = 0, len(line_starts) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if line_starts[mid] <= off:
                    lo = mid
                else:
                    hi = mid - 1
            return lo + 1

        lines = text.split("\n")
        for fname, params, ob, cb in fn_spans(text):
            env, byval = {}, set()
            for pm in PARAM_RE.finditer(params):
                pt = pm.group(2).strip()
                env[pm.group(1)] = pt
                if not pt.startswith("&"):
                    byval.add(pm.group(1))
            first, last = line_no(ob), line_no(cb)
            loop_heads = []  # (line, end_line)
            for ln in range(first, last + 1):
                raw = strip_comment(lines[ln - 1])
                if re.match(r"^\s*(while|for|loop)\b", raw):
                    # approximate loop end by brace matching from this line
                    off = line_starts[ln - 1] + raw.find("{") if "{" in raw else -1
                    if off >= 0:
                        loop_heads.append((ln, line_no(match_brace(text, off))))
                m = LET_RE.match(raw)
                if not m:
                    continue
                kind, name, ann, rhs = m.groups()
                rhs = rhs.strip()
                if ann:
                    env[name] = ann.strip()
                    typed = ann.strip()
                else:
                    typed = None
                    sm = re.match(r"^(" + ID + r")\s*\{", rhs)
                    cm = re.match(r"^(" + ID + r")\s*\(", rhs)
                    im = re.match(r"^(" + ID + r")$", rhs)
                    if sm and sm.group(1) in structs:
                        typed = sm.group(1)
                    elif cm and cm.group(1) in rets:
                        typed = rets[cm.group(1)]
                    elif im and im.group(1) in env:
                        typed = env[im.group(1)]
                place_t, root_id = resolve_place(rhs, env, structs)
                if place_t is not None:
                    typed = typed or place_t
                if typed:
                    env[name] = typed
                if root_id is None:
                    continue  # not a place
                if re.match(r"^" + ID + r"$", rhs):
                    continue  # bare identifier: already copied by Madaros
                counts["place_bindings"] += 1
                if place_t is None:
                    counts["unresolved_root" if root_id not in env else "unresolved_chain"] += 1
                    continue
                agg = aggregate_kind(place_t, structs)
                if agg is None:
                    counts["scalar"] += 1
                    continue
                counts["aggregate"] += 1
                # live range: rest of fn; widen to enclosing outermost loop head
                start = ln + 1
                in_loop = False
                for lh, le in loop_heads:
                    if lh < ln <= le:
                        start = min(start, lh + 1)
                        in_loop = True
                src_text = re.sub(r"\s+", "", rhs)
                if src_text.startswith("*") and not src_text.startswith("*("):
                    src_targets = [src_text[1:]]
                    src_text_w = "(*" + src_text[1:] + ")"
                    src_targets.append(src_text_w)
                else:
                    src_targets = [src_text]
                # a whole-root mutable borrow also writes the place
                root_borrow = re.compile(r"&!\s*" + re.escape(root_id) + r"\b(?!\s*\.)")
                a_hits, b_hits = [], []
                for k in range(start, last + 1):
                    if k == ln:
                        continue
                    l = lines[k - 1]
                    if any(written_in_place(l, t) for t in src_targets) or root_borrow.search(strip_comment(l)):
                        a_hits.append(k)
                    if k > ln and written_in_place(l, name):
                        b_hits.append(k)
                # A-class needs the snapshot read after the write
                a_live = []
                for k in a_hits:
                    tail = "\n".join(strip_comment(x) for x in lines[k:last])
                    if re.search(r"(?<![A-Za-z0-9_.])" + re.escape(name) + r"\b", tail) or k < ln:
                        a_live.append(k)
                cls = []
                if a_live:
                    cls.append("A")
                if b_hits:
                    cls.append("B-param" if root_id in byval else "B")
                rows.append({
                    "file": p, "line": ln, "fn": fname, "binding": f"{kind} {name} = {rhs}",
                    "type": place_t, "agg": agg, "class": ",".join(cls) or "-",
                    "a": a_live[:4], "b": b_hits[:4], "root_byval": root_id in byval,
                    "in_loop": in_loop,
                })
    return rows, counts, tables.ambiguous_structs


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    rows, counts, ambiguous = sweep(sys.argv[1], sys.argv[2:])
    flagged = [r for r in rows if r["class"] != "-"]
    c = counts
    assert c["place_bindings"] == c["aggregate"] + c["scalar"] + c["unresolved_root"] + c["unresolved_chain"]
    assert c["aggregate"] == len(rows)
    print(f"place bindings: {c['place_bindings']}  = aggregate {c['aggregate']}"
          f" + scalar {c['scalar']} + unresolved root {c['unresolved_root']}"
          f" + unresolved field chain {c['unresolved_chain']}")
    print(f"struct names with conflicting definitions (left unresolved): {len(ambiguous)}")
    print(f"aggregate place bindings: {len(rows)}  flagged: {len(flagged)}")
    by = {}
    for r in rows:
        by[r["class"]] = by.get(r["class"], 0) + 1
    for k in sorted(by):
        print(f"  class {k}: {by[k]}")
    for r in flagged:
        print(f"{r['class']:8} {r['file']}:{r['line']} fn {r['fn']}: {r['binding']}  [{r['type']}]"
              f"  A@{r['a']} B@{r['b']}")
    # Handle-cost exposure once these bindings copy: one handle per executed
    # binding of an aggregate > 16 B, never reclaimed (exit 182 when full).
    looped = [r for r in rows if r["in_loop"]]
    print(f"inside a loop (a copy would cost one handle per iteration): {len(looped)}")
    for r in looped:
        print(f"  LOOP {r['class']:8} {r['file']}:{r['line']} fn {r['fn']}: {r['binding']}  [{r['type']}]")


if __name__ == "__main__":
    main()
