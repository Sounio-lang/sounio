#!/usr/bin/env python3
"""Static census + caller closure for `with Mod` → `with Mut` E035 radius.

Measurement only. Does not write the substitution into any git tree.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import subprocess
import sys

SKIP_PREFIXES = (
    "archive/",
    "bootstrap/",
    "self-hosted/bootstrap/",
)
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
WITH_RE = re.compile(r"\bwith\b")
FN_RE = re.compile(r"\b(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)")
CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def skip_path(rel: str) -> bool:
    if rel.endswith(".sio.old"):
        return True
    return any(rel == p[:-1] or rel.startswith(p) for p in SKIP_PREFIXES)


def mask_comments_and_strings(text: str) -> str:
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            out.extend("  ")
            i += 2
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            out.extend("  ")
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
            if i + 1 < n:
                out.extend("  ")
                i += 2
            continue
        if ch == '"':
            out.append(" ")
            i += 1
            while i < n:
                if text[i] == "\\" and i + 1 < n:
                    out.extend("  ")
                    i += 2
                    continue
                if text[i] == '"':
                    out.append(" ")
                    i += 1
                    break
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def skip_ws(s: str, pos: int) -> int:
    while pos < len(s) and s[pos] in " \t\n":
        pos += 1
    return pos


def skip_payload(s: str, pos: int) -> int:
    if pos >= len(s) or s[pos] != "(":
        return pos
    depth = 0
    while pos < len(s):
        if s[pos] == "(":
            depth += 1
        elif s[pos] == ")":
            depth -= 1
            pos += 1
            if depth == 0:
                return pos
            continue
        pos += 1
    return pos


def extract_with_names(text: str) -> list[tuple[int, str, int]]:
    """Return (line, name, start_index) for each with-clause identifier."""
    masked = mask_comments_and_strings(text)
    hits = []
    for m in WITH_RE.finditer(masked):
        pos = m.end()
        while True:
            pos = skip_ws(masked, pos)
            im = IDENT_RE.match(masked, pos)
            if not im:
                break
            name = im.group(0)
            after = skip_ws(masked, skip_payload(masked, im.end()))
            if after < len(masked) and masked[after] == ",":
                nxt = skip_ws(masked, after + 1)
                im2 = IDENT_RE.match(masked, nxt)
                if im2:
                    after2 = skip_ws(masked, skip_payload(masked, im2.end()))
                    if after2 < len(masked) and masked[after2] == ":":
                        hits.append(
                            (masked.count("\n", 0, im.start()) + 1, name, im.start())
                        )
                        break
            hits.append((masked.count("\n", 0, im.start()) + 1, name, im.start()))
            pos = after
            if pos < len(masked) and masked[pos] == ",":
                pos += 1
                continue
            break
    return hits


def match_parens(s: str, open_pos: int) -> int | None:
    if open_pos >= len(s) or s[open_pos] != "(":
        return None
    depth = 0
    i = open_pos
    while i < len(s):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def skip_type(s: str, pos: int) -> int:
    """Skip a type after `->`. Conservative: stop at `with` / `{` / `;`."""
    pos = skip_ws(s, pos)
    depth_paren = 0
    depth_brack = 0
    depth_angle = 0
    while pos < len(s):
        ch = s[pos]
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            if depth_paren == 0:
                break
            depth_paren -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]":
            depth_brack -= 1
        elif ch == "<":
            depth_angle += 1
        elif ch == ">":
            depth_angle -= 1
        elif depth_paren == 0 and depth_brack == 0 and depth_angle == 0:
            if s.startswith("with", pos) and (pos + 4 == len(s) or not (s[pos + 4].isalnum() or s[pos + 4] == "_")):
                break
            if ch in "{;":
                break
        pos += 1
    return pos


def parse_functions(text: str) -> list[dict]:
    """Function-owned with-clauses (not parameter fn-types)."""
    masked = mask_comments_and_strings(text)
    fns = []
    for m in FN_RE.finditer(masked):
        name = m.group(1)
        pos = skip_ws(masked, m.end())
        if pos >= len(masked) or masked[pos] != "(":
            continue
        close = match_parens(masked, pos)
        if close is None:
            continue
        after = skip_ws(masked, close + 1)
        if after < len(masked) - 1 and masked.startswith("->", after):
            after = skip_type(masked, after + 2)
            after = skip_ws(masked, after)
        effects = []
        w = WITH_RE.match(masked, after) if after < len(masked) else None
        if w:
            p = w.end()
            while True:
                p = skip_ws(masked, p)
                im = IDENT_RE.match(masked, p)
                if not im:
                    break
                effects.append(im.group(0))
                p = skip_ws(masked, skip_payload(masked, im.end()))
                if p < len(masked) and masked[p] == ",":
                    nxt = skip_ws(masked, p + 1)
                    im2 = IDENT_RE.match(masked, nxt)
                    if im2:
                        after2 = skip_ws(masked, skip_payload(masked, im2.end()))
                        if after2 < len(masked) and masked[after2] == ":":
                            break
                    p += 1
                    continue
                break
        line = masked.count("\n", 0, m.start()) + 1
        # body start
        brace = masked.find("{", after if after else close)
        body_end = None
        if brace >= 0:
            depth = 0
            i = brace
            while i < len(masked):
                if masked[i] == "{":
                    depth += 1
                elif masked[i] == "}":
                    depth -= 1
                    if depth == 0:
                        body_end = i
                        break
                i += 1
        fns.append(
            {
                "name": name,
                "line": line,
                "effects": effects,
                "start": m.start(),
                "body_start": brace if brace >= 0 else None,
                "body_end": body_end,
            }
        )
    return fns


def versioned_sio(root: pathlib.Path) -> list[str]:
    raw = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z", "*.sio"], text=False
    )
    return [p.decode() for p in raw.split(b"\0") if p]


def grep_mod_sites(root: pathlib.Path) -> tuple[int, int, list[str]]:
    """#2009 instrument: git grep -lE plus per-file occurrence count."""
    raw = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "grep",
            "-nE",
            r"\bwith[[:space:]]+Mod\b",
            "--",
            "*.sio",
            ":!archive/*",
            ":!bootstrap/*",
        ],
        text=True,
    )
    files = []
    n = 0
    for line in raw.splitlines():
        if not line:
            continue
        n += 1
        path = line.split(":", 1)[0]
        if path not in files:
            files.append(path)
    return n, len(files), files


def dir_bucket(rel: str) -> str:
    parts = rel.split("/")
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0]


def analyse(root: pathlib.Path) -> dict:
    rels = [r for r in versioned_sio(root) if not skip_path(r)]
    token_mod = []  # parser-faithful every with-Mod token
    fns_by_file: dict[str, list[dict]] = {}
    for rel in rels:
        path = root / rel
        text = path.read_text(encoding="utf-8", errors="replace")
        for line, name, _idx in extract_with_names(text):
            if name == "Mod":
                token_mod.append({"path": rel, "line": line})
        fns = parse_functions(text)
        for fn in fns:
            fn["path"] = rel
        fns_by_file[rel] = fns

    grep_sites, grep_files, grep_file_list = grep_mod_sites(root)

    affected_fns = []
    for rel, fns in fns_by_file.items():
        for fn in fns:
            if "Mod" in fn["effects"]:
                affected_fns.append(fn)

    # file coexistence
    files_mod = set()
    files_mut = set()
    for rel, fns in fns_by_file.items():
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        names = [n for _l, n, _i in extract_with_names(text)]
        if "Mod" in names:
            files_mod.add(rel)
        if "Mut" in names:
            files_mut.add(rel)

    import bisect

    # call graph: file-local first, then unique name.
    by_name: dict[str, list[dict]] = collections.defaultdict(list)
    for rel, fns in fns_by_file.items():
        for fn in fns:
            by_name[fn["name"]].append(fn)

    def key(fn: dict) -> str:
        return f"{fn['path']}:{fn['name']}:{fn['line']}"

    affected_keys = {key(fn) for fn in affected_fns}
    affected_names = {fn["name"] for fn in affected_fns}
    has_mut = {}
    has_mod = {}
    for rel, fns in fns_by_file.items():
        for fn in fns:
            has_mut[key(fn)] = "Mut" in fn["effects"]
            has_mod[key(fn)] = "Mod" in fn["effects"]

    def would_have_mut(k: str) -> bool:
        return has_mut.get(k, False) or has_mod.get(k, False)

    local_by_name: dict[str, dict[str, list[dict]]] = {}
    body_index: dict[str, tuple[list[int], list[dict]]] = {}
    for rel, fns in fns_by_file.items():
        lb: dict[str, list[dict]] = collections.defaultdict(list)
        starts = []
        owners = []
        for fn in fns:
            lb[fn["name"]].append(fn)
            if fn["body_start"] is not None and fn["body_end"] is not None:
                starts.append(fn["body_start"])
                owners.append(fn)
        local_by_name[rel] = lb
        # sort by body_start for bisect
        order = sorted(range(len(starts)), key=lambda i: starts[i])
        body_index[rel] = ([starts[i] for i in order], [owners[i] for i in order])

    def owning_fn(rel: str, pos: int) -> dict | None:
        starts, owners = body_index.get(rel, ([], []))
        i = bisect.bisect_right(starts, pos) - 1
        if i < 0:
            return None
        fn = owners[i]
        if fn["body_start"] < pos < fn["body_end"]:
            return fn
        return None

    def resolve(rel: str, name: str) -> dict | None:
        local = local_by_name.get(rel, {}).get(name, [])
        if local:
            return local[0]
        cands = by_name.get(name, [])
        if len(cands) == 1:
            return cands[0]
        return None

    callers_of: dict[str, set[str]] = collections.defaultdict(set)
    unresolved_calls = 0
    resolved_calls = 0
    # Only resolve calls whose name is an affected function. Transitive
    # callers are themselves found when we walk callers_of of those keys,
    # then of *their* callers — so we must also record calls TO any
    # function that we later put in the closure. That requires all calls.
    # Compromise: record every call whose callee resolves, but skip
    # keyword/builtin names that never match a defined fn (resolve=None).
    for rel, fns in fns_by_file.items():
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        masked = mask_comments_and_strings(text)
        for cm in CALL_RE.finditer(masked):
            name = cm.group(1)
            if name not in by_name:
                continue
            owner = owning_fn(rel, cm.start())
            if owner is None:
                continue
            callee = resolve(rel, name)
            if callee is None:
                unresolved_calls += 1
                continue
            resolved_calls += 1
            callers_of[key(callee)].add(key(owner))

    # Transitive caller closure of affected functions
    # Wave 0 = affected fns themselves
    # Wave d+1 = callers of wave d that are not already in the closure
    closure = set(affected_keys)
    wave = list(affected_keys)
    depth_of = {k: 0 for k in affected_keys}
    max_depth = 0
    while wave:
        nxt = []
        for k in wave:
            for c in callers_of.get(k, ()):
                if c not in closure:
                    closure.add(c)
                    depth_of[c] = depth_of[k] + 1
                    max_depth = max(max_depth, depth_of[c])
                    nxt.append(c)
        wave = nxt

    # Who would fail E035 at wave 1: callers of an affected fn that would
    # NOT have Mut after the substitution.
    wave1_fail = set()
    wave1_ok_already_mut = set()
    wave1_ok_because_mod = set()
    for ak in affected_keys:
        for c in callers_of.get(ak, ()):
            if would_have_mut(c):
                if has_mut.get(c) and not has_mod.get(c):
                    wave1_ok_already_mut.add(c)
                else:
                    wave1_ok_because_mod.add(c)
            else:
                wave1_fail.add(c)

    # In the FULL closure, after iteratively adding Mut to every failing
    # caller, who still needs a new Mut annotation?
    # = members of closure that do not already have Mut and do not have Mod
    need_mut_annotation = {k for k in closure if not would_have_mut(k)}
    already_mut_in_closure = {k for k in closure if has_mut.get(k) and not has_mod.get(k)}
    get_mut_from_mod = {k for k in closure if has_mod.get(k)}

    def files_of(keys):
        return sorted({k.split(":")[0] for k in keys})

    closure_files = files_of(closure)
    need_files = files_of(need_mut_annotation)
    affected_files = files_of(affected_keys)

    # Does the closure reach the compiler or the whole stdlib?
    reaches_compiler = any(f.startswith("self-hosted/") for f in closure_files)
    stdlib_files = [f for f in closure_files if f.startswith("stdlib/")]
    test_files = [f for f in closure_files if f.startswith("tests/")]

    # buckets of affected files
    buckets = collections.Counter(dir_bucket(f) for f in affected_files)

    # grep-only comment sites: grep hits whose line is a comment
    grep_comment_only = 0
    grep_not_in_extractor = []
    extractor_set = {(h["path"], h["line"]) for h in token_mod}
    # rebuild grep line list
    raw = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "grep",
            "-nE",
            r"\bwith[[:space:]]+Mod\b",
            "--",
            "*.sio",
            ":!archive/*",
            ":!bootstrap/*",
        ],
        text=True,
    )
    grep_pairs = []
    for line in raw.splitlines():
        path, rest = line.split(":", 1)
        ln_s, _src = rest.split(":", 1)
        ln = int(ln_s)
        grep_pairs.append((path, ln))
        if (path, ln) not in extractor_set:
            grep_not_in_extractor.append({"path": path, "line": ln, "text": _src.strip()[:120]})

    extractor_not_in_grep = [
        h for h in token_mod if (h["path"], h["line"]) not in {(p, l) for p, l in grep_pairs}
    ]

    return {
        "sha": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "instruments": {
            "git_grep_with_Mod": {
                "sites": grep_sites,
                "files": grep_files,
                "note": "counts comments and strings; #2009 instrument",
            },
            "parser_faithful_tokens": {
                "sites": len(token_mod),
                "files": len({h["path"] for h in token_mod}),
                "note": "closed-list gate extractor; excludes comments/strings",
            },
            "function_owned_signatures": {
                "functions": len(affected_fns),
                "files": len(affected_files),
                "note": "fn's own with-clause, not parameter fn-types",
            },
            "disagreement": {
                "grep_not_in_extractor": grep_not_in_extractor[:40],
                "grep_not_in_extractor_count": len(grep_not_in_extractor),
                "extractor_not_in_grep_count": len(extractor_not_in_grep),
                "extractor_not_in_grep_sample": extractor_not_in_grep[:20],
                "closed_list_stdout_cap_explain": (
                    "PR #2004 printed omitted=2793 (= 2813-20 stdout cap). "
                    "That is not a second count."
                ),
            },
        },
        "coexistence": {
            "files_with_mod": len(files_mod),
            "files_with_mod_and_mut": len(files_mod & files_mut),
            "files_with_mod_never_mut": len(files_mod - files_mut),
        },
        "buckets": dict(buckets),
        "call_graph": {
            "resolved_calls": resolved_calls,
            "unresolved_calls": unresolved_calls,
            "affected_functions": len(affected_keys),
            "affected_files": len(affected_files),
            "closure_functions": len(closure),
            "closure_files": len(closure_files),
            "max_depth": max_depth,
            "wave1_e035_predicted": len(wave1_fail),
            "wave1_ok_already_mut": len(wave1_ok_already_mut),
            "wave1_ok_because_also_mod": len(wave1_ok_because_mod),
            "closure_need_new_mut": len(need_mut_annotation),
            "closure_already_mut": len(already_mut_in_closure),
            "closure_get_mut_from_mod": len(get_mut_from_mod),
            "reaches_compiler": reaches_compiler,
            "stdlib_files_in_closure": len(stdlib_files),
            "test_files_in_closure": len(test_files),
        },
        "wave1_fail_sample": sorted(wave1_fail)[:40],
        "need_mut_sample": sorted(need_mut_annotation)[:40],
        "closure_files": closure_files,
        "need_mut_files": need_files,
        "affected_fn_sample": [
            {"path": fn["path"], "line": fn["line"], "name": fn["name"], "effects": fn["effects"]}
            for fn in affected_fns[:15]
        ],
    }


def apply_mod_to_mut(root: pathlib.Path) -> int:
    """Replace with-clause Mod tokens with Mut. Refuses a git checkout."""
    if (root / ".git").exists():
        raise SystemExit(
            "refuse: will not apply Mod→Mut inside a git checkout; "
            "pass a disposable scratch copy"
        )
    n = 0
    for p in root.rglob("*.sio"):
        rel = p.relative_to(root).as_posix()
        if skip_path(rel):
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        hits = [(idx, name) for _ln, name, idx in extract_with_names(text) if name == "Mod"]
        if not hits:
            continue
        chars = list(text)
        for idx, _name in sorted(hits, reverse=True):
            # 'Mod' is 3 chars; masked indices align with original.
            if text[idx : idx + 3] != "Mod":
                raise SystemExit(f"index mismatch {rel} @{idx}")
            chars[idx : idx + 3] = list("Mut")
            n += 1
        p.write_text("".join(chars), encoding="utf-8")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--json-out", default="")
    ap.add_argument(
        "--apply-scratch",
        default="",
        help="disposable tree only; refuses if ROOT/.git exists",
    )
    args = ap.parse_args()
    if args.apply_scratch:
        n = apply_mod_to_mut(pathlib.Path(args.apply_scratch).resolve())
        print(f"applied_mod_to_mut sites={n} root={args.apply_scratch}")
        return 0
    root = pathlib.Path(args.root).resolve()
    result = analyse(root)
    text = json.dumps(result, indent=2)
    if args.json_out:
        pathlib.Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
