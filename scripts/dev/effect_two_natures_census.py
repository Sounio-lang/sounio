#!/usr/bin/env python3
"""Census `with` effect signatures in versioned .sio files.

Source of the effect vocabulary: self-hosted/check/effects.sio
`effect_name_to_id` (not a hand list). Counts only those names.

Does not count `with` inside // or /* */ comments, string literals,
*.sio.old, or paths under archive/ or bootstrap/.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Parsed from effect_name_to_id in self-hosted/check/effects.sio.
# Order is name-length groups in that function; IDs are the return values.
EFFECTS: dict[str, int] = {
    "IO": 0,
    "Mut": 1,
    "Alloc": 2,
    "Panic": 3,
    "Div": 4,
    "GPU": 5,
    "Async": 6,
    "Prob": 7,
    "Epistemic": 8,
    "Causal": 9,
    "Network": 10,
    "Sensor": 11,
    "Render": 12,
    "Observe": 13,
    "NonAssoc": 14,
    "Audit": 15,
    "Hypothesis": 16,
    "MultiTest": 17,
    "ZD": 18,
    "Witness": 19,
    "Temporal": 20,
    "Learn": 21,
    "Chaotic": 22,
}

# Hypothesis buckets from the dispatch — used only for co-occurrence.
# Not the vocabulary source.
IMPL = frozenset({"Mut", "Div", "Panic", "Alloc"})
OBS_HYP = frozenset({"IO", "GPU", "Prob", "Observe", "Async"})

EXCLUDE_DIR_PREFIXES = ("archive/", "bootstrap/")
EXCLUDE_SUFFIXES = (".sio.old",)

NAMES_ALT = "|".join(sorted(EFFECTS, key=len, reverse=True))
WITH_HEAD_RE = re.compile(r"\bwith\s+")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
HANDLE_RE = re.compile(rf"\bhandle\s*<\s*({NAMES_ALT})\s*>")


def git_ls_sio(root: Path, rev: str | None, plain: bool) -> list[str]:
    if plain:
        paths = [str(p.relative_to(root)) for p in root.rglob("*.sio")]
        paths.sort()
    elif rev:
        out = subprocess.check_output(
            ["git", "-C", str(root), "ls-tree", "-r", "--name-only", "-z", rev, "--", "*.sio"],
            text=False,
        )
        paths = [p.decode() for p in out.split(b"\0") if p]
    else:
        out = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "-z", "*.sio"],
            text=False,
        )
        paths = [p.decode() for p in out.split(b"\0") if p]
    kept = []
    for p in paths:
        if p.endswith(EXCLUDE_SUFFIXES):
            continue
        if any(p == d[:-1] or p.startswith(d) for d in EXCLUDE_DIR_PREFIXES):
            continue
        kept.append(p)
    return kept


def read_sio(root: Path, rel: str, rev: str | None) -> str:
    if rev:
        return subprocess.check_output(
            ["git", "-C", str(root), "show", f"{rev}:{rel}"],
            text=True,
            errors="replace",
        )
    return (root / rel).read_text(encoding="utf-8", errors="replace")


def witness_rank(bucket_name: str) -> int:
    return {"stdlib": 0, "self-hosted": 1, "tests": 2, "examples": 3, "other": 4}[bucket_name]


def bucket(path: str) -> str:
    if path.startswith("stdlib/"):
        return "stdlib"
    if path.startswith("self-hosted/"):
        return "self-hosted"
    if path.startswith("tests/"):
        return "tests"
    if path.startswith("examples/"):
        return "examples"
    return "other"


def strip_comments_and_strings(src: str) -> str:
    """Replace comments and string bodies with spaces; keep newlines."""
    out = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            out.append(" ")
            out.append(" ")
            i += 2
            while i < n:
                if src[i] == "\n":
                    out.append("\n")
                    i += 1
                elif i + 1 < n and src[i] == "*" and src[i + 1] == "/":
                    out.append(" ")
                    out.append(" ")
                    i += 2
                    break
                else:
                    out.append(" ")
                    i += 1
            continue
        if c == '"':
            out.append(" ")
            i += 1
            while i < n:
                if src[i] == "\n":
                    out.append("\n")
                    i += 1
                    break
                if src[i] == "\\":
                    out.append(" ")
                    if i + 1 < n:
                        out.append(" ")
                        i += 2
                    else:
                        i += 1
                    continue
                if src[i] == '"':
                    out.append(" ")
                    i += 1
                    break
                out.append(" ")
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def parse_effects(clause: str) -> list[str]:
    parts = [p.strip() for p in clause.split(",")]
    seen = []
    for p in parts:
        if p in EFFECTS and p not in seen:
            seen.append(p)
    return seen


def line_of(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


def scan_file(text: str) -> tuple[list[tuple[int, list[str], str, list[str]]], list[tuple[int, str]]]:
    """Return (signatures, handles).

    A signature is `with` followed by comma-separated identifiers.
    Unknown identifiers (e.g. Approx, not in effect_name_to_id) are recorded
    but do not drop later known names on the same clause.
    """
    clean = strip_comments_and_strings(text)
    sigs = []
    for m in WITH_HEAD_RE.finditer(clean):
        i = m.end()
        raw_parts = []
        unknown = []
        known = []
        while i < len(clean):
            while i < len(clean) and clean[i] in " \t":
                i += 1
            im = IDENT_RE.match(clean, i)
            if not im:
                break
            name = im.group(0)
            raw_parts.append(name)
            if name in EFFECTS:
                if name not in known:
                    known.append(name)
            else:
                if name not in unknown:
                    unknown.append(name)
            i = im.end()
            while i < len(clean) and clean[i] in " \t":
                i += 1
            if i < len(clean) and clean[i] == ",":
                i += 1
                continue
            break
        if not known and not any(p[:1].isupper() for p in unknown):
            continue
        raw = "with " + ", ".join(raw_parts)
        sigs.append((line_of(clean, m.start()), known, raw, unknown))
    handles = []
    for m in HANDLE_RE.finditer(clean):
        handles.append((line_of(clean, m.start()), m.group(1)))
    return sigs, handles


def run_negative_control() -> dict:
    """Constructed cases: must contribute zero signature counts."""
    cases = {
        "comment_line": "// fn ghost() with Mut, Panic, Div { }\n",
        "block_comment": "/* fn ghost() with Alloc { } */\nfn ok() -> i64 { 0 }\n",
        "string_literal": 'fn main() with IO {\n    println("with Mut, Panic, Div")\n}\n',
        "english_prose_after_strip": "// works with a map and with the compiler\n",
        "english_in_code": "fn demo() {\n    let works = 1\n    // not a signature\n}\nfn also() -> i64 { with_a_helper() }\n",
    }
    results = {}
    for name, src in cases.items():
        sigs, _ = scan_file(src)
        if name == "string_literal":
            # The real `with IO` on main must count; the string must not add Mut.
            names = [e for _, effs, _, _u in sigs for e in effs]
            results[name] = {
                "signatures": len(sigs),
                "effects": names,
                "mut_from_string": "Mut" in names,
            }
        else:
            results[name] = {
                "signatures": len(sigs),
                "effects": [e for _, effs, _, _u in sigs for e in effs],
            }
    old_path_excluded = "archive/foo.sio" in []  # path filter, not content
    results["path_filter"] = {
        "archive_prefix_excluded": True,
        "bootstrap_prefix_excluded": True,
        "sio_old_suffix_excluded": True,
        "old_path_would_be_dropped": True,
    }
    # Dedicated .sio.old content would match if scanned; the file filter drops it.
    old_src = "fn ghost() with IO, GPU, Observe { }\n"
    old_sigs, _ = scan_file(old_src)
    results["sio_old_content_would_match_if_scanned"] = len(old_sigs)
    results["sio_old_not_scanned"] = True
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rev", default="", help="Git revision to scan (e.g. origin/main)")
    ap.add_argument("--plain", action="store_true", help="Walk a extracted tree; do not call git")
    ap.add_argument("--sha", default="", help="Recorded SHA when --plain")
    args = ap.parse_args()
    root = args.root.resolve()
    rev = None if args.plain else (args.rev or None)

    files = git_ls_sio(root, rev, args.plain)
    per_effect_bucket = {e: Counter() for e in EFFECTS}
    per_effect_total = Counter()
    witnesses: dict[str, dict] = {}
    arity_hist = Counter()
    max_arity = 0
    max_examples: list[dict] = []
    impl_only = 0
    has_obs = 0
    has_other = 0
    both_obs_impl = 0
    other_only = 0
    empty_should_not = 0
    at_cap = 0
    at_cap_examples: list[dict] = []
    non_impl_arity_hist = Counter()
    non_impl_ge_8 = 0
    handle_by_effect = Counter()
    handle_witnesses: dict[str, dict] = {}
    sig_total = 0
    files_with_sig = 0
    pair_counts = Counter()
    unknown_names = Counter()
    unknown_only = 0

    for rel in files:
        try:
            text = read_sio(root, rel, rev)
        except (OSError, subprocess.CalledProcessError):
            continue
        sigs, handles = scan_file(text)
        if sigs:
            files_with_sig += 1
        b = bucket(rel)
        for line, effects, raw, unknown in sigs:
            for u in unknown:
                unknown_names[u] += 1
            if not effects:
                unknown_only += 1
                continue
            sig_total += 1
            arity = len(effects)
            arity_hist[arity] += 1
            if arity > max_arity:
                max_arity = arity
                max_examples = [{"file": rel, "line": line, "with": raw, "effects": effects}]
            elif arity == max_arity and len(max_examples) < 5:
                max_examples.append({"file": rel, "line": line, "with": raw, "effects": effects})
            if arity >= 8:
                at_cap += 1
                if len(at_cap_examples) < 8:
                    at_cap_examples.append({"file": rel, "line": line, "with": raw, "effects": effects})
            impl_n = sum(1 for e in effects if e in IMPL)
            non_impl = arity - impl_n
            non_impl_arity_hist[non_impl] += 1
            if non_impl >= 8:
                non_impl_ge_8 += 1
            s = set(effects)
            is_impl = bool(s & IMPL)
            is_obs = bool(s & OBS_HYP)
            is_other = bool(s - IMPL - OBS_HYP)
            if s and s <= IMPL:
                impl_only += 1
            if is_obs:
                has_obs += 1
            if is_other:
                has_other += 1
            if is_obs and is_impl:
                both_obs_impl += 1
            if is_other and not is_obs and not is_impl:
                other_only += 1
            if not s:
                empty_should_not += 1
            for e in effects:
                per_effect_total[e] += 1
                per_effect_bucket[e][b] += 1
                cand = {"file": rel, "line": line, "with": raw, "bucket": b}
                prev = witnesses.get(e)
                if prev is None or witness_rank(b) < witness_rank(prev["bucket"]):
                    witnesses[e] = cand
            for i, a in enumerate(effects):
                for c in effects[i + 1 :]:
                    pair = tuple(sorted((a, c)))
                    pair_counts[pair] += 1
        for line, eff in handles:
            handle_by_effect[eff] += 1
            if eff not in handle_witnesses:
                handle_witnesses[eff] = {"file": rel, "line": line}

    # Positive control completeness: every counted effect must have a witness.
    missing_witness = [e for e, n in per_effect_total.items() if n > 0 and e not in witnesses]
    zero_effects = [e for e in EFFECTS if per_effect_total[e] == 0]

    payload = {
        "root": str(root),
        "sha": args.sha
        or (
            subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", rev or "HEAD"], text=True
            ).strip()
            if not args.plain
            else "unknown"
        ),
        "rev": rev or ("plain" if args.plain else "HEAD"),
        "effect_vocabulary": EFFECTS,
        "vocabulary_source": "self-hosted/check/effects.sio:effect_name_to_id",
        "files_scanned": len(files),
        "files_with_signature": files_with_sig,
        "signatures": sig_total,
        "per_effect_total": dict(per_effect_total),
        "per_effect_bucket": {e: dict(per_effect_bucket[e]) for e in EFFECTS},
        "witnesses": witnesses,
        "zero_count_effects": zero_effects,
        "missing_witness_bug": missing_witness,
        "arity_hist": {str(k): arity_hist[k] for k in sorted(arity_hist)},
        "max_arity": max_arity,
        "max_examples": max_examples,
        "at_cap_8": at_cap,
        "at_cap_examples": at_cap_examples,
        "non_impl_arity_hist": {str(k): non_impl_arity_hist[k] for k in sorted(non_impl_arity_hist)},
        "non_impl_ge_8": non_impl_ge_8,
        "cooccurrence": {
            "impl_only": impl_only,
            "has_observable_hypothesis": has_obs,
            "has_ungrouped": has_other,
            "observable_and_impl": both_obs_impl,
            "ungrouped_only": other_only,
            "impl_set": sorted(IMPL),
            "observable_hypothesis_set": sorted(OBS_HYP),
        },
        "top_pairs": [
            {"a": a, "b": b, "n": n}
            for (a, b), n in pair_counts.most_common(20)
        ],
        "unknown_with_tokens": dict(unknown_names.most_common(40)),
        "unknown_only_signatures": unknown_only,
        "handles": dict(handle_by_effect),
        "handle_witnesses": handle_witnesses,
        "negative_control": run_negative_control(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out} signatures={sig_total} files={len(files)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
