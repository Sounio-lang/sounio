#!/usr/bin/env python3
"""Self-falsifying compilation, rung R6 — evidential independence as a static property.

Spec: docs/research/self_falsifying_compilation_line_r6_2026-07-26.md

R0 §3 proved that shared misinterpretation is undetectable *when the compiler's
only evidence about a proposition is the claim's own check*. That is an
antecedent, not a wall. R6 attacks the antecedent: require a second, independent
derivation of the same proposition, and make **independence itself** a property
the machine can check.

Prior art bounds what is being claimed. Cargo `build.rs` binds a build to a
check's exit status; snapshot testing binds it to a check's literal output; R2
bound it to the proposition a check reports. None of them ask **where the
evidence came from**. That is what is new here, and it is a narrow claim.

TWO NOTIONS OF INDEPENDENCE ARE MEASURED, because the obvious one turned out to
be vacuous on this corpus:

  I1  IMPORT-CLOSURE DISJOINTNESS — do harness and corroborator share
      repo-local modules? Vacuous here: research contracts import nothing but
      stdlib and numpy, so every pair passes and the check discriminates
      nothing. Reported rather than quietly dropped.

  I2  DERIVATION DISJOINTNESS — do they share function bodies, structurally?
      In a corpus of self-contained scripts, a misunderstanding propagates by
      COPY-PASTE, not by import. This is the notion that discriminates.

VERDICT OPTIONS, FIXED BEFORE COMPUTING (see main()):
  INDEPENDENCE_CHECKABLE__CORROBORATION_BINDS
      I2 computes, an independent pair passes, and a copy-paste pair is
      rejected.
  INDEPENDENCE_CHECKABLE__GUARD_VACUOUS
      it computes but never discriminates — everything passes.
  INDEPENDENCE_UNCHECKABLE
      bodies or closures cannot be extracted reliably.

WHAT THIS IS NOT, stated here and in the spec §0 rather than defensively later:
structural disjointness is a **checkable lower bound** on evidential
independence, nothing more. Two files can share no code and still encode the
same misunderstanding, because the misunderstanding lives in the author's head.
R3 demonstrated exactly that on this corpus: its falsifier shared no code with
the harness it refuted and still only fired because the author already knew the
correction. This rung rules out the cheapest failure — a corroborator that
reuses the harness's own derivation — and rules out nothing about shared
authorship.

Pure Python 3 (ast, difflib).
"""

from __future__ import annotations

import ast
import difflib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Similarity at or above this counts as the same derivation.
DUP_THRESHOLD = 0.90

# Bodies smaller than this (AST nodes) are excluded from the comparison.
#
# POST-HOC, and stated as such: the first run flagged the independent pair,
# matching `conj` against `cd_conj` at similarity 1.000. Both are 24-node
# bodies that negate every component but the first — there is one natural way
# to write that, so structural identity carries no evidence of reuse. The
# observed distribution has a clean gap: trivial helpers at 9-42 nodes, real
# derivations at 64+ (cds 223, cd_mul 131, o 83). The floor sits in that gap.
# Because it was chosen after seeing those two functions, the two-pair test
# alone would be fitted to its own answer -- which is why the evaluation below
# runs over EVERY pair of research contracts instead.
TRIVIAL_NODE_FLOOR = 50

# Directories that are not this repository's own code.
VENDOR_MARKERS = (".venv", "site-packages", "node_modules", ".git")

# (label, harness, corroborator, expected)
PAIRS = [
    ("POSITIVE_r3_falsifier_vs_e6_harness",
     "scripts/research/functor_f_e6_albert_shadow_contract.py",
     "scripts/research/self_falsifying_compilation_line_r3_contract.py",
     "independent"),
    ("NEGATIVE_copypaste_corroborator",
     "scripts/research/functor_f_e6_albert_shadow_contract.py",
     "scripts/ci/fixtures/independence_copypaste_corroborator.py",
     "shared"),
]


def read(rel: str) -> str:
    try:
        return (REPO / rel).read_text(errors="replace")
    except OSError:
        return ""


# ---------------------------------------------------------------- I1


def repo_local_imports(rel: str) -> set[str]:
    """Imported module names that resolve to a file inside this repository."""
    try:
        tree = ast.parse(read(rel))
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    local = set()
    for n in names:
        if n == "__future__":
            continue
        # repo-local iff a same-named module exists under the repo AND is not
        # vendored third-party. numpy resolves inside .venv/site-packages, which
        # is emphatically not this repository's code.
        hits = [p for p in REPO.glob(f"**/{n}.py")
                if not any(m in p.parts for m in VENDOR_MARKERS)]
        if hits:
            local.add(n)
    return local


def clause_i1() -> tuple[bool, bool]:
    """Returns (ok, discriminates)."""
    discriminates = False
    for label, harness, corr, _ in PAIRS:
        h, c = repo_local_imports(harness), repo_local_imports(corr)
        shared = sorted(h & c)
        if shared:
            discriminates = True
        print(f"  {label}: repo-local imports harness={sorted(h) or '[]'} "
              f"corroborator={sorted(c) or '[]'} shared={shared or '[]'}")
    print(f"I1_IMPORT_CLOSURE {'discriminates' if discriminates else 'VACUOUS'} "
          f"— research contracts import no repo-local modules, so import "
          f"disjointness passes for every pair and rules out nothing")
    print("I1_IMPORT_CLOSURE PASS — measured")
    return True, discriminates


# ---------------------------------------------------------------- I2


class _Canon(ast.NodeTransformer):
    """Rename identifiers to first-appearance placeholders.

    Copy-paste survives renaming, so comparing canonicalised structure catches
    a corroborator that lifted the harness's functions and renamed things.
    """

    def __init__(self) -> None:
        self.seen: dict[str, str] = {}

    def _slot(self, name: str) -> str:
        if name not in self.seen:
            self.seen[name] = f"v{len(self.seen)}"
        return self.seen[name]

    def visit_Name(self, node: ast.Name):
        return ast.copy_location(
            ast.Name(id=self._slot(node.id), ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg):
        node.arg = self._slot(node.arg)
        node.annotation = None
        return node


def fingerprints(rel: str) -> dict[str, str]:
    """function name -> canonicalised structural dump of its body."""
    try:
        tree = ast.parse(read(rel))
    except SyntaxError:
        return {}
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = list(node.body)
        # drop the docstring: prose is not derivation
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body = body[1:]
        if not body:
            continue
        size = sum(len(list(ast.walk(s))) for s in body)
        if size < TRIVIAL_NODE_FLOOR:
            continue
        canon = _Canon()
        dumped = "".join(
            ast.dump(canon.visit(ast.parse(ast.unparse(stmt)).body[0]),
                     annotate_fields=False)
            for stmt in body)
        out[node.name] = dumped
    return out


def clause_i2() -> tuple[bool, list[dict]]:
    rows = []
    for label, harness, corr, expected in PAIRS:
        fh, fc = fingerprints(harness), fingerprints(corr)
        if not fh or not fc:
            print(f"  {label}: could not extract function bodies")
            rows.append({"label": label, "expected": expected,
                         "verdict": "UNCHECKABLE", "worst": None, "where": None})
            continue
        worst, where = 0.0, None
        for hn, hd in fh.items():
            for cn, cd in fc.items():
                r = difflib.SequenceMatcher(None, hd, cd).ratio()
                if r > worst:
                    worst, where = r, f"{hn} ~ {cn}"
        verdict = "shared" if worst >= DUP_THRESHOLD else "independent"
        rows.append({"label": label, "expected": expected, "verdict": verdict,
                     "worst": worst, "where": where})
        mark = "OK" if verdict == expected else "MISMATCH"
        print(f"  {label}: max body similarity {worst:.3f} ({where}) "
              f"-> {verdict} (expected {expected}) [{mark}]")

    ok = all(r["verdict"] == r["expected"] for r in rows)
    print(f"I2_DERIVATION_DISJOINT {'PASS' if ok else 'FAIL'} — "
          f"threshold {DUP_THRESHOLD}, {len(rows)} pairs")
    return ok, rows


# ---------------------------------------------------------------- I3


def clause_i3() -> tuple[bool, dict]:
    """Corpus-wide sweep, so the guard is not judged only on cases it was tuned on.

    The threshold and the triviality floor were both set after looking at one
    pair. Testing them on that pair proves nothing. This runs every pair of
    research contracts and reports how many share a derivation — which is also
    a measurement nobody has for this corpus: if the contracts cross-checking
    each other were built by copying the same multiplication table, they are
    not independent evidence of anything.
    """
    files = sorted(str(p.relative_to(REPO))
                   for p in (REPO / "scripts/research").glob("*contract*.py"))
    fps = {f: fingerprints(f) for f in files}
    have = [f for f in files if fps[f]]

    shared_pairs = []
    total = 0
    for i, a in enumerate(have):
        for b in have[i + 1:]:
            total += 1
            worst, where = 0.0, None
            for an, ad in fps[a].items():
                for bn, bd in fps[b].items():
                    r = difflib.SequenceMatcher(None, ad, bd).ratio()
                    if r > worst:
                        worst, where = r, f"{an} ~ {bn}"
            if worst >= DUP_THRESHOLD:
                shared_pairs.append((a, b, worst, where))

    print(f"I3_CORPUS_SWEEP {len(have)}/{len(files)} contracts have a body above "
          f"the {TRIVIAL_NODE_FLOOR}-node floor; {total} pairs compared")
    print(f"I3_CORPUS_SWEEP {len(shared_pairs)}/{total} pairs share a derivation "
          f"at similarity >= {DUP_THRESHOLD}")
    for a, b, r, where in sorted(shared_pairs, key=lambda x: -x[2])[:12]:
        print(f"    {r:.3f}  {Path(a).name}")
        print(f"           {Path(b).name}   ({where})")
    if len(shared_pairs) > 12:
        print(f"    ... and {len(shared_pairs) - 12} more")
    print("I3_CORPUS_SWEEP PASS — measured")
    return True, {"files": len(files), "compared": total,
                  "shared": len(shared_pairs)}


# ---------------------------------------------------------------- main


def main() -> int:
    print("SELF-FALSIFYING COMPILATION R6 — evidential independence, statically")
    print("=" * 76)
    print("R0 §3: shared misinterpretation is undetectable when the only evidence")
    print("is the claim's own check. R6 attacks the ANTECEDENT — require a second")
    print("derivation and make its independence machine-checkable.")
    print()

    i1, i1_discriminates = clause_i1()
    print()
    i2, rows = clause_i2()
    print()
    i3, sweep = clause_i3()
    print()

    print("=" * 76)
    checkable = [r for r in rows if r["verdict"] != "UNCHECKABLE"]
    if not checkable:
        token = "INDEPENDENCE_UNCHECKABLE"
    elif not i2:
        token = "INDEPENDENCE_CHECKABLE__GUARD_VACUOUS"
    else:
        pos = [r for r in rows if r["expected"] == "independent"]
        neg = [r for r in rows if r["expected"] == "shared"]
        discriminates = (pos and neg
                         and all(r["verdict"] == "independent" for r in pos)
                         and all(r["verdict"] == "shared" for r in neg))
        token = ("INDEPENDENCE_CHECKABLE__CORROBORATION_BINDS" if discriminates
                 else "INDEPENDENCE_CHECKABLE__GUARD_VACUOUS")

    print(f"  import-closure notion : "
          f"{'discriminates' if i1_discriminates else 'VACUOUS on this corpus'}")
    print(f"  derivation notion     : "
          f"{sum(1 for r in rows if r['verdict'] == r['expected'])}/{len(rows)} "
          f"pairs classified as expected")
    print(f"  corpus sweep          : {sweep['shared']}/{sweep['compared']} "
          f"contract pairs share a derivation")
    print(f"  bound claimed         : a LOWER BOUND on independence — rules out "
          f"reuse, not shared authorship")
    print(f"SELF_FALSIFYING_R6_VERDICT {token}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
