#!/usr/bin/env python3
"""Self-falsifying compilation, rung R8 — the trusted base of a research corpus.

Spec: docs/research/self_falsifying_compilation_line_r8_2026-07-26.md

R6 found that a third of this corpus's contract pairs share a derivation. R7
audited one shared function and found it sound. Two data points are not a
method. R8 makes it one:

    1. enumerate EVERY shared derivation, not just the biggest
    2. rank by blast radius — how many contracts inherit it
    3. collapse WRAPPERS into the kernels they call, because a wrapper's blast
       radius is really its kernel's
    4. audit what is left: the irreducible base the corpus rests on

Step 3 is what turns a list into a base. `omul`, `mul` and `o` all multiply by
looking up `cds`; auditing them audits `cds` plus a loop. The irreducible
kernels are the ones that compute structure constants from nothing.

WHAT THE AUDIT FOUND, and it was not designed this way: the corpus already
contained a SECOND, independent derivation of its own sign table — `cd_sigma`,
recursive, structurally unrelated to the iterative `cds` — sitting in three
contracts, never compared with it. R8 compares them, and adds a third
derivation of its own.

VERDICT OPTIONS, FIXED BEFORE COMPUTING (see main()):
  TRUSTED_BASE_MAPPED__KERNELS_AGREE
      the base is enumerated and every independent derivation of it agrees.
  TRUSTED_BASE_MAPPED__KERNELS_DIVERGE
      the base is enumerated and two derivations of it disagree — meaning part
      of the corpus computes something different from another part.
  TRUSTED_BASE_UNMAPPABLE
      clusters or kernels cannot be extracted.

Pure Python 3 + numpy.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
R6 = REPO / "scripts/research/self_falsifying_compilation_line_r6_contract.py"


def _load_r6():
    spec = importlib.util.spec_from_file_location("r6", R6)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- oracle
# Third derivation, re-derived here: recursive Cayley-Dickson on split arrays.
# No sign table appears; the signs fall out of the doubling recursion.


def cd_conj(x):
    c = -x.copy()
    c[0] = x[0]
    return c


def cd_mul(x, y):
    n = len(x)
    if n == 1:
        return x * y
    h = n // 2
    a, b, c, d = x[:h], x[h:], y[:h], y[h:]
    return np.concatenate([cd_mul(a, c) - cd_mul(cd_conj(d), b),
                           cd_mul(d, a) + cd_mul(b, cd_conj(c))])


def oracle_sign(i: int, j: int, bits: int):
    n = 1 << bits
    x = np.zeros(n); x[i] = 1.0
    y = np.zeros(n); y[j] = 1.0
    p = cd_mul(x, y)
    k = i ^ j
    nz = np.nonzero(np.abs(p) > 1e-9)[0]
    if len(nz) != 1 or nz[0] != k:
        return None
    return int(round(p[k]))


# ---------------------------------------------------------------- helpers


def extract(rel: str, fn: str):
    try:
        src = (REPO / rel).read_text(errors="replace")
        for node in ast.parse(src).body:
            if isinstance(node, ast.FunctionDef) and node.name == fn:
                ns: dict = {"np": np}
                exec(compile(ast.Module(body=[node], type_ignores=[]),
                             f"<{rel}:{fn}>", "exec"), ns)
                return ns[fn]
    except (OSError, SyntaxError, TypeError, NameError):
        return None
    return None


def calls_of(rel: str, fn: str) -> set[str]:
    """Names this function calls — used to tell a wrapper from a kernel."""
    try:
        for node in ast.parse((REPO / rel).read_text(errors="replace")).body:
            if isinstance(node, ast.FunctionDef) and node.name == fn:
                return {n.func.id for n in ast.walk(node)
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    except (OSError, SyntaxError):
        pass
    return set()


# ---------------------------------------------------------------- K1


def clause_k1(r6) -> tuple[bool, list]:
    files = sorted(str(p.relative_to(REPO))
                   for p in (REPO / "scripts/research").glob("*contract*.py"))
    by_fp = defaultdict(list)
    for f in files:
        for name, fp in r6.fingerprints(f).items():
            by_fp[fp].append((f, name))
    clusters = sorted((v for v in by_fp.values() if len(v) > 1),
                      key=len, reverse=True)
    instances = sum(len(c) for c in clusters)
    print(f"K1_CLUSTER_MAP {len(clusters)} shared-derivation clusters over "
          f"{len(files)} contracts, {instances} function instances")
    for c in clusters[:8]:
        names = sorted({n for _, n in c})
        print(f"    blast radius {len(c):2d}  {','.join(names)[:38]}")
    if len(clusters) > 8:
        print(f"    ... and {len(clusters) - 8} smaller clusters")
    ok = bool(clusters)
    print(f"K1_CLUSTER_MAP {'PASS' if ok else 'FAIL'} — measured")
    return ok, clusters


# ---------------------------------------------------------------- K2


def clause_k2(clusters) -> tuple[bool, list, list]:
    """Split clusters into wrappers (call another shared function) and kernels."""
    shared_names = {n for c in clusters for _, n in c}
    kernels, wrappers = [], []
    for c in clusters:
        rel, fn = c[0]
        called = calls_of(rel, fn)
        inherited = (called & shared_names) - {fn}   # parenthesised: this line
        if inherited:                                   # decides kernel vs wrapper
            wrappers.append((len(c), fn, sorted(inherited)))
        else:
            kernels.append((len(c), fn, rel))
    kernels.sort(reverse=True)
    wrappers.sort(reverse=True)

    print(f"K2_TRUSTED_BASE {len(kernels)} irreducible kernels, "
          f"{len(wrappers)} wrappers that inherit another kernel's risk")
    for n, fn, rel in kernels[:6]:
        print(f"    kernel   radius {n:2d}  {fn:22s} {Path(rel).name[:36]}")
    for n, fn, called in wrappers[:6]:
        print(f"    wrapper  radius {n:2d}  {fn:22s} -> calls {called}")
    ok = bool(kernels)
    print(f"K2_TRUSTED_BASE {'PASS' if ok else 'FAIL'} — measured")
    return ok, kernels, wrappers


# ---------------------------------------------------------------- K3


SIGN_KERNELS = [
    ("cds (iterative)", "scripts/research/functor_f_e6_albert_shadow_contract.py", "cds"),
    ("cds (variant 2)", "scripts/research/cd_tower_nullity_histogram_law_contract.py", "cds"),
    ("cd_sigma (recursive)",
     "scripts/research/cd_tower_zd_fiber_signed_localization_contract.py", "cd_sigma"),
]


def clause_k3() -> tuple[bool, dict]:
    """Every independent derivation of the sign table, against each other and
    against a third one re-derived here."""
    impls = []
    for label, rel, fn in SIGN_KERNELS:
        f = extract(rel, fn)
        if f is None:
            print(f"    could not extract {label}")
            continue
        impls.append((label, f))
    impls.append(("oracle (this harness, recursive on split arrays)",
                  lambda i, j, bits: oracle_sign(i, j, bits)))

    if len(impls) < 2:
        print("K3_KERNELS_AGREE FAIL — fewer than two derivations available")
        return False, {}

    # Independence is MEASURED, not asserted from reading the source — that is
    # what R6 exists to replace. Implementations closer than R6's threshold are
    # the same derivation in different clothes and must not be counted twice.
    r6 = _load_r6()
    fps = {}
    for label, rel, fn in SIGN_KERNELS:
        f = r6.fingerprints(rel).get(fn)
        if f:
            fps[label] = f
    ora = r6.fingerprints("scripts/research/self_falsifying_compilation_line_r8_contract.py")
    if "cd_mul" in ora:
        fps["oracle (this harness, recursive on split arrays)"] = ora["cd_mul"]

    import difflib as _dl
    labels = list(fps)
    print("    independence matrix (canonicalised body similarity):")
    same_as = {}
    for a in range(len(labels)):
        for b in range(a + 1, len(labels)):
            s = _dl.SequenceMatcher(None, fps[labels[a]], fps[labels[b]]).ratio()
            verdict = "SAME derivation" if s >= r6.DUP_THRESHOLD else "independent"
            print(f"      {labels[a][:26]:28s} vs {labels[b][:26]:28s} "
                  f"{s:.3f}  {verdict}")
            if s >= r6.DUP_THRESHOLD:
                same_as.setdefault(labels[b], labels[a])
    distinct = [l for l in labels if l not in same_as]
    print(f"    -> {len(distinct)} DISTINCT derivations "
          f"({len(labels) - len(distinct)} are textual variants of another)")

    total = 0
    ungradeable = 0
    disagreements = []
    for bits in (3, 4, 5, 6):
        n = 1 << bits
        for i in range(n):
            for j in range(n):
                vals = []
                for label, f in impls:
                    try:
                        vals.append((label, f(i, j, bits)))
                    except TypeError:
                        vals.append((label, f(i, j)))
                # A None means the product was not a signed basis element,
                # which for a Cayley-Dickson algebra is an anomaly, not a
                # non-event. Counted separately so "0 disagreements" means
                # "over N fully comparable products".
                if any(v is None for _, v in vals):
                    ungradeable += 1
                    continue
                total += 1
                base = vals[0][1]
                for label, v in vals[1:]:
                    if v != base:
                        disagreements.append((bits, i, j, vals))
                        break

    ok = not disagreements and ungradeable == 0
    print(f"K3_KERNELS_AGREE {len(distinct)} distinct derivations "
          f"({len(impls)} implementations) compared over {total} fully "
          f"comparable basis products (levels 3-6); {ungradeable} ungradeable")
    if disagreements:
        b, i, j, vals = disagreements[0]
        print(f"    FIRST DISAGREEMENT level {b}, e_{i}*e_{j}: "
              + ", ".join(f"{l}={v}" for l, v in vals))
    print(f"K3_KERNELS_AGREE {'PASS' if ok else 'FAIL'} — "
          f"{len(disagreements)} disagreements")
    return ok, {"impls": len(impls), "distinct": len(distinct),
                "products": total, "disagreements": len(disagreements),
                "ungradeable": ungradeable}


# ---------------------------------------------------------------- main


def main() -> int:
    print("SELF-FALSIFYING COMPILATION R8 — the trusted base of a research corpus")
    print("=" * 78)
    print("R6: a third of the pairs share a derivation. R7: one shared function is")
    print("sound. R8: enumerate the whole shared base, collapse wrappers into the")
    print("kernels they call, and audit what is irreducible.")
    print()

    r6 = _load_r6()
    k1, clusters = clause_k1(r6)
    print()
    k2, kernels, wrappers = clause_k2(clusters)
    print()
    k3, stats = clause_k3()
    print()

    print("=" * 78)
    if not (k1 and k2) or not stats:
        token = "TRUSTED_BASE_UNMAPPABLE"
    elif k3:
        token = "TRUSTED_BASE_MAPPED__KERNELS_AGREE"
    else:
        token = "TRUSTED_BASE_MAPPED__KERNELS_DIVERGE"

    print(f"  shared clusters      : {len(clusters)}")
    print(f"  irreducible kernels  : {len(kernels)}  (wrappers: {len(wrappers)})")
    print(f"  sign derivations     : {stats.get('distinct', 0)} distinct "
          f"({stats.get('impls', 0)} implementations) over "
          f"{stats.get('products', 0)} products")
    print(f"  disagreements        : {stats.get('disagreements', 0)} "
          f"({stats.get('ungradeable', 0)} ungradeable)")
    print(f"SELF_FALSIFYING_R8_VERDICT {token}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
