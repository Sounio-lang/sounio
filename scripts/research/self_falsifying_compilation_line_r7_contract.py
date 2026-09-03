#!/usr/bin/env python3
"""Self-falsifying compilation, rung R7 — auditing the shared kernel R6 found.

Spec: docs/research/self_falsifying_compilation_line_r7_2026-07-26.md

R6 measured that 343 of 1081 pairs of this repository's research contracts share
a derivation, and that what they share is almost entirely ONE function: `cds`,
the Cayley-Dickson sign table, copy-pasted verbatim across the functor-F and
CD-tower corpus. Those results look mutually corroborating and are not: they
rest on a single point of failure.

Independence checking earns its keep by telling you WHERE TO LOOK. This rung
looks: it re-derives the Cayley-Dickson structure constants from first
principles, by a route structurally unrelated to `cds` (R6 measured the two at
similarity 0.151), and compares them everywhere.

ADJUDICATION MATTERS. If the two disagree, "which one is wrong?" cannot be
answered by the comparison itself. So the independent oracle is first checked
against axioms that hold for Cayley-Dickson algebras regardless of
implementation:

    e_i^2 = -1            for i >= 1
    e_i e_j = -e_j e_i    for i != j, both >= 1
    e_0 is the unit
    level 3 (octonions) is alternative: (xx)y = x(xy)
    level 4 (sedenions) has zero divisors; level 3 does not

An oracle that fails those is not evidence against anything.

VERDICT OPTIONS, FIXED BEFORE COMPUTING (see main()):
  SHARED_KERNEL_CORROBORATED   independent derivation agrees with `cds`
                               everywhere tested, and the oracle passes its
                               axioms.
  SHARED_KERNEL_REFUTED        they disagree, and the oracle passes its axioms
                               while `cds` does not.
  SHARED_KERNEL_UNTESTABLE     `cds` cannot be extracted, or the oracle fails
                               its own axioms, so nothing is adjudicated.

Pure Python 3 + numpy.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# Any contract carrying the shared kernel; they are byte-identical copies, which
# is R6's finding. This one is picked arbitrarily and the copies are checked to
# agree with it below.
KERNEL_SOURCES = [
    "scripts/research/chingon_zd_contract.py",
    "scripts/research/routon_zd_contract.py",
    "scripts/research/trigintaduonion_zd_contract.py",
    "scripts/research/g2_zd_fibers_contract.py",
    "scripts/research/zd_qec_prediction_contract.py",
    "scripts/research/cd_tower_nullity_histogram_law_contract.py",
]


# ---------------------------------------------------------------- oracle
# Independent re-derivation: recursive Cayley-Dickson on split arrays.
# (a,b)(c,d) = (ac - conj(d) b, d a + b conj(c)). Nothing here consults a sign
# table; the signs FALL OUT of the doubling recursion.


def cd_conj(x: np.ndarray) -> np.ndarray:
    c = -x.copy()
    c[0] = x[0]
    return c


def cd_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    if n == 1:
        return x * y
    h = n // 2
    a, b = x[:h], x[h:]
    c, d = y[:h], y[h:]
    return np.concatenate([
        cd_mul(a, c) - cd_mul(cd_conj(d), b),
        cd_mul(d, a) + cd_mul(b, cd_conj(c)),
    ])


def basis(i: int, n: int) -> np.ndarray:
    v = np.zeros(n)
    v[i] = 1.0
    return v


def oracle_sign(i: int, j: int, bits: int) -> int | None:
    """Sign s with e_i e_j = s * e_(i xor j), or None if the product is not
    a signed basis element (which would itself be a finding)."""
    n = 1 << bits
    p = cd_mul(basis(i, n), basis(j, n))
    k = i ^ j
    nz = np.nonzero(np.abs(p) > 1e-9)[0]
    if len(nz) != 1 or nz[0] != k or abs(abs(p[k]) - 1.0) > 1e-9:
        return None
    return int(round(p[k]))


# ---------------------------------------------------------------- axioms


def clause_a1_oracle_axioms() -> tuple[bool, list[str]]:
    notes = []
    ok = True

    for bits in (3, 4, 5):
        n = 1 << bits
        for i in range(1, n):
            if oracle_sign(i, i, bits) != -1:
                notes.append(f"e_{i}^2 != -1 at level {bits}")
                ok = False
                break
        for i in range(1, min(n, 16)):
            for j in range(1, min(n, 16)):
                if i == j:
                    continue
                a, b = oracle_sign(i, j, bits), oracle_sign(j, i, bits)
                if a is None or b is None or a != -b:
                    notes.append(f"anticommutativity fails at ({i},{j}) level {bits}")
                    ok = False
                    break
            if not ok:
                break
        for i in range(n):
            if oracle_sign(0, i, bits) != 1 or oracle_sign(i, 0, bits) != 1:
                notes.append(f"e_0 is not the unit at level {bits}")
                ok = False
                break

    # Octonions are alternative; sedenions are not.
    rng = np.random.default_rng(20260726)
    worst_alt = 0.0
    for _ in range(200):
        x, y = rng.normal(size=8), rng.normal(size=8)
        worst_alt = max(worst_alt, float(np.max(np.abs(
            cd_mul(cd_mul(x, x), y) - cd_mul(x, cd_mul(x, y))))))
    if worst_alt > 1e-9:
        notes.append(f"level 3 not alternative (max dev {worst_alt:.2e})")
        ok = False
    notes.append(f"level 3 alternativity max deviation {worst_alt:.2e}")

    # Sedenions must have zero divisors; octonions must not.
    def has_zd(bits: int) -> bool:
        n = 1 << bits
        for i in range(1, n):
            for j in range(i + 1, n):
                a, b = basis(i, n) + basis(j, n), basis(i, n) - basis(j, n)
                for u, v in ((a, b), (b, a)):
                    for k in range(1, n):
                        for m in range(k + 1, n):
                            pass
                    break
                break
            break
        # direct search over the canonical 2-term pairs
        for i in range(1, n):
            for j in range(i + 1, n):
                x = basis(i, n) + basis(j, n)
                for k in range(1, n):
                    for m in range(k + 1, n):
                        y = basis(k, n) - basis(m, n)
                        if np.max(np.abs(cd_mul(x, y))) < 1e-9:
                            return True
        return False

    zd8, zd16 = has_zd(3), has_zd(4)
    if zd8 or not zd16:
        notes.append(f"zero-divisor structure wrong: level3={zd8} level4={zd16}")
        ok = False
    notes.append(f"zero divisors: level 3 {zd8} (expect False), "
                 f"level 4 {zd16} (expect True)")

    print(f"A1_ORACLE_AXIOMS {'PASS' if ok else 'FAIL'} — the independent oracle "
          f"satisfies the Cayley-Dickson axioms, so it can adjudicate")
    for nline in notes:
        print(f"    {nline}")
    return ok, notes


# ---------------------------------------------------------------- kernel


def extract_cds(rel: str):
    """Pull `cds` out of a corpus contract without importing the module."""
    src = (REPO / rel).read_text(errors="replace")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "cds":
            ns: dict = {"np": np}
            exec(compile(ast.Module(body=[node], type_ignores=[]),
                         f"<{rel}>", "exec"), ns)
            return ns["cds"]
    return None


def clause_a2_copies_identical() -> tuple[bool, int]:
    fns, present = {}, []
    for rel in KERNEL_SOURCES:
        f = extract_cds(rel)
        if f is not None:
            fns[rel] = f
            present.append(rel)
    ok = len(present) >= 2
    if not ok:
        print("A2_COPIES_AGREE FAIL — fewer than two copies of `cds` extractable")
        return False, len(present)

    ref = fns[present[0]]
    mismatches = 0
    for bits in (3, 4, 5):
        n = 1 << bits
        for i in range(n):
            for j in range(n):
                base = ref(i, j, bits)
                for rel in present[1:]:
                    if fns[rel](i, j, bits) != base:
                        mismatches += 1
    ok = mismatches == 0
    print(f"A2_COPIES_AGREE {'PASS' if ok else 'FAIL'} — {len(present)} copies of "
          f"`cds` extracted; {mismatches} disagreements between them")
    return ok, len(present)


def clause_a3_kernel_vs_oracle() -> tuple[bool, dict]:
    ref = None
    for rel in KERNEL_SOURCES:
        ref = extract_cds(rel)
        if ref is not None:
            break
    if ref is None:
        print("A3_KERNEL_VS_ORACLE FAIL — `cds` not extractable")
        return False, {}

    stats = {}
    agree_all = True
    for bits in (3, 4, 5, 6):
        n = 1 << bits
        compared = disagree = degenerate = 0
        first_bad = None
        for i in range(n):
            for j in range(n):
                s_o = oracle_sign(i, j, bits)
                if s_o is None:
                    degenerate += 1
                    continue
                compared += 1
                if ref(i, j, bits) != s_o:
                    disagree += 1
                    if first_bad is None:
                        first_bad = (i, j, ref(i, j, bits), s_o)
        stats[bits] = {"compared": compared, "disagree": disagree,
                       "degenerate": degenerate, "first_bad": first_bad}
        if disagree:
            agree_all = False
        print(f"    level {bits} ({n:3d} units): {compared:5d} products compared, "
              f"{disagree} disagreements"
              + (f"  first at e_{first_bad[0]}*e_{first_bad[1]}: "
                 f"cds={first_bad[2]} oracle={first_bad[3]}" if first_bad else ""))

    print(f"A3_KERNEL_VS_ORACLE {'PASS' if agree_all else 'FAIL'} — the "
          f"copy-pasted sign table against an independent recursive derivation")
    return agree_all, stats


# ---------------------------------------------------------------- main


def main() -> int:
    print("SELF-FALSIFYING COMPILATION R7 — auditing the shared kernel")
    print("=" * 76)
    print("R6 found 343/1081 contract pairs sharing one function: `cds`. Those")
    print("results are not independent evidence of each other — they rest on a")
    print("single point of failure. This rung tests that point.")
    print()

    a1, _ = clause_a1_oracle_axioms()
    print()
    a2, n_copies = clause_a2_copies_identical()
    print()
    a3, stats = clause_a3_kernel_vs_oracle()
    print()

    print("=" * 76)
    if not a1:
        token = "SHARED_KERNEL_UNTESTABLE"
    elif not stats:
        token = "SHARED_KERNEL_UNTESTABLE"
    elif a3:
        token = "SHARED_KERNEL_CORROBORATED"
    else:
        token = "SHARED_KERNEL_REFUTED"

    total = sum(s["compared"] for s in stats.values()) if stats else 0
    bad = sum(s["disagree"] for s in stats.values()) if stats else 0
    print(f"  oracle axioms        : {'pass' if a1 else 'FAIL'}")
    print(f"  copies of the kernel : {n_copies}, mutually identical: {a2}")
    print(f"  products compared    : {total} across levels 3-6")
    print(f"  disagreements        : {bad}")
    print(f"SELF_FALSIFYING_R7_VERDICT {token}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
