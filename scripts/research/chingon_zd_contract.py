#!/usr/bin/env python3
"""
Chingon zero-divisor structure — level-6 annihilation geometry (64 dimensions).

Companion to:
  docs/research/chingon_zd_spec_2026-07-25.md
  docs/research/chingon_zd_falsifiers_2026-07-25.md

Extends the catastrophe scan past the sedenions 𝕊 (L4, dim 16) and the
trigintaduonions/pathions 𝕋 (L5, dim 32) to the chingons 𝕀 (L6, dim 64).

Self-contained; re-implements the Cayley-Dickson sign law for auditability.

Performance note: the naive per-pair Lmatrix assembly used in
trigintaduonion_zd_contract.py costs O(4^b) cds() calls per candidate pair,
which is impractical at b = 6 (3906 pairs x 4096 entries).  Here the sign
table S[i, j] = cds(i, j) is precomputed once per level and each left
multiplication matrix L_a for a = e_i + sgn*e_j is assembled by fancy
indexing: row k of L_a is nonzero only in columns k^i and k^j.  This is
the identical matrix, only faster to build.
"""

import numpy as np

np.seterr(all='ignore')


def cds(a, b, bits):
    s = 1
    while bits > 0:
        if a == 0 or b == 0:
            return s
        if bits == 1:
            return -s
        h = 1 << (bits - 1)
        ah = a >= h
        bh = b >= h
        al = a & (h - 1)
        bl = b & (h - 1)
        if not ah and not bh:
            a, b = al, bl
        elif not ah and bh:
            a, b = bl, al
        elif ah and not bh:
            a, b, s = ((al, 0, s) if bl == 0 else (al, bl, -s))
        else:
            a, b, s = ((0, al, -s) if bl == 0 else (bl, al, s))
        bits -= 1
    return s


def sign_matrix(bits):
    """S[i, j] = sign of e_i * e_j in the level-`bits` Cayley-Dickson algebra."""
    n = 1 << bits
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            S[i, j] = cds(i, j, bits)
    return S


_SIGN_TABLES = {}


def get_sign_matrix(bits):
    if bits not in _SIGN_TABLES:
        _SIGN_TABLES[bits] = sign_matrix(bits)
    return _SIGN_TABLES[bits]


def canonical_zd_pairs(bits):
    """All (i, sgn, j), 1 <= i < j < 2^bits, sgn in {+1, -1}, such that
    a = e_i + sgn*e_j is a zero divisor (L_a singular)."""
    n = 1 << bits
    S = get_sign_matrix(bits)
    rows = np.arange(n)
    pairs = []
    for i in range(1, n):
        coli = rows ^ i
        vi = S[i, coli]
        for j in range(i + 1, n):
            colj = rows ^ j
            vj = S[j, colj]
            for sgn in (1, -1):
                L = np.zeros((n, n))
                L[rows, coli] = vi
                L[rows, colj] = sgn * vj
                sv = np.linalg.svd(L, compute_uv=False)
                if sv.min() < 1e-9:
                    pairs.append((i, sgn, j))
    return pairs


def compute_fibers(pairs):
    """Group canonical ZD pairs by xor-label i ^ j."""
    fibers = {}
    for i, sgn, j in pairs:
        label = i ^ j
        if label not in fibers:
            fibers[label] = []
        fibers[label].append((i, sgn, j))
    return fibers


def sedenion_zd_pairs():
    return canonical_zd_pairs(4)


def trigintaduonion_zd_pairs():
    return canonical_zd_pairs(5)


def chingon_zd_pairs():
    return canonical_zd_pairs(6)


# ------------------------------------------------------------------
# Closed-form combinatorial laws (observed at b = 4, 5, 6; see spec §2)
# ------------------------------------------------------------------

def census_law(b):
    """Z(b) = 4^b - (3b - 1) * 2^b + 2^(b-1) - 4.

    Equivalent derivation: sum over birth levels m = 4..b of
    (fibers born at m) x (size of an m-born fiber at level b):
        Z(b) = sum_{m=4}^{b} (2^(m-1) - 1) * (2^b - 2^(b-m+2)).
    """
    return 4 ** b - (3 * b - 1) * (2 ** b) + 2 ** (b - 1) - 4


def fiber_count_law(b):
    """F(b) = 2^b - b - 5: labels are {8..2^b - 1} minus the powers of two."""
    return 2 ** b - b - 5


def fiber_birth_level(label):
    """A ZD label ell satisfies 2^(m-1) < ell < 2^m for a unique m >= 4;
    that m is the tower level at which the fiber is born."""
    return label.bit_length()


def fiber_size_law(m, b):
    """Size at level b of a fiber born at level m: 2^b - 2^(b-m+2).

    Equivalently: of the 2^(b-1) - 1 candidate index pairs per label
    (times 2 signs), exactly 2^(b-m+1) - 1 index pairs are missing.
    """
    return (1 << b) - (1 << (b - m + 2))


def missing_diagonal(label, b):
    """Predicted missing index pairs for the fiber of `label` at level b.

    With m = birth level and r = label - 2^(m-1) (1 <= r <= 2^(m-1) - 1),
    the missing pairs are {(a, a ^ label) : a in D \\ {0}} where D is the
    F2-xor-span of {r, 2^m, 2^(m+1), ..., 2^(b-1)}.  The fundamental
    missing pair is (r, 2^(m-1)): e_r +/- e_{2^(m-1)} is invertible and
    doubling propagates the defect up the tower.
    """
    m = fiber_birth_level(label)
    r = label - (1 << (m - 1))
    gens = [r] + [1 << k for k in range(m, b)]
    span = {0}
    for g in gens:
        span |= {x ^ g for x in list(span)}
    out = set()
    for a in span:
        if a == 0:
            continue
        i, j = a, a ^ label
        out.add((min(i, j), max(i, j)))
    return out


def expected_labels(b):
    """ZD labels at level b: {ell in [1, 2^b) : ell >= 8 and ell not a power of 2}."""
    n = 1 << b
    powers = {1 << k for k in range(b)}
    return {ell for ell in range(1, n) if ell >= 8 and ell not in powers}


# ------------------------------------------------------------------
# Contract clauses
# ------------------------------------------------------------------

def check_C1_zd_census():
    pairs = chingon_zd_pairs()
    ok = len(pairs) == 3036
    print(f"C1_ZD_CENSUS count={len(pairs)} expected=3036 -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_C2_growth_law():
    censuses = {b: len(canonical_zd_pairs(b)) for b in (4, 5, 6)}
    ok = all(censuses[b] == census_law(b) for b in (4, 5, 6))
    detail = " ".join(f"Z({b})={censuses[b]}/law={census_law(b)}" for b in (4, 5, 6))
    print(f"C2_GROWTH_LAW {detail} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_C3_fiber_decomposition():
    ok = True
    for b in (4, 5, 6):
        fibers = compute_fibers(canonical_zd_pairs(b))
        labels = set(fibers.keys())
        if len(fibers) != fiber_count_law(b):
            ok = False
        if labels != expected_labels(b):
            ok = False
        if b == 6:
            sizes = sorted(len(v) for v in fibers.values())
            print(f"C3_FIBER_DECOMPOSITION L6_fibers={len(fibers)} sizes={sizes}")
    print(f"C3_FIBER_DECOMPOSITION F(4,5,6)="
          f"{fiber_count_law(4)},{fiber_count_law(5)},{fiber_count_law(6)} "
          f"labels_match -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_C4_fiber_size_law():
    ok = True
    for b in (4, 5, 6):
        fibers = compute_fibers(canonical_zd_pairs(b))
        for label, members in fibers.items():
            m = fiber_birth_level(label)
            if len(members) != fiber_size_law(m, b):
                ok = False
                print(f"  C4 mismatch label={label} birth=L{m} "
                      f"size={len(members)} law={fiber_size_law(m, b)}")
    print(f"C4_FIBER_SIZE_LAW size(m,b)=2^b-2^(b-m+2) verified at b=4,5,6 "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_C5_tower_embedding():
    s_pairs = sedenion_zd_pairs()
    t_pairs = trigintaduonion_zd_pairs()
    c_pairs = chingon_zd_pairs()
    s_set = set((i, sgn, j) for i, sgn, j in s_pairs)
    t_sub = set((i, sgn, j) for i, sgn, j in t_pairs if i < 16 and j < 16)
    t_set = set((i, sgn, j) for i, sgn, j in t_pairs)
    c_sub = set((i, sgn, j) for i, sgn, j in c_pairs if i < 32 and j < 32)
    ok = (s_set == t_sub) and (t_set == c_sub)
    s_labels = set(compute_fibers(s_pairs).keys())
    t_labels = set(compute_fibers(t_pairs).keys())
    c_labels = set(compute_fibers(c_pairs).keys())
    ok = ok and s_labels.issubset(t_labels) and t_labels.issubset(c_labels)
    print(f"C5_TOWER_EMBEDDING 𝕊⊂𝕋={s_set == t_sub} 𝕋⊂𝕀={t_set == c_sub} "
          f"labels_nested={s_labels.issubset(t_labels) and t_labels.issubset(c_labels)} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_C6_density_growth():
    dens = {}
    for b in (4, 5, 6):
        n = 1 << b
        total = (n - 1) * (n - 2)  # 2 * C(n-1, 2) candidate 2-unit sums
        dens[b] = len(canonical_zd_pairs(b)) / total
    ok = dens[4] < dens[5] < dens[6]
    print(f"C6_DENSITY_GROWTH d4={dens[4]:.4f} d5={dens[5]:.4f} d6={dens[6]:.4f} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_C7_native_defect():
    bits = 6
    n = 1 << bits
    pairs = chingon_zd_pairs()
    fibers = compute_fibers(pairs)
    ok = True
    for label, members in sorted(fibers.items()):
        present_idx = set()
        signs = {}
        for i, sgn, j in members:
            present_idx.add((i, j))
            signs.setdefault((i, j), set()).add(sgn)
        # sign duality: every present index pair carries both signs
        if any(len(v) != 2 for v in signs.values()):
            ok = False
            print(f"  C7 sign-duality fail label={label}")
        # actual missing index pairs among the 2^(b-1) - 1 candidates
        candidates = set()
        for i in range(1, n):
            j = i ^ label
            if j > i:
                candidates.add((i, j))
        missing = candidates - present_idx
        predicted = missing_diagonal(label, bits)
        if missing != predicted:
            ok = False
            print(f"  C7 defect fail label={label} "
                  f"missing={sorted(missing)} predicted={sorted(predicted)}")
        m = fiber_birth_level(label)
        if len(missing) != (1 << (bits - m + 1)) - 1:
            ok = False
            print(f"  C7 defect-count fail label={label} "
                  f"missing={len(missing)} law={(1 << (bits - m + 1)) - 1}")
    # chingon-native fibers (labels 33..63) each miss exactly (ell-32, 32)
    native = [ell for ell in fibers if fiber_birth_level(ell) == 6]
    native_ok = all(missing_diagonal(ell, bits) == {(ell - 32, 32)} for ell in native)
    ok = ok and native_ok and len(native) == 31
    print(f"C7_NATIVE_DEFECT fibers=53 native_L6={len(native)} "
          f"native_missing=(ell-32,32)={native_ok} diagonal_law -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    results = []
    print("=" * 70)
    print("CHINGON ZD STRUCTURE (LEVEL 6, DIM 64) — contract")
    print("=" * 70)
    results.append(("C1", check_C1_zd_census()))
    results.append(("C2", check_C2_growth_law()))
    results.append(("C3", check_C3_fiber_decomposition()))
    results.append(("C4", check_C4_fiber_size_law()))
    results.append(("C5", check_C5_tower_embedding()))
    results.append(("C6", check_C6_density_growth()))
    results.append(("C7", check_C7_native_defect()))
    print("=" * 70)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    if passed == total:
        print(f"CHINGON_ZD_VERDICT C_GREEN ({passed}/{total} clauses PASS)")
        print("CHINGON_ZD_NOTE level6_ZD_fibers; census=3036; fibers=53; "
              "growth_law_verified; novel_computation")
        return 0
    else:
        print(f"CHINGON_ZD_VERDICT C_AMBER ({passed}/{total} clauses PASS)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
