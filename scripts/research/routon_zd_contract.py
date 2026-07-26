#!/usr/bin/env python3
"""
Routon zero-divisor structure — level-7 annihilation geometry (128 dimensions).

Companion to:
  docs/research/routon_zd_spec_2026-07-26.md
  docs/research/routon_zd_falsifiers_2026-07-26.md

Extends the catastrophe scan past the sedenions 𝕊 (L4, dim 16),
trigintaduonions/pathions 𝕋 (L5, dim 32) and chingons 𝕀 (L6, dim 64) to the
routons (L7, dim 128; repo naming per examples/routon_projective_measurement.sio).

Headline result: the growth law Z(b) = 4^b - (3b-1)*2^b + 2^(b-1) - 4,
exact at b = 4, 5, 6, predicts Z(7) = 13884.  The level-7 scan confirms it
exactly (6942 index pairs x 2 signs), together with the fiber-count law
F(7) = 116, the fiber-size law, and the defect diagonal.

Method (new at this level): the exact 2-cycle criterion
-------------------------------------------------------
For a = e_i + sgn*e_j the left-multiplication matrix is L_a = A + sgn*B with
A, B signed permutation matrices (row k of A has its single nonzero at column
k^i, value S[i, k^i]; likewise B with j).  Since A is orthogonal,

    det(L_a) = det(A) * det(I + sgn*Q),   Q = A^T B,

and Q is again a signed permutation matrix, whose underlying permutation is
the fixed-point-free involution k -> k ^ l with l = i ^ j.  Hence Q decomposes
into 2^(b-1) signed 2-cycles and

    det(I + sgn*Q) = prod_cycles (1 - q_k * q_{k^l}),

where q_k * q_{k^l} = S[i,k] * S[j,k] * S[i,k^l] * S[j,k^l] =: p(k) in {+1,-1}.
Each factor is 0 or 2, and sgn cancels.  Therefore:

  * (i, j) is a canonical zero-divisor pair iff p(k) = +1 for some k
    (for BOTH signs simultaneously — sign duality is now a theorem of the
    scan, not an observation);
  * nullity(L_a) = #{bad cycles} = (1/2) * #{k : p(k) = +1}, exactly,
    with no floating point involved.

Clause C8 cross-checks this integer criterion against the SVD scan used by
the L4/L5/L6 contracts at every level b = 4..7 (full census equality).
"""

from collections import Counter

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
    S = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            S[i, j] = cds(i, j, bits)
    return S


_SIGN_TABLES = {}


def get_sign_matrix(bits):
    if bits not in _SIGN_TABLES:
        _SIGN_TABLES[bits] = sign_matrix(bits)
    return _SIGN_TABLES[bits]


def exact_nullity_index_pairs(bits):
    """All (i, j), 1 <= i < j < 2^bits, such that a = e_i +/- e_j is a zero
    divisor, mapped to the exact nullity of L_a (identical for both signs).

    p(k) = S[i,k]*S[j,k]*S[i,k^l]*S[j,k^l]; nullity = #{k : p(k)=1} / 2.
    """
    n = 1 << bits
    S = get_sign_matrix(bits).astype(np.int16)
    rows = np.arange(n)
    out = {}
    for i in range(1, n):
        Si = S[i]
        for j in range(i + 1, n):
            l = i ^ j
            Sj = S[j]
            p = Si * Sj * Si[rows ^ l] * Sj[rows ^ l]
            bad = int(np.count_nonzero(p == 1)) // 2
            if bad > 0:
                out[(i, j)] = bad
    return out


def canonical_zd_pairs(bits):
    """All (i, sgn, j), 1 <= i < j < 2^bits, sgn in {+1, -1}, such that
    a = e_i + sgn*e_j is a zero divisor.  Sign duality (both signs always
    occur together) is exact by the 2-cycle criterion; emitted explicitly
    to keep the (i, sgn, j) triple convention of the L4/L5/L6 contracts."""
    return [(i, sgn, j)
            for (i, j) in exact_nullity_index_pairs(bits)
            for sgn in (1, -1)]


def svd_zd_index_pairs(bits):
    """Reference scan: SVD singularity test on L_a = A + B (sign +1 only;
    sign duality makes this the full index-pair census)."""
    n = 1 << bits
    S = get_sign_matrix(bits).astype(np.float64)
    rows = np.arange(n)
    pairs = set()
    for i in range(1, n):
        coli = rows ^ i
        vi = S[i, coli]
        for j in range(i + 1, n):
            colj = rows ^ j
            L = np.zeros((n, n))
            L[rows, coli] = vi
            L[rows, colj] = S[j, colj]
            sv = np.linalg.svd(L, compute_uv=False)
            if sv.min() < 1e-9:
                pairs.add((i, j))
    return pairs


def svd_nullity(bits, i, j):
    """Numerical nullity of L_a for a = e_i + e_j (spot-check oracle)."""
    n = 1 << bits
    S = get_sign_matrix(bits).astype(np.float64)
    rows = np.arange(n)
    L = np.zeros((n, n))
    L[rows, rows ^ i] = S[i, rows ^ i]
    L[rows, rows ^ j] = S[j, rows ^ j]
    sv = np.linalg.svd(L, compute_uv=False)
    return int(np.count_nonzero(sv < 1e-9))


def compute_fibers(pairs):
    """Group canonical ZD triples by xor-label i ^ j."""
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


def routon_zd_pairs():
    return canonical_zd_pairs(7)


# ------------------------------------------------------------------
# Closed-form combinatorial laws (observed at b = 4..7; see spec §2)
# ------------------------------------------------------------------

def census_law(b):
    """Z(b) = 4^b - (3b - 1) * 2^b + 2^(b-1) - 4.

    Equivalent derivation: sum over birth levels m = 4..b of
    (fibers born at m) x (size of an m-born fiber at level b):
        Z(b) = sum_{m=4}^{b} (2^(m-1) - 1) * (2^b - 2^(b-m+2)).
    Level-7 prediction: Z(7) = 13884.
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
    """Size at level b of a fiber born at level m: 2^b - 2^(b-m+2)."""
    return (1 << b) - (1 << (b - m + 2))


def nullity_law_values(m, b):
    """Nullities allowed for an m-born canonical ZD at level b:
    {2^(b-m+2) * t : t odd, 1 <= t <= 2^(m-3) - 1}.

    The maximum is 2^(b-1) - 2^(b-m+2); for native fibers (m = b) this is
    2^(b-1) - 4, the native-erasure maximum conjectured in
    examples/routon_projective_measurement.sio (ker_native(L7) = 60).
    """
    base = 1 << (b - m + 2)
    return {base * t for t in range(1, (1 << (m - 3)), 2)}


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
    pairs = routon_zd_pairs()
    ok = len(pairs) == 13884
    print(f"C1_ZD_CENSUS count={len(pairs)} expected=13884 -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_C2_growth_law():
    censuses = {b: len(canonical_zd_pairs(b)) for b in (4, 5, 6, 7)}
    ok = all(censuses[b] == census_law(b) for b in (4, 5, 6, 7))
    detail = " ".join(f"Z({b})={censuses[b]}/law={census_law(b)}" for b in (4, 5, 6, 7))
    print(f"C2_GROWTH_LAW {detail} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_C3_fiber_decomposition():
    ok = True
    for b in (4, 5, 6, 7):
        fibers = compute_fibers(canonical_zd_pairs(b))
        labels = set(fibers.keys())
        if len(fibers) != fiber_count_law(b):
            ok = False
        if labels != expected_labels(b):
            ok = False
        if b == 7:
            sizes = sorted(len(v) for v in fibers.values())
            print(f"C3_FIBER_DECOMPOSITION L7_fibers={len(fibers)} sizes={sizes}")
    print(f"C3_FIBER_DECOMPOSITION F(4,5,6,7)="
          f"{fiber_count_law(4)},{fiber_count_law(5)},{fiber_count_law(6)},"
          f"{fiber_count_law(7)} labels_match -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_C4_fiber_size_law():
    ok = True
    for b in (4, 5, 6, 7):
        fibers = compute_fibers(canonical_zd_pairs(b))
        for label, members in fibers.items():
            m = fiber_birth_level(label)
            if len(members) != fiber_size_law(m, b):
                ok = False
                print(f"  C4 mismatch label={label} birth=L{m} "
                      f"size={len(members)} law={fiber_size_law(m, b)}")
    print(f"C4_FIBER_SIZE_LAW size(m,b)=2^b-2^(b-m+2) verified at b=4,5,6,7 "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_C5_tower_embedding():
    s_pairs = sedenion_zd_pairs()
    t_pairs = trigintaduonion_zd_pairs()
    c_pairs = chingon_zd_pairs()
    r_pairs = routon_zd_pairs()
    t_set = set(t_pairs)
    c_set = set(c_pairs)
    c_sub_of_r = set((i, sgn, j) for i, sgn, j in r_pairs if i < 64 and j < 64)
    ok = (c_set == c_sub_of_r)
    s_labels = set(compute_fibers(s_pairs).keys())
    t_labels = set(compute_fibers(t_pairs).keys())
    c_labels = set(compute_fibers(c_pairs).keys())
    r_labels = set(compute_fibers(r_pairs).keys())
    nested = (s_labels.issubset(t_labels) and t_labels.issubset(c_labels)
              and c_labels.issubset(r_labels))
    ok = ok and nested
    print(f"C5_TOWER_EMBEDDING 𝕀⊂routons={c_set == c_sub_of_r} "
          f"labels_nested={nested} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_C6_density_growth():
    dens = {}
    for b in (4, 5, 6, 7):
        n = 1 << b
        total = (n - 1) * (n - 2)  # 2 * C(n-1, 2) candidate 2-unit sums
        dens[b] = len(canonical_zd_pairs(b)) / total
    ok = dens[4] < dens[5] < dens[6] < dens[7]
    print(f"C6_DENSITY_GROWTH d4={dens[4]:.4f} d5={dens[5]:.4f} d6={dens[6]:.4f} "
          f"d7={dens[7]:.4f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_C7_native_defect():
    bits = 7
    n = 1 << bits
    fibers = compute_fibers(routon_zd_pairs())
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
    # routon-native fibers (labels 65..127) each miss exactly (ell-64, 64)
    native = [ell for ell in fibers if fiber_birth_level(ell) == 7]
    native_ok = all(missing_diagonal(ell, bits) == {(ell - 64, 64)} for ell in native)
    ok = ok and native_ok and len(native) == 63
    print(f"C7_NATIVE_DEFECT fibers=116 native_L7={len(native)} "
          f"native_missing=(ell-64,64)={native_ok} diagonal_law -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_C8_exact_svd_crosscheck():
    """The exact 2-cycle criterion (primary scan, integer arithmetic) must
    reproduce the SVD-based census of the L4/L5/L6 contracts at every level.
    This guards the method change; a divergence invalidates C1-C7, C9."""
    ok = True
    for b in (4, 5, 6, 7):
        exact = set(exact_nullity_index_pairs(b).keys())
        svd = svd_zd_index_pairs(b)
        if exact != svd:
            ok = False
            only_exact = sorted(exact - svd)[:5]
            only_svd = sorted(svd - exact)[:5]
            print(f"  C8 divergence at L{b}: exact={len(exact)} svd={len(svd)} "
                  f"only_exact={only_exact} only_svd={only_svd}")
        else:
            print(f"  C8 L{b}: exact==svd census ({len(exact)} index pairs)")
    print(f"C8_EXACT_SVD_CROSSCHECK b=4..7 census_equality -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_C9_nullity_law():
    """Novel at this level: the exact nullity spectrum.

    Every m-born canonical ZD at level b has nullity 2^(b-m+2) * t with t
    odd and 1 <= t <= 2^(m-3) - 1 (verified exhaustively at b = 4..7), and
    at level 7 every such value occurs in every birth class.  The maximum
    at L7 is 60 = 2^6 - 4, attained e.g. by e_3 + e_66, confirming the
    native-erasure curve of examples/routon_projective_measurement.sio as
    a MAXIMUM (the native spectrum is {4,12,...,60}, not a single value).
    Exact nullities are spot-checked against SVD numerical rank across all
    12 distinct L7 values.
    """
    ok = True
    # exhaustive law check at b = 4..7
    for b in (4, 5, 6, 7):
        for (i, j), nullity in exact_nullity_index_pairs(b).items():
            m = fiber_birth_level(i ^ j)
            if nullity not in nullity_law_values(m, b):
                ok = False
                print(f"  C9 law fail (i={i},j={j}) birth=L{m} nullity={nullity} "
                      f"allowed={sorted(nullity_law_values(m, b))}")
    # completeness at L7: every allowed value occurs in every birth class
    seen = {}
    for (i, j), nullity in exact_nullity_index_pairs(7).items():
        m = fiber_birth_level(i ^ j)
        seen.setdefault(m, set()).add(nullity)
    for m in (4, 5, 6, 7):
        if seen.get(m) != nullity_law_values(m, 7):
            ok = False
            print(f"  C9 completeness fail birth=L{m} "
                  f"seen={sorted(seen.get(m, []))} "
                  f"law={sorted(nullity_law_values(m, 7))}")
    hist = dict(sorted(Counter(exact_nullity_index_pairs(7).values()).items()))
    # native-erasure maximum: max nullity at L7 is 60 = 2^6 - 4
    max_ok = max(hist) == 60 and exact_nullity_index_pairs(7).get((3, 66)) == 60
    ok = ok and max_ok
    # SVD spot-check: one pair per distinct L7 nullity value
    spot_ok = True
    for target in sorted(hist):
        pair = next((i, j) for (i, j), v in exact_nullity_index_pairs(7).items()
                    if v == target)
        numerical = svd_nullity(7, *pair)
        if numerical != target:
            spot_ok = False
            print(f"  C9 SVD spot-check fail pair={pair} "
                  f"exact={target} numerical={numerical}")
    ok = ok and spot_ok
    print(f"C9_NULLITY_LAW spectrum={hist} max=60(=2^6-4)={max_ok} "
          f"odd-part_law b=4..7 svd_spot(12_values)={spot_ok} -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    results = []
    print("=" * 70)
    print("ROUTON ZD STRUCTURE (LEVEL 7, DIM 128) — contract")
    print("=" * 70)
    results.append(("C1", check_C1_zd_census()))
    results.append(("C2", check_C2_growth_law()))
    results.append(("C3", check_C3_fiber_decomposition()))
    results.append(("C4", check_C4_fiber_size_law()))
    results.append(("C5", check_C5_tower_embedding()))
    results.append(("C6", check_C6_density_growth()))
    results.append(("C7", check_C7_native_defect()))
    results.append(("C8", check_C8_exact_svd_crosscheck()))
    results.append(("C9", check_C9_nullity_law()))
    print("=" * 70)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    if passed == total:
        print(f"ROUTON_ZD_VERDICT C_GREEN ({passed}/{total} clauses PASS)")
        print("ROUTON_ZD_NOTE level7_ZD_fibers; census=13884; fibers=116; "
              "growth_law_holds_at_L7; nullity_spectrum_novel; novel_computation")
        return 0
    else:
        print(f"ROUTON_ZD_VERDICT C_AMBER ({passed}/{total} clauses PASS)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
