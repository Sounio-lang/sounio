#!/usr/bin/env python3
"""
Functor F — cross-column: the algebra -> Petitot (morphodynamic) bridge.

Tests the reconciliation hypothesis of petitot-semantic-potential.md §3/§5: is the
Petitot stratification (cusp = contrariety, butterfly = mediating complex term)
canonically reproduced by restricting the octonion associator to coupled Fano lines?

VERDICT TYPE NAMED IN ADVANCE (D3 forbids an identity): this is an OPERATIONAL bridge
with a stated divergence, never "Petitot bifurcation set == ZD/associator locus".

Findings (probe-first, count-before-well-count, per the falsifiability rule):
  B1  §3 divergence SURVIVES: isolated Fano square has associator 0 => Booleanizable
      (cusp-level), NOT butterfly. A bridge that broke this would break the algebra.
  B2  2 coupled lines supply exactly 2 canonical continuous G2-invariants
      (depth + tilt) -> the CUSP closes canonically (this is R3_GREEN, re-derived).
  B3  3 coupled lines supply exactly 3 independent continuous G2-invariants
      -> the dimension MATCHES the butterfly's 3 controls (t,v,w). NECESSARY only.
  B4  OBSTRUCTION: the canonical antisymmetric-cubic G2 invariant phi(a1,a2,a3) on the
      three coupling jets VANISHES over all 840 3-line configs (measured; structural
      proof open), and the 3 independent invariants are three same-type single-axis
      DEPTHS (symmetric). So there is NO canonical algebra-forced x^4 "butterfly factor";
      reaching the 3-well pocket needs an arbitrary t by hand (= fabrication). The 3<->3
      match is a dimension coincidence, not a canonical map. (Argument needs only the
      domain-bounded rank=3 and phi(jets)=0, not a completeness theorem for invariants.)

Conclusion: the bridge closes CANONICALLY at the cusp, is OBSTRUCTED at the butterfly.
This locates the algebra/Petitot divergence precisely at the mediating stratum.

Self-contained; embeds an independent axiom-audit of the inherited octonion core.
"""
import numpy as np
import itertools

np.seterr(all='ignore')
EXACT = 1e-9


def cds(a, b, bits=3):
    s = 1
    while bits > 0:
        if a == 0 or b == 0:
            return s
        if bits == 1:
            return -s
        h = 1 << (bits - 1)
        ah, bh = a >= h, b >= h
        al, bl = a & (h - 1), b & (h - 1)
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


def omul(A, B):
    C = np.zeros(8)
    for i in range(8):
        for j in range(8):
            C[i ^ j] += cds(i, j) * A[i] * B[j]
    return C


def e(i):
    v = np.zeros(8); v[i] = 1.0; return v


def assoc(u, v, w):
    return omul(omul(u, v), w) - omul(u, omul(v, w))


def phi3(a, b, c):
    return float(np.dot(omul(a, b), c))


FANO = [(i, j, i ^ j) for i in range(1, 8) for j in range(i + 1, 8) if (i ^ j) > j]


def audit_core():
    ident = all(np.allclose(omul(e(0), e(j)), e(j)) for j in range(8))
    sq = all(np.allclose(omul(e(i), e(i)), -e(0)) for i in range(1, 8))
    anti = all(np.allclose(omul(e(i), e(j)), -omul(e(j), e(i)))
               for i in range(1, 8) for j in range(1, 8) if i != j)
    alt = all(np.allclose(omul(omul(e(i), e(i)), e(j)), omul(e(i), omul(e(i), e(j))))
              for i in range(8) for j in range(8))
    ok = ident and sq and anti and alt
    print(f"B0_CORE_AUDIT identity={ident} sq=-1={sq} anticomm={anti} alternative={alt} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def invariants(vecs):
    n = len(vecs); out = []
    for a in range(n):
        for b in range(a, n):
            out.append(float(np.dot(vecs[a], vecs[b])))
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                out.append(phi3(vecs[a], vecs[b], vecs[c]))
    return np.array(out)


def functional_rank(vec_fn, d0, eps=1e-6):
    I0 = vec_fn(d0)
    cols = []
    for i in range(len(d0)):
        dp = np.array(d0, float); dp[i] += eps
        cols.append((vec_fn(dp) - I0) / eps)
    return np.linalg.matrix_rank(np.array(cols).T, tol=1e-6)


def two_vecs(line, u1, u2, d):
    i, j, k = line
    return [d[0] * assoc(e(u1), e(j), e(k)), d[1] * assoc(e(i), e(u2), e(k)),
            d[0] * d[1] * assoc(e(u1), e(u2), e(k))]


def three_vecs(line, u1, u2, u3, d):
    i, j, k = line
    a1 = d[0] * assoc(e(u1), e(j), e(k))
    a2 = d[1] * assoc(e(i), e(u2), e(k))
    a3 = d[2] * assoc(e(i), e(j), e(u3))
    R12 = d[0] * d[1] * assoc(e(u1), e(u2), e(k))
    R13 = d[0] * d[2] * assoc(e(u1), e(j), e(u3))
    R23 = d[1] * d[2] * assoc(e(i), e(u2), e(u3))
    return [a1, a2, a3, R12, R13, R23]


def main():
    print("=" * 70)
    print("FUNCTOR F — algebra -> Petitot bridge (cross-column, OPERATIONAL not identity)")
    print("=" * 70)
    core = audit_core()

    # B1 — §3 divergence: isolated Fano square is associative => Booleanizable, not butterfly
    b1 = all(np.linalg.norm(assoc(e(i), e(j), e(k))) < EXACT for (i, j, k) in FANO)
    print(f"B1_DIVERGENCE_SURVIVES isolated Fano square associator=0 (Booleanizable/cusp) "
          f"{'PASS' if b1 else 'FAIL'}")

    # B2 — 2 lines -> 2 canonical continuous invariants (cusp)
    r2 = max(functional_rank(lambda dd: invariants(two_vecs((1, 2, 3), u1, u2, dd)), [0.7, 0.31])
             for u1 in [4, 5, 6] for u2 in [5, 6, 7] if u1 < u2)
    b2 = (r2 == 2)
    print(f"B2_CUSP_CANONICAL 2-line -> {r2} continuous invariants (cusp needs 2) {'PASS' if b2 else 'FAIL'}")

    # B3 — 3 lines -> 3 independent continuous invariants (dimension matches butterfly)
    r3 = max(functional_rank(lambda dd: invariants(three_vecs((1, 2, 3), u1, u2, u3, dd)),
                             [0.7, 0.31, 0.53])
             for (u1, u2, u3) in itertools.combinations([4, 5, 6, 7], 3))
    b3 = (r3 == 3)
    print(f"B3_DIM_MATCHES 3-line -> {r3} continuous invariants (butterfly needs 3; NECESSARY only) "
          f"{'PASS' if b3 else 'FAIL'}")

    # B4 — OBSTRUCTION: canonical butterfly-factor phi(a1,a2,a3) is identically 0
    rng = np.random.default_rng(0)
    worst_phi = 0.0
    ncfg = 0
    for (i, j, k) in FANO:
        offs = [u for u in range(1, 8) if u not in (i, j, k)]
        for (u1, u2, u3) in itertools.permutations(offs, 3):
            for _ in range(5):
                d = rng.random(3) * 1.5
                a1 = d[0] * assoc(e(u1), e(j), e(k))
                a2 = d[1] * assoc(e(i), e(u2), e(k))
                a3 = d[2] * assoc(e(i), e(j), e(u3))
                worst_phi = max(worst_phi, abs(phi3(a1, a2, a3)))
                ncfg += 1
    b4 = worst_phi < EXACT
    print(f"B4_BUTTERFLY_FACTOR_VANISHES canonical phi(a1,a2,a3)=0 over {ncfg} configs "
          f"(worst={worst_phi:.1e}) => NO canonical x^4 control {'PASS' if b4 else 'FAIL'}")

    print("=" * 70)
    if core and b1 and b2 and b3 and b4:
        print("FUNCTOR_F_PETITOT_VERDICT B_OBSTRUCTED (cusp-canonical, butterfly-obstructed)")
        print("FUNCTOR_F_PETITOT_NOTE cusp<->2line canonical; butterfly<->3line is dimension "
              "coincidence (phi(jets)=0, no canonical x^4 factor); divergence_§3_survives; "
              "operational_not_identity; D3_respected")
        return 0
    print("FUNCTOR_F_PETITOT_VERDICT B_INCONCLUSIVE")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
