r"""
APPLICATION (part 2): the annihilation=associator bridge as a computable "zero-divisor risk map" for
hypercomplex (Cayley-Dickson) neural-network layers, organized by the frozen 168 = PSL(2,7).

MOTIVATION (from the 2026-07-11 applications recon).  Octonion-valued neural nets are an active area;
SEDENION-valued layers (dim 16) exist (a few papers + one US patent) but the field flags the zero
divisors as an unresolved problem: when a sedenion product hits the ZD locus it ANNIHILATES, destroying
information / killing gradients.  Nobody has a principled, symmetry-aware handle on WHICH directions are
dangerous.  Our proven bridge gives exactly that handle, for free.

THE HANDLE.  cd_tower_zd_annihilation_is_associator.py proves (forall n, verified n<=6):
    deg_annih(e_lo + s*e_hi) = #{ a : Psi(lo, a, hi_lo) = 1 }   (the middle-slot associator degree).
So the associator 3-form -- which we compute exactly -- IS a per-direction "how many partners annihilate
me" count = a computable GRADIENT-DEATH RISK for each mixed-half weight/activation direction.  Higher
degree = more annihilating partners = more ways a product collapses to zero.

SEDENION RESULT (dim 16, computed below).  The risk takes exactly TWO values:
    * 7 directions have risk 0  -- NO zero-divisor partners at all (the "safe" subspace).  They form a
      SINGLE 168-orbit of size 7 = the Fano plane PG(2,2) directions.
    * 42 directions have risk 4 -- maximal gradient-death risk.  A single 168-orbit of size 42
      -- exactly de Marrais's "42 Assessors" of the sedenion zero divisors.
  And -- the design-enabling fact -- **risk is CONSTANT on every 168-orbit** (equivariance: an
  automorphism is an isometry of the multiplication table, so it preserves the annihilation count).

DESIGN HYPOTHESES this enables (TESTABLE, not yet validated -- honest tag: these are proposals):
  (H1) ZD-avoidance prior: initialize / regularize sedenion-layer weights toward the 7 risk-0 directions
       (or penalize projection onto the 42 risk-4 directions) to reduce gradient collapse.
  (H2) 168-equivariant weight sharing: since risk (and all multiplication-table structure) is constant on
       168-orbits, tie/share parameters across an orbit -- a symmetry-respecting structured-sparsity
       prior with a built-in 168:1 (here 7- and 42-fold) parameter reduction, analogous to how CNNs share
       weights across a translation group but here across PSL(2,7).
  (H3) tower scaling: the orbit structure (2^{n-4} Fano orbits + growing fixed seams, PROVEN forall n)
       gives the SAME construction for higher CD layers (dim 32, 64, ...) with a frozen 168 symmetry --
       the parameter-sharing group does not grow as the layer widens.

STATUS -- HONEST.  The risk map and its 168-orbit invariance are COMPUTED FACTS (below, dim 16; the
bridge that justifies them is proven forall n / verified n<=6).  H1-H3 are DESIGN HYPOTHESES: plausible,
grounded in the proven structure, and concretely testable on the existing sedenion-NN benchmarks -- but
NOT experimentally validated here.  Do not cite them as demonstrated improvements; cite them as a
principled, symmetry-derived design space that the annihilation=associator identity opens up.
"""
from collections import Counter
import itertools


def cd_sigma(a, b, bits):
    if a == 0 or b == 0:
        return 1
    if bits <= 1:
        return -1
    half = 1 << (bits - 1)
    aH, bH, aL, bL = a >= half, b >= half, a & (half - 1), b & (half - 1)
    if not aH and not bH:
        return cd_sigma(aL, bL, bits - 1)
    if not aH and bH:
        return cd_sigma(bL, aL, bits - 1)
    if aH and not bH:
        return cd_sigma(aL, bL, bits - 1) if bL == 0 else -cd_sigma(aL, bL, bits - 1)
    return -cd_sigma(bL, aL, bits - 1) if bL == 0 else cd_sigma(bL, aL, bits - 1)


def f(i, j, m):
    return 0 if cd_sigma(i, j, m) == 1 else 1


def Psi(i, j, k, m):
    return f(i, j, m) ^ f(i ^ j, k, m) ^ f(j, k, m) ^ f(i, j ^ k, m)


def GL3():
    out = []
    for cols in itertools.permutations(range(1, 8), 3):
        span = {0}
        for b in cols:
            span = span | {x ^ b for x in span}
        if len(span) == 8:
            A = [0] * 8
            for v in range(8):
                acc = 0
                for bit, c in enumerate(cols):
                    if (v >> bit) & 1:
                        acc ^= c
                A[v] = acc
            out.append(A)
    return out


def main():
    m = 3
    H = 1 << m                                   # sedenions: octonion level m=3
    risk = {}
    for lo in range(1, H):
        for hilo in range(H):
            if lo == hilo:
                continue
            risk[(lo, hilo)] = sum(1 for a in range(1, H) if Psi(lo, a, hilo, m) == 1)
    print(f"Sedenion (dim 16) ZD-risk = annihilation degree = middle-slot associator degree per direction:")
    print(f"  risk histogram {{level: #directions}} = {dict(sorted(Counter(risk.values()).items()))}"
          f"   (0 = no ZD partners = safe; {max(risk.values())} = max gradient-death risk)")
    GL = GL3()
    seen = set()
    orbits = []
    bad = 0
    for d in risk:
        if d in seen:
            continue
        orb = set()
        frontier = [d]
        while frontier:
            lo, hilo = frontier.pop()
            if (lo, hilo) in orb:
                continue
            orb.add((lo, hilo))
            for A in GL:
                nd = (A[lo], A[hilo])
                if nd in risk and nd not in orb:
                    frontier.append(nd)
        seen |= orb
        orbits.append(orb)
        if len({risk[x] for x in orb}) > 1:
            bad += 1
    ok = (len(GL) == 168 and bad == 0)
    print(f"  frozen group |GL(3,2)| = {len(GL)}; #direction-orbits = {len(orbits)}; "
          f"risk constant on each orbit: {'YES' if bad == 0 else 'NO'}")
    print(f"  (orbit size, risk): {sorted((len(o), risk[next(iter(o))]) for o in orbits)} "
          f"-- 7 safe (Fano) + 42 risky (de Marrais 42 Assessors)")
    print("\nRISK MAP + 168-EQUIVARIANCE:", "computed (dim 16); enables testable design hypotheses "
          "H1 ZD-avoidance / H2 168-equivariant weight-sharing / H3 tower-scaling (NOT yet validated)."
          if ok else "MISMATCH")
    return ok


if __name__ == "__main__":
    main()
