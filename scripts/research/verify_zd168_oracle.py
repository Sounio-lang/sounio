#!/usr/bin/env python3
"""Independent (non-souc) oracle for the sedenion zero-divisor census.

Transcribed DIRECTLY from the Lean spec (formal/lean4/SounioCayleyDickson.lean cdSigma +
SounioZeroDivisorBridge.lean primProd/isZeroPair/validPrims/orderedZDPairs/unorderedZDPairs),
NOT from the Sounio .sio engine. Purpose: cross-toolchain replication of 84 -> 336 -> 168 so a
souc-specific false-green / miscompile cannot masquerade as a proof of execution.

Emits the 168 canonical unordered pairs as data on stdout (one per line, `PAIR ulo uhi uneg vlo
vhi vneg`) plus the three counts, for element-wise diff against the souc-emitted list.
"""
import sys

def cdSigma(a, b, bits):
    # formal/lean4/SounioCayleyDickson.lean:57-73
    if a == 0 or b == 0:
        return 1
    if bits <= 1:
        return -1
    half = 1 << (bits - 1)
    aHi = a >= half
    bHi = b >= half
    aLo = a & (half - 1)
    bLo = b & (half - 1)
    if (not aHi) and (not bHi):
        return cdSigma(aLo, bLo, bits - 1)
    elif (not aHi) and bHi:
        return cdSigma(bLo, aLo, bits - 1)
    elif aHi and (not bHi):
        return cdSigma(aLo, bLo, bits - 1) if bLo == 0 else -(cdSigma(aLo, bLo, bits - 1))
    else:  # both hi
        return -(cdSigma(bLo, aLo, bits - 1)) if bLo == 0 else cdSigma(bLo, aLo, bits - 1)

def sedSigma(a, b):
    return cdSigma(a, b, 4)

def primProd(u, v, k):
    # SounioZeroDivisorBridge.lean:94-102
    ulo, uhi, uneg = u
    vlo, vhi, vneg = v
    s_u = -1 if uneg else 1
    s_v = -1 if vneg else 1
    c_ll = sedSigma(ulo, vlo) if (ulo ^ vlo) == k else 0
    c_lh = s_v * sedSigma(ulo, vhi) if (ulo ^ vhi) == k else 0
    c_hl = s_u * sedSigma(uhi, vlo) if (uhi ^ vlo) == k else 0
    c_hh = s_u * s_v * sedSigma(uhi, vhi) if (uhi ^ vhi) == k else 0
    return c_ll + c_lh + c_hl + c_hh

def isZeroPair(u, v):
    return all(primProd(u, v, k) == 0 for k in range(16))

def isPrimValid(v):
    lo, hi, _ = v
    return 1 <= lo <= 7 and 9 <= hi <= 15 and (lo ^ hi) != 8

def primLt(u, v):
    ulo, uhi, uneg = u
    vlo, vhi, vneg = v
    return (ulo < vlo) or (ulo == vlo and uhi < vhi) or \
           (ulo == vlo and uhi == vhi and (not uneg) and vneg)

def compute():
    allPrims = [(lo, hi, neg) for lo in range(1, 8) for hi in range(9, 16) for neg in (False, True)]
    validPrims = [v for v in allPrims if isPrimValid(v)]
    ordered = [(u, v) for u in validPrims for v in validPrims if u != v and isZeroPair(u, v)]
    unordered = [(u, v) for (u, v) in ordered if primLt(u, v)]
    return validPrims, ordered, unordered

# ---- The OTHER 168: non-Fano (non-associative) octonion triples ----
# SounioCayleyDickson.lean: octSigma = cdSigma(.,.,3); alphaSign = σ(i,j)·σ(i⊕j,k);
# betaSign = σ(j,k)·σ(i,j⊕k); non-Fano iff alpha != beta over {1..7}^3. Proven = 168.
def octSigma(a, b):
    return cdSigma(a, b, 3)

def alphaSign(i, j, k):
    return octSigma(i, j) * octSigma(i ^ j, k)

def betaSign(i, j, k):
    return octSigma(j, k) * octSigma(i, j ^ k)

def isFano(i, j, k):
    return alphaSign(i, j, k) == betaSign(i, j, k)

def waveFunc(i, j, k):
    return alphaSign(i, j, k) - betaSign(i, j, k)

def nonfano_triples():
    return [(i, j, k) for i in range(1, 8) for j in range(1, 8) for k in range(1, 8)
            if not isFano(i, j, k)]

def dagger_arrows():
    # 84<->84 duality: dag(i,j,k)=(k,j,i) negates the wave. Emit each forward (wave=+2) triple
    # with its dagger (backward) image: (i,j,k, k,j,i). Free involution on the 168 non-Fano triples.
    return [(i, j, k, k, j, i) for i in range(1, 8) for j in range(1, 8) for k in range(1, 8)
            if waveFunc(i, j, k) == 2]

def pair_key(pair):
    (ulo, uhi, uneg), (vlo, vhi, vneg) = pair
    return (ulo, uhi, int(uneg), vlo, vhi, int(vneg))

from fractions import Fraction  # unbounded exact rationals -> overflow-free reference for souc i64

def measure_case(samples):
    # samples: list of (alpha, beta, gamma, delta) as Fractions. F = r5 = alpha*gamma + beta*delta.
    N = len(samples)
    rs = [a * g + b * d for (a, b, g, d) in samples]
    E = sum(rs, Fraction(0)) / N
    Var = sum((r * r for r in rs), Fraction(0)) / N - E * E
    return E, Var

if __name__ == "__main__":
    validPrims, ordered, unordered = compute()
    print(f"COUNT validPrims {len(validPrims)}")
    print(f"COUNT ordered {len(ordered)}")
    print(f"COUNT unordered {len(unordered)}")
    for p in sorted(pair_key(p) for p in unordered):
        print("PAIR " + " ".join(str(x) for x in p))
    # The OTHER 168: non-Fano octonion triples (the bridge equates the two 168s).
    nf = nonfano_triples()
    print(f"COUNT nonfano {len(nf)}")
    for t in sorted(nf):
        print("TRIPLE " + " ".join(str(x) for x in t))
    # The 84<->84 dagger bijection (explicit map, not just counts).
    arrows = dagger_arrows()
    print(f"COUNT arrows {len(arrows)}")
    for a in sorted(arrows):
        print("ARROW " + " ".join(str(x) for x in a))
    # Measure-layer (Frente A): exact E/Var over Q, same sample points as the Sounio test.
    F = Fraction
    onloc = [(F(1), F(1), F(1), F(-1)), (F(2), F(2), F(1), F(-1)), (F(1), F(1), F(3), F(-3))]
    offloc = [(F(9, 10), F(1), F(1), F(-1)), (F(1), F(1), F(1), F(-1)), (F(11, 10), F(1), F(1), F(-1))]
    for tag, samples in ((0, onloc), (1, offloc)):
        E, Var = measure_case(samples)
        print(f"MEASURE {tag} E {E.numerator} {E.denominator} VAR {Var.numerator} {Var.denominator}")
    # Generalized sweep (unbounded exact): Var at scale k = 2/(3*10^(2k)). Python fractions computes
    # every k, INCLUDING past the i64 wall (k>=10) where a bounded engine must censor -> locates the boundary.
    I64_MAX = 9223372036854775807
    for k in range(1, 13):
        V = F(2, 3 * 10 ** (2 * k))
        fits = "FITS" if V.denominator <= I64_MAX else "BIGINT"
        print(f"SWEEP {k} {V.numerator} {V.denominator} {fits}")
    # Unbounded exact reference for the souc BIGINT sweep (k=1..20, well past the i64 wall).
    for k in range(1, 21):
        V = F(2, 3 * 10 ** (2 * k))
        print(f"BIG {k} {V.numerator} {V.denominator}")
    # self-check against the Lean-proven counts + the exact measure values
    E0, V0 = measure_case(onloc)
    E1, V1 = measure_case(offloc)
    ok = (len(validPrims) == 84 and len(ordered) == 336 and len(unordered) == 168
          and len(nf) == 168 and len(arrows) == 84
          and E0 == 0 and V0 == 0 and E1 == 0 and V1 == Fraction(1, 150))
    print("ORACLE " + ("OK zd=84/336/168 nonfano=168 arrows=84 measure=0/0,0/(1/150)" if ok else "MISMATCH"))
    sys.exit(0 if ok else 1)
