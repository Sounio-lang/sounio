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
    # self-check against the Lean-proven counts
    ok = (len(validPrims) == 84 and len(ordered) == 336 and len(unordered) == 168
          and len(nf) == 168 and len(arrows) == 84)
    print("ORACLE " + ("OK zd=84/336/168 nonfano=168 arrows=84" if ok else "MISMATCH"))
    sys.exit(0 if ok else 1)
