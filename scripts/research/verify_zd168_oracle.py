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
    # self-check against the Lean-proven counts
    ok = (len(validPrims) == 84 and len(ordered) == 336 and len(unordered) == 168)
    print("ORACLE " + ("OK 84/336/168" if ok else "MISMATCH"))
    sys.exit(0 if ok else 1)
