#!/usr/bin/env python3
"""Oracle: unbounded-Q Cayley-Dickson product, ALL 16 components, via the common-denominator reduction
(16 integer numerators over one common denom -> integer 16-comp product; den = Ad*Bd). Coefficients here
are UNBOUNDED (numerators ~1e40-1e80) -- far beyond i64 -- so this is the exact-Q CD at arbitrary
precision. Raw (unreduced) numerators + den emitted so the souc minimal-BigInt leg can match exactly
without needing gcd. See sedenion_cd_qbig.md."""


def cd_sigma(a, b, bits=4):
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


def cd16(An, Bn):
    out = [0] * 16
    for i in range(16):
        if An[i] == 0:
            continue
        for j in range(16):
            if Bn[j] == 0:
                continue
            out[i ^ j] += cd_sigma(i, j) * An[i] * Bn[j]
    return out


# Case 1: a known ZD pair scaled by 10^40 -> exact annihilation, den 1. (e3+e10)(e6-e15).
S = 10 ** 40
A1 = [0] * 16; A1[3] = S; A1[10] = S
B1 = [0] * 16; B1[6] = S; B1[15] = -S
# Case 2: general 16-comp product, unbounded rational coeffs over common denoms.
A2 = [0] * 16; A2[1] = 10 ** 30; A2[2] = 3 * 10 ** 28; A2[4] = -5 * 10 ** 35; A2[8] = 10 ** 33; A2[15] = 7 * 10 ** 20
Ad2 = 6
B2 = [0] * 16; B2[1] = 2 * 10 ** 31; B2[5] = -10 ** 29; B2[7] = 4 * 10 ** 27; B2[10] = 9 * 10 ** 34
Bd2 = 10

cases = [("1", A1, 1, B1, 1), ("2", A2, Ad2, B2, Bd2)]
P = 1000000007
allzero1 = True
for (tag, An, Ad, Bn, Bd) in cases:
    out = cd16(An, Bn)
    den = Ad * Bd
    for k in range(16):
        print(f"EXACT {tag} {k} {out[k]}")      # exact decimal (record; also witnessed by Lean)
        print(f"RES {out[k] % P}")              # residue mod 1e9+7 (gated vs souc, same order)
        if tag == "1" and out[k] != 0:
            allzero1 = False
    print(f"EXACT {tag} DEN {den}")
    print(f"RES {den % P}")
print(f"ANNIHILATION_C1 {1 if allzero1 else 0}")
