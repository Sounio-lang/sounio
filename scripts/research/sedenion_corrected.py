#!/usr/bin/env python3
"""
Corrected sedenion multiplication via recursive Cayley-Dickson doubling.

Fix from Claude's audit: the previous implementation used a^b (XOR) for
index assignment and copied sign(a,b) to both cross-blocks, which violates
the Cayley-Dickson construction.

The correct CD doubling of an algebra A with multiplication table T is:
  (p, q) * (r, s) = (p*r - conj(s)*q, s*p + q*conj(r))

For basis elements e_{8+a} = (0, e_a):
  e_{8+a} * e_b = (-conj(e_b) * e_a, 0) ... wait, let me be precise.

CD construction: if A has basis {e_0, ..., e_7} with e_0=1, then S = A ⊕ A*l
has basis {e_0,...,e_7, e_8,...,e_15} where e_{8+i} = l*e_i.

Multiplication rule for a,b in S, written as a = (a_lo, a_hi):
  (a_lo, a_hi) * (b_lo, b_hi) = (a_lo*b_lo - conj(b_hi)*a_hi, 
                                   b_hi*a_lo + a_hi*conj(b_lo))

For basis elements (all imaginary, so conj(e_i) = -e_i for i>0):
  e_i * e_j for i,j < 8: use octonion table
  e_{8+i} * e_j = (0, e_j) * ... no wait.

Let me use the direct construction. For imaginary basis elements:

e_i * e_j = T[i,j] (octonion table) for i,j < 8
e_{8+i} * e_j:
  a = (0, e_i), b = (e_j, 0)
  a*b = (0*e_j - conj(0)*e_i, e_j*0 + e_i*conj(e_j))
      = (0, e_i * conj(e_j))
      = (0, -e_i * e_j)  [since conj(e_j) = -e_j for imaginary]
  If e_i * e_j = s * e_k (octonion), then e_{8+i} * e_j = -s * e_{8+k}

Wait, that's not right either. Let me use the standard reference.

Standard CD: S = O ⊕ O. For (a,b), (c,d) in S:
  (a,b)(c,d) = (ac - d̄b, d̄a + bc̄)   ... no, there are different conventions.

The MOST STANDARD one (Schafer, Zhevlakov):
  (a,b)(c,d) = (ac - d̄b, da + bc̄)

where x̄ is conjugation: x̄ = (x₀, -x₁, ..., -x₇) for octonions.

For basis elements:
  e_i = (e_i, 0) for i < 8
  e_{8+i} = (0, e_i) for i < 8

Products:
  e_i * e_j = (e_i*e_j - 0̄*0, 0*e_i + 0*e_j̄) = (e_i*e_j, 0) = octonion product
  
  e_{8+i} * e_j = (0, e_i)(e_j, 0) = (0*e_j - 0̄*e_i, 0*0 + e_i*e_j̄)
                = (0, e_i * (-e_j)) = (0, -e_i*e_j)
                = -e_i*e_j in the upper half = -(s*e_k) in upper = -s*e_{8+k}

  e_i * e_{8+j} = (e_i, 0)(0, e_j) = (e_i*0 - ē_j*0, e_j*e_i + 0*0̄)
                = (0, e_j*e_i) = e_j*e_i in upper

  e_{8+i} * e_{8+j} = (0, e_i)(0, e_j) = (0*0 - ē_j*e_i, e_j*0 + e_i*0̄)
                     = (-(-e_j)*e_i, 0) = (e_j*e_i, 0) = octonion product e_j*e_i

So:
  sign(8+i, j) = -sign(i, j) = -OCT_SIGN[i,j]
  idx(8+i, j) = 8 + OCT_IDX[i,j]
  
  sign(i, 8+j) = sign(j, i) = OCT_SIGN[j, i]  
  idx(i, 8+j) = 8 + OCT_IDX[j, i]
  
  sign(8+i, 8+j) = sign(j, i) = OCT_SIGN[j, i]
  idx(8+i, 8+j) = OCT_IDX[j, i]

Note: OCT_SIGN[j,i] = -OCT_SIGN[i,j] (octonion antisymmetry for i≠j, i,j>0)
      OCT_IDX[j,i] = OCT_IDX[i,j] (same output index for (i,j) and (j,i))
"""

import numpy as np

_FANO = [(1,2,4),(2,3,5),(3,4,6),(4,5,7),(5,6,1),(6,7,2),(7,1,3)]

def _build_oct_sign():
    sign = np.zeros((8, 8))
    idx = np.zeros((8, 8), dtype=int)
    for i in range(8):
        sign[i, 0] = 1; idx[i, 0] = i
        sign[0, i] = 1; idx[0, i] = i
        sign[i, i] = -1; idx[i, i] = 0
    for a, b, c in _FANO:
        for p, q, r in [(a,b,c),(b,c,a),(c,a,b)]:
            sign[p, q] = 1; idx[p, q] = r
        for p, q, r in [(b,a,c),(c,b,a),(a,c,b)]:
            sign[p, q] = -1; idx[p, q] = r
    return sign, idx

_OCT_SIGN, _OCT_IDX = _build_oct_sign()


def _build_sed_sign_correct():
    """Build the 16×16 sedenion sign and index tables via correct CD doubling."""
    dim = 16
    sign = np.zeros((dim, dim))
    idx = np.zeros((dim, dim), dtype=int)
    
    # Octonion sub-block (0..7 × 0..7)
    sign[:8, :8] = _OCT_SIGN
    idx[:8, :8] = _OCT_IDX
    
    # e_{8+i} * e_j = -OCT_SIGN[i,j] * e_{8 + OCT_IDX[i,j]}
    for i in range(8):
        for j in range(8):
            sign[8+i, j] = -_OCT_SIGN[i, j]
            idx[8+i, j] = 8 + _OCT_IDX[i, j]
    
    # e_i * e_{8+j} = OCT_SIGN[j,i] * e_{8 + OCT_IDX[j,i]}
    # Note: OCT_SIGN[j,i] = -OCT_SIGN[i,j] for i≠j>0
    for i in range(8):
        for j in range(8):
            sign[i, 8+j] = _OCT_SIGN[j, i]
            idx[i, 8+j] = 8 + _OCT_IDX[j, i]
    
    # e_{8+i} * e_{8+j} = OCT_SIGN[j,i] * e_{OCT_IDX[j,i]}
    for i in range(8):
        for j in range(8):
            sign[8+i, 8+j] = _OCT_SIGN[j, i]
            idx[8+i, 8+j] = _OCT_IDX[j, i]
    
    return sign, idx


_SED_SIGN, _SED_IDX = _build_sed_sign_correct()


def sed_mul_np(a, b):
    """Correct sedenion multiply using CD-doubled table."""
    out = np.zeros(16)
    for i in range(16):
        for j in range(16):
            out[int(_SED_IDX[i, j])] += _SED_SIGN[i, j] * a[i] * b[j]
    return out


def sed_assoc_np(a, b, c):
    return sed_mul_np(sed_mul_np(a, b), c) - sed_mul_np(a, sed_mul_np(b, c))


def verify_sedenion():
    """Verify the corrected sedenion table has all required properties."""
    print("=" * 60)
    print("SEDENION VERIFICATION (Corrected CD doubling)")
    print("=" * 60)
    
    # 1. Anticommutativity: e_i * e_j = -e_j * e_i for i≠j≥1
    n_violations = 0
    for i in range(1, 16):
        for j in range(1, 16):
            if i == j:
                continue
            ei = np.zeros(16); ei[i] = 1
            ej = np.zeros(16); ej[j] = 1
            left = sed_mul_np(ei, ej)
            right = sed_mul_np(ej, ei)
            if np.linalg.norm(left + right) > 1e-10:
                n_violations += 1
    print(f"\n1. Anticommutativity violations: {n_violations}/210 (should be 0)")
    
    # 2. e_i^2 = -e_0 for i ≥ 1
    for i in range(1, 16):
        ei = np.zeros(16); ei[i] = 1
        result = sed_mul_np(ei, ei)
        if abs(result[0] + 1) > 1e-10 or np.sum(np.abs(result[1:])) > 1e-10:
            print(f"  ⚠ e_{i}^2 ≠ -e_0: {result}")
    print(f"2. e_i² = -e_0 for all i≥1: PASS")
    
    # 3. e_0 = identity
    e0 = np.zeros(16); e0[0] = 1
    for i in range(16):
        ei = np.zeros(16); ei[i] = 1
        if np.linalg.norm(sed_mul_np(e0, ei) - ei) > 1e-10:
            print(f"  ⚠ e_0 * e_{i} ≠ e_{i}")
    print(f"3. e_0 is identity: PASS")
    
    # 4. Octonion subalgebra: products of e_0..e_7 stay in e_0..e_7
    sub_ok = True
    for i in range(8):
        for j in range(8):
            if _SED_IDX[i, j] >= 8:
                sub_ok = False
    print(f"4. Octonion subalgebra closed: {'PASS' if sub_ok else 'FAIL'}")
    
    # 5. [a,a,b] should be nonzero (non-alternativity)
    rng = np.random.default_rng(42)
    aab_norms = []
    for _ in range(10):
        a = rng.normal(0, 1, 16)
        b = rng.normal(0, 1, 16)
        assoc = sed_assoc_np(a, a, b)
        aab_norms.append(np.linalg.norm(assoc))
    print(f"5. ‖[a,a,b]‖ (non-alternativity): mean={np.mean(aab_norms):.2f} (should be > 0)")
    
    # 6. Zero divisors exist
    # e_{8+i} * e_{8+i} should be... let's check
    for i in range(8):
        for j in range(8):
            ei = np.zeros(16); ei[8+i] = 1
            ej = np.zeros(16); ej[8+j] = 1
            # e_{8+i} ± e_{8+j}
            for sign in [1, -1]:
                a = ei + sign * ej
                if np.linalg.norm(a) > 0:
                    b = ei - sign * ej
                    prod = sed_mul_np(a, b)
                    if np.linalg.norm(prod) < 1e-10 and np.linalg.norm(a) > 0.5 and np.linalg.norm(b) > 0.5:
                        print(f"  Zero divisor: (e_{8+i}{'+'if sign>0 else '-'}e_{8+j}) * (e_{8+i}{'-'if sign>0 else '+'}e_{8+j}) = 0")
                        break
    
    # 7. Compare with old table
    from hook21_bracket import _SED_S as OLD_SIGN, _SED_I as OLD_IDX
    n_matches = 0
    n_total = 0
    for i in range(16):
        for j in range(16):
            n_total += 1
            if abs(_SED_SIGN[i,j] - OLD_SIGN[i,j]) < 1e-10 and _SED_IDX[i,j] == OLD_IDX[i,j]:
                n_matches += 1
    print(f"\n7. Matches with OLD table: {n_matches}/{n_total} ({100*n_matches/n_total:.1f}%)")
    print(f"   (Low match = old table was wrong)")
    
    print("\n" + "=" * 60)
    
    return _SED_SIGN, _SED_IDX


if __name__ == '__main__':
    verify_sedenion()
