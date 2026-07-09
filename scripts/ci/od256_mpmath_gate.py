#!/usr/bin/env python3
"""od256 (oct-double, ~424-bit) reference + mpmath validation gate.

Executable spec for a software octuple-precision float built as an 8-component
non-overlapping double expansion. EFT primitives match stdlib/math/dd64.sio
(Knuth two_sum, Dekker split two_prod — no FMA). Accumulation uses Shewchuk
grow-expansion (exact), then truncation to the 8 most-significant limbs =
rounding to ~424 bits. This validates the DESIGN against mpmath ground truth.
"""
import math, random
from mpmath import mp, mpf, fabs, log

K = 8  # limbs

# --- EFT primitives (bit-identical to stdlib/math/dd64.sio) ---
def two_sum(a, b):
    s = a + b; bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e

def split(a):                      # Dekker split (dd_split in dd64.sio)
    c = 134217729.0 * a            # 2^27 + 1
    hi = c - (c - a); lo = a - hi
    return hi, lo

def two_prod(a, b):
    p = a * b
    ah, al = split(a); bh, bl = split(b)
    e = ((ah*bh - p) + ah*bl + al*bh) + al*bl
    return p, e

# --- Shewchuk grow-expansion: h = e (increasing, nonoverlapping) + scalar b, exact ---
def grow(e, b):
    Q = b; h = []
    for ei in e:
        Q, q = two_sum(Q, ei)
        if q != 0.0: h.append(q)
    if Q != 0.0: h.append(Q)
    return h

def expansion_sum(terms):
    e = []
    for t in terms:
        if t != 0.0: e = grow(e, t)
    return e  # increasing magnitude, nonoverlapping, EXACT sum of terms

def renorm_to_K(terms):
    """Exact-accumulate (Shewchuk) then keep the K most-significant limbs.

    expansion_sum returns an EXACT non-overlapping expansion (increasing
    magnitude); the K most-significant limbs are the octuple result, and the
    discarded tail IS the rounding error (~2^-424 relative). Returns K limbs
    most-significant-first, zero-padded.
    """
    e = expansion_sum(terms)
    top = e[-K:][::-1]                    # most-significant first
    return (top + [0.0]*K)[:K]

# --- od256 ops ---
def od_from_float(x): return [x] + [0.0]*(K-1)
def od_add(a, b):     return renorm_to_K(a + b)
def od_sub(a, b):     return renorm_to_K(a + [-x for x in b])
def od_mul(a, b):
    terms = []
    for ai in a:
        if ai == 0.0: continue
        for bj in b:
            if bj == 0.0: continue
            p, e = two_prod(ai, bj); terms.append(p); terms.append(e)
    return renorm_to_K(terms)

def od_recip(b):
    """1/b via Newton: y_{k+1} = y_k + y_k*(1 - b*y_k). 53->106->212->424->848 bits."""
    y = od_from_float(1.0 / b[0])
    for _ in range(4):                       # 4 iters comfortably past 424 bits
        r = od_sub(od_from_float(1.0), od_mul(b, y))
        y = od_add(y, od_mul(y, r))
    return y

def od_div(a, b):
    return od_mul(a, od_recip(b))

def od_sqrt(a):
    """sqrt(a)=a*(1/sqrt(a)); inv-sqrt Newton: y += 0.5*y*(1 - a*y*y)."""
    if a[0] == 0.0: return od_from_float(0.0)
    y = od_from_float(1.0 / math.sqrt(a[0]))
    half = od_from_float(0.5)
    for _ in range(4):
        r = od_sub(od_from_float(1.0), od_mul(a, od_mul(y, y)))
        y = od_add(y, od_mul(od_mul(half, y), r))
    return od_mul(a, y)

def od_to_mpf(a):
    mp.prec = 700
    return sum((mpf(x) for x in a), mpf(0))

def od_from_mpf(v):
    """Round a high-precision value into an 8-double expansion."""
    a = []
    for _ in range(K):
        d = float(v)               # nearest double
        a.append(d); v = v - mpf(d)
    return renorm_to_K(a)

# --- validation gate vs mpmath ---
def rel_bits(approx_mpf, true_mpf):
    mp.prec = 700
    if true_mpf == 0: return 1e9 if approx_mpf == 0 else 0.0
    r = fabs((approx_mpf - true_mpf) / true_mpf)
    return 1e9 if r == 0 else float(-log(r, 2))

def gate(n=4000, seed=20260708):
    mp.prec = 700
    rng = random.Random(seed)
    worst = {"add": 1e9, "sub": 1e9, "mul": 1e9, "div": 1e9, "sqrt": 1e9}
    for _ in range(n):
        # random ~424-bit operands
        va = mpf(rng.uniform(-1, 1)) * mpf(2)**rng.randint(-40, 40)
        vb = mpf(rng.uniform(-1, 1)) * mpf(2)**rng.randint(-40, 40)
        va += mpf(rng.getrandbits(400)) / mpf(2)**(rng.randint(430, 470))
        vb += mpf(rng.getrandbits(400)) / mpf(2)**(rng.randint(430, 470))
        a, b = od_from_mpf(va), od_from_mpf(vb)
        ta, tb = od_to_mpf(a), od_to_mpf(b)   # true values of the rounded operands
        worst["add"] = min(worst["add"], rel_bits(od_to_mpf(od_add(a, b)), ta + tb))
        worst["sub"] = min(worst["sub"], rel_bits(od_to_mpf(od_sub(a, b)), ta - tb))
        worst["mul"] = min(worst["mul"], rel_bits(od_to_mpf(od_mul(a, b)), ta * tb))
        if tb != 0:
            worst["div"] = min(worst["div"], rel_bits(od_to_mpf(od_div(a, b)), ta / tb))
        aa = od_abs(a) if False else [abs(x) for x in a]   # sqrt on |a|
        aa = od_from_mpf(fabs(ta))
        worst["sqrt"] = min(worst["sqrt"], rel_bits(od_to_mpf(od_sqrt(aa)), mp.sqrt(fabs(ta))))
    return worst

if __name__ == "__main__":
    # sanity: round-trip a known constant to ~424 bits
    mp.prec = 700
    pi_od = od_from_mpf(mp.pi)
    print(f"round-trip mp.pi  -> effective bits = {rel_bits(od_to_mpf(pi_od), mp.pi):.1f}  (target ~424)")
    w = gate()
    print("worst-case effective bits over 4000 random trials:")
    for op, b in w.items():
        print(f"  {op}: {b:.1f} bits  ({b/ math.log2(10):.1f} decimal digits)")
    ok = (w["add"] >= 400 and w["sub"] >= 400 and w["mul"] >= 390
          and w["div"] >= 380 and w["sqrt"] >= 380)
    print("GATE:", "PASS" if ok else "FAIL", "(add/sub>=400, mul>=390, div/sqrt>=380 bits)")
    raise SystemExit(0 if ok else 1)
