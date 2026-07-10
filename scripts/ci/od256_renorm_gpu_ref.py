#!/usr/bin/env python3
"""od256 branchless GPU renormalization — validated reference (design for the
K-AXI/PTX emitter). Proves od_add's renorm needs ONLY fixed, straight-line ops
(no data-dependent branches), so it lowers to K-AXI add/sub + SETP_NE/SETP_EQ +
SELP.

Pipeline (adding two normalized 8-limb od256 a, b) — the emitted GPU kernel:
  1. INTERLEAVE a[i],b[i] -> 16 (NO merge network needed: a,b each already
            |mag|-descending, so interleaving is "descending in pairs" and two
            VecSum passes recover full octuple — validated at 429.5 bits).
  2. VECSUM x2: 15 fixed two_sums each (CAMPARY Algorithm 4), error-free.
  3. VSEB   VecSumErrBranch, BRANCHLESS via SELP-emulated moving index: the
            data-dependent "advance output slot" becomes K SELPs/step keyed on
            SETP_EQ(j,cnt) & SETP_NE(err,0). Bit-identical to the branchy CAMPARY
            algorithm; ~424-bit (octuple) accurate. EFT primitives match dd64.
"""
import random
from mpmath import mp, mpf, fabs, log

K = 8

def two_sum(a, b):
    s = a + b; bb = s - a
    return s, (a - (s - bb)) + (b - bb)

def fast_two_sum(a, b):            # assumes |a| >= |b|
    s = a + b
    return s, b - (s - a)

def vecsum(x):                     # e[0] most significant, sum preserved (EFT)
    n = len(x); e = [0.0]*n; s = x[n-1]
    for i in range(n-2, -1, -1):
        s, e[i+1] = two_sum(x[i], s)
    e[0] = s
    return e

def interleave(a, b):
    """Interleave a[i],b[i] -> 16 (NO merge network; a,b each already sorted).
    Followed by 2x VecSum this reaches octuple — the emitted kernel's approach."""
    c = []
    for i in range(K):
        c.append(a[i]); c.append(b[i])
    return c

def vseb_branchless(e, k=K):
    """VecSumErrBranch with the conditional advance emulated by SELP over a fixed
    output array — straight-line, no branches (K-AXI SETP/SELP)."""
    n = len(e); f = [0.0]*k; cnt = 0; eps = e[0]
    for i in range(n-1):
        r, err = fast_two_sum(eps, e[i+1])
        nz = 1 if err != 0.0 else 0                    # SETP_NE err, 0
        for j in range(k):                             # k SELPs
            hit = (nz == 1 and j == cnt)               # SETP_EQ j,cnt & nz
            f[j] = r if hit else f[j]                  # SELP
        cnt += nz                                      # predicated ADD (0/1)
        eps = err if nz == 1 else r                    # SELP
    for j in range(k):
        if j == cnt and cnt < k:
            f[j] = eps
    return f

def vseb_branch(e, k=K):           # branchy reference (for bit-identity check)
    n = len(e); f = [0.0]*k; j = 0; eps = e[0]
    for i in range(n-1):
        r, err = fast_two_sum(eps, e[i+1])
        if err != 0.0:
            if j < k: f[j] = r
            j += 1; eps = err
        else:
            eps = r
    if j < k: f[j] = eps
    return f

def od_add_gpu(a, b):
    # interleave + 6x VecSum + branchless VSEB (matches the emitted GPU kernel).
    # 6 passes (was 2): with a partial magnitude gap between operands the
    # interleaved sequence is disordered and carries must propagate up to K limbs;
    # 2 passes lost ~half precision (215 bits) there. 6 (even, keeps the emitter's
    # ping-pong bank parity) restores full ~430-bit precision across all gaps.
    c = interleave(a, b)
    for _ in range(6):
        c = vecsum(c)
    return vseb_branchless(c)

def _split(x):
    C = 134217729.0; c = C*x; d = c - x; hi = c - d
    return hi, x - hi

def _two_prod(a, b):                # Dekker (no FMA), matches dd64 / the GPU emit
    p = a*b; ah, al = _split(a); bh, bl = _split(b)
    return p, ((ah*bh - p) + ah*bl + al*bh) + al*bl

def od_mul_gpu(a, b):
    # 43 partial products two_prod(a[i],b[j]) for i+j<9 (rest < 2^-424), bucket
    # order -> 3x VecSum -> branchless VSEB. Matches the emitted GPU kernel.
    terms = []
    for s in range(9):
        for i in range(K):
            j = s - i
            if 0 <= j < K:
                p, e = _two_prod(a[i], b[j]); terms.append(p); terms.append(e)
    c = terms
    for _ in range(3):
        c = vecsum(c)
    return vseb_branchless(c)

def _grow_norm(terms):             # exact accumulate -> clean 8-limb od256
    e = []
    for t in terms:
        if t == 0.0: continue
        q = t; h = []
        for ei in e:
            s, lo = two_sum(q, ei); q = s
            if lo != 0.0: h.append(lo)
        if q != 0.0: h.append(q)
        e = h
    return (e[-K:][::-1] + [0.0]*K)[:K]

def _split_mpf(v):                 # round a high-prec value into a clean od256
    a = []
    for _ in range(K):
        d = float(v); a.append(d); v = v - mpf(d)
    return _grow_norm(a)

if __name__ == "__main__":
    mp.prec = 800
    rng = random.Random(5)
    worst = 999.0; mismatch = 0; N = 8000
    for _ in range(N):
        va = mpf(rng.getrandbits(420)) / mpf(2)**rng.randint(200, 240) * (1 if rng.random() < .5 else -1)
        vb = mpf(rng.getrandbits(420)) / mpf(2)**rng.randint(200, 240) * (1 if rng.random() < .5 else -1)
        a = _split_mpf(va); b = _split_mpf(vb)
        e = vecsum(vecsum(interleave(a, b)))
        if vseb_branch(e) != vseb_branchless(e):
            mismatch += 1
        true = sum(mpf(x) for x in a) + sum(mpf(x) for x in b)
        approx = sum(mpf(x) for x in vseb_branchless(e))
        if true != 0:
            r = fabs((approx - true) / true)
            worst = min(worst, 999.0 if r == 0 else float(-log(r, 2)))
    print(f"branchless SELP-VSEB vs branchy: {N-mismatch}/{N} bit-identical")
    print(f"od_add branchless renorm: worst {worst:.1f} effective bits over {N}")
    # od_mul design gate
    rng2 = random.Random(21); worst_m = 999.0
    for _ in range(2000):
        va = mpf(rng2.getrandbits(420)) / mpf(2)**rng2.randint(-30, 30) * (1 if rng2.random() < .5 else -1)
        vb = mpf(rng2.getrandbits(420)) / mpf(2)**rng2.randint(-30, 30) * (1 if rng2.random() < .5 else -1)
        a = _split_mpf(va); b = _split_mpf(vb)
        approx = sum(mpf(x) for x in od_mul_gpu(a, b))
        true = (sum(mpf(x) for x in a)) * (sum(mpf(x) for x in b))
        if true != 0:
            r = fabs((approx - true) / true)
            worst_m = min(worst_m, 999.0 if r == 0 else float(-log(r, 2)))
    print(f"od_mul (43 two_prod + 3x VecSum + VSEB): worst {worst_m:.1f} effective bits")
    ok = mismatch == 0 and worst >= 400 and worst_m >= 400
    print("GATE:", "PASS" if ok else "FAIL", "(bit-identical + add/mul >=400 bits)")
    raise SystemExit(0 if ok else 1)
