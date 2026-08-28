#!/usr/bin/env python3
"""Compare Sounio special-function emitter output against mpmath (dps=30).
Bit-exact wire on stdin: `<fn> <nargs> <arg_bits...> <val_bits>`, each field a
signed-i64 IEEE-754 bit pattern (Sounio f64_to_bits). No scipy dependency."""
import sys, struct, mpmath as mp
mp.mp.dps = 30

def _ibetainv(a, b, p):
    lo, hi = mp.mpf(0), mp.mpf(1)
    for _ in range(200):
        mid = (lo + hi) / 2
        if mp.betainc(a, b, 0, mid, regularized=True) < p: lo = mid
        else: hi = mid
    return (lo + hi) / 2

def _jacobi_explicit(n, a, b, x):
    # Explicit finite-sum formula (DLMF 18.5.7), independent of mpmath's
    # mp.jacobi(): mp.jacobi's hypergeometric-series representation spuriously
    # fails to converge for several alpha==beta test points here (a genuine
    # mpmath convergence-detection quirk, not a property of the polynomial).
    n = int(n)
    def binom(nn, kk):
        # Generalized binomial via the rising-product form (kk a small
        # non-negative int here), not gamma ratios: gamma(x+1)/gamma(y+1) for
        # half-integer x,y goes through sqrt(pi) that only cancels
        # symbolically, leaving ~1e-32 floating residue at dps=30 — enough to
        # turn an exact zero (odd Jacobi poly at x=0 with alpha==beta) into a
        # spurious "1.0 relative error" against Sounio's exact 0.0.
        kk = int(kk)
        num = mp.mpf(1)
        for i in range(kk):
            num *= (nn - i)
        den = mp.mpf(1)
        for i in range(1, kk + 1):
            den *= i
        return num / den
    s = mp.mpf(0)
    for k in range(n + 1):
        s += binom(n + a, n - k) * binom(n + b, k) * ((x - 1) / 2) ** k * ((x + 1) / 2) ** (n - k)
    return s

def bits_to_f64(field):
    return struct.unpack('<d', struct.pack('<q', int(field)))[0]

# fn -> (callable(*args)->mpf, gross_threshold). Thresholds calibrated later;
# default gross bar 1e-2 (fail loudly only on that).
REF = {
    "erf":            (lambda x: mp.erf(x), 1e-2),
    "erfc":           (lambda x: mp.erfc(x), 1e-2),
    "erfinv":         (lambda x: mp.erfinv(x), 1e-2),
    "normal_cdf":     (lambda x: mp.ncdf(x), 1e-2),
    "normal_quantile":(lambda p: mp.sqrt(2)*mp.erfinv(2*p-1), 1e-2),
    "gamma":  (lambda x: mp.gamma(x), 1e-2),
    "lgamma": (lambda x: mp.loggamma(x), 1e-2),
    "digamma":(lambda x: mp.digamma(x), 1e-2),
    "beta":     (lambda a,b: mp.beta(a,b), 1e-2),
    "lbeta":    (lambda a,b: mp.log(mp.beta(a,b)), 1e-2),
    "ibeta":    (lambda a,b,x: mp.betainc(a,b,0,x,regularized=True), 1e-2),
    "ibeta_inv":(lambda a,b,p: _ibetainv(a,b,p), 1e-2),
    "igamma_lower":(lambda a,x: mp.gammainc(a, 0, x, regularized=True), 1e-2),
    "igamma_upper":(lambda a,x: mp.gammainc(a, x, mp.inf, regularized=True), 1e-2),
    "chi2_cdf":    (lambda x,k: mp.gammainc(k/2, 0, x/2, regularized=True), 1e-2),
    "bessel_j0":(lambda x: mp.besselj(0,x), 1e-2),
    "bessel_j1":(lambda x: mp.besselj(1,x), 1e-2),
    "bessel_y0":(lambda x: mp.bessely(0,x), 1e-2),
    "bessel_y1":(lambda x: mp.bessely(1,x), 1e-2),
    "bessel_i0":(lambda x: mp.besseli(0,x), 1e-2),
    "bessel_i1":(lambda x: mp.besseli(1,x), 1e-2),
    "bessel_k0":(lambda x: mp.besselk(0,x), 1e-2),
    "bessel_k1":(lambda x: mp.besselk(1,x), 1e-2),
    "bessel_jn":(lambda n,x: mp.besselj(n,x), 1e-2),
    "bessel_yn":(lambda n,x: mp.bessely(n,x), 1e-2),
    "bessel_in":(lambda n,x: mp.besseli(n,x), 1e-2),
    "bessel_kn":(lambda n,x: mp.besselk(n,x), 1e-2),
    "airy_ai":      (lambda x: mp.airyai(x), 1e-2),
    "airy_bi":      (lambda x: mp.airybi(x), 1e-2),
    "airy_ai_prime":(lambda x: mp.airyai(x, 1), 1e-2),
    "airy_bi_prime":(lambda x: mp.airybi(x, 1), 1e-2),
    "riemann_zeta": (lambda s: mp.zeta(s), 1e-2),
    "hurwitz_zeta": (lambda s,a: mp.zeta(s, a), 1e-2),
    "dirichlet_eta":(lambda s: mp.altzeta(s), 1e-2),
    "elliptic_k":   (lambda k: mp.ellipk(k**2), 1e-2),
    "elliptic_e":   (lambda k: mp.ellipe(k**2), 1e-2),
    "hyperg_0f1":(lambda b,z: mp.hyp0f1(b,z), 1e-2),
    "hyperg_1f1":(lambda a,b,z: mp.hyp1f1(a,b,z), 1e-2),
    "hyperg_u":  (lambda a,b,z: mp.hyperu(a,b,z), 1e-2),
    "hyperg_2f1":(lambda a,b,c,z: mp.hyp2f1(a,b,c,z), 1e-2),
    "legendre_p":  (lambda n,x: mp.legendre(n, x), 1e-2),
    "chebyshev_t": (lambda n,x: mp.chebyt(n, x), 1e-2),
    "chebyshev_u": (lambda n,x: mp.chebyu(n, x), 1e-2),
    "hermite_h":   (lambda n,x: mp.hermite(n, x), 1e-2),
    "hermite_he":  (lambda n,x: 2**(-n/2) * mp.hermite(n, x/mp.sqrt(2)), 1e-2),
    "laguerre_l":  (lambda n,x: mp.laguerre(n, 0, x), 1e-2),
    "laguerre_l_assoc":(lambda n,a,x: mp.laguerre(n, a, x), 1e-2),
    "jacobi_p":    (lambda n,a,b,x: _jacobi_explicit(n, a, b, x), 1e-2),
}

def main(require_all=False):
    # NOTE: tokenized over the whole stream, not split per-line. The Sounio
    # `print_int` builtin emits a trailing newline after every integer it
    # prints (confirmed empirically: `print_int(a); print(" "); print_int(b)`
    # renders as "a\n b\n", not "a b"), so a single logical record like
    # `erf 1 <a_bits> <val_bits>` lands across multiple physical lines.
    # Whitespace (space or newline) is not semantically meaningful in this
    # wire format, so tokenizing across line breaks is the correct fix, not
    # a tolerance change.
    rows = {}
    tokens = [t for t in sys.stdin.read().split() if not t.startswith("#")]
    idx = 0
    n = len(tokens)
    while idx < n:
        fn = tokens[idx]
        if fn not in REF:
            idx += 1
            continue
        nargs = int(tokens[idx + 1])
        args = [bits_to_f64(tokens[idx + 2 + i]) for i in range(nargs)]
        value = bits_to_f64(tokens[idx + 2 + nargs])
        idx += 2 + nargs + 1
        ref = float(REF[fn][0](*[mp.mpf(a) for a in args]))
        rel = abs(value - ref) / max(abs(ref), 1e-300)
        rows.setdefault(fn, []).append((args, value, ref, rel))
    fail = 0
    print(f"{'function':<18}{'points':>7}{'max_rel_err':>16}  verdict")
    for fn in REF:
        pts = rows.get(fn, [])
        if not pts:
            # NO-DATA is SKIP for partial/self-test runs, but a coverage FAIL
            # under --require-all (a full gate run) so a missing/broken emitter
            # can't silently drop a function from the map.
            print(f"{fn:<18}{0:>7}{'NO DATA':>16}  {'FAIL(no-data)' if require_all else 'SKIP'}")
            if require_all: fail = 1
            continue
        worst = max(pts, key=lambda r: r[3]); mre = worst[3]; thr = REF[fn][1]
        ok = mre <= thr
        print(f"{fn:<18}{len(pts):>7}{mre:>16.3e}  {'PASS' if ok else 'FAIL(thr=%.0e)'%thr}")
        if not ok: fail = 1
    print("SPECIAL_SCIPY_PARITY_OK" if not fail else "SPECIAL_SCIPY_PARITY_FAIL")
    return fail

def _bits(x):
    return struct.unpack('<q', struct.pack('<d', x))[0]

def selftest():
    val = float(mp.erf(mp.mpf('0.5')))
    line = f"erf 1 {_bits(0.5)} {_bits(val)}\n"
    import io
    sys.stdin = io.StringIO(line)
    rc = main()
    assert rc == 0, "selftest: erf(0.5) should pass"
    print("REF_SELFTEST_OK")

if __name__ == "__main__":
    if "--selftest" in sys.argv: selftest()
    else: sys.exit(main(require_all=("--require-all" in sys.argv)))
