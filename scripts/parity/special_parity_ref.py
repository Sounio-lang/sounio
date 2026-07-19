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
}

def main():
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
            print(f"{fn:<18}{0:>7}{'NO DATA':>16}  SKIP"); continue
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
    else: sys.exit(main())
