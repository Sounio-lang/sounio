#!/usr/bin/env python3
"""stats distribution parity vs mpmath (dps=30). Bit-exact wire on stdin:
`<fn> <nargs> <arg_bits...> <val_bits>` (signed-i64 IEEE-754). No scipy."""
import sys, struct, math, mpmath as mp
mp.mp.dps = 30
def bits_to_f64(f): return struct.unpack('<d', struct.pack('<q', int(f)))[0]
def _bits(x): return struct.unpack('<q', struct.pack('<d', x))[0]
SQRT2PI = mp.sqrt(2*mp.pi)

def _norm_pdf(x, mu, s):  z=(x-mu)/s; return mp.e**(-z*z/2)/(s*SQRT2PI)
def _lognorm_pdf(x, mu, s):
    if x <= 0: return mp.mpf(0)
    z=(mp.log(x)-mu)/s; return mp.e**(-z*z/2)/(x*s*SQRT2PI)
def _gamma_pdf(x, sh, rate):
    if x <= 0: return mp.mpf(0)
    return rate**sh * x**(sh-1) * mp.e**(-rate*x) / mp.gamma(sh)
def _beta_pdf(x, a, b):
    if x <= 0 or x >= 1: return mp.mpf(0)
    return x**(a-1)*(1-x)**(b-1)/mp.beta(a,b)
def _uniform_pdf(x, a, b): return mp.mpf(1)/(b-a) if a <= x <= b else mp.mpf(0)
def _uniform_cdf(x, a, b): return mp.mpf(0) if x < a else (mp.mpf(1) if x > b else (x-a)/(b-a))
def _pois_pmf(k, lam): k=int(round(k)); return lam**k * mp.e**(-lam) / mp.factorial(k)
def _pois_cdf(k, lam): k=int(round(k)); return mp.gammainc(k+1, lam, mp.inf, regularized=True)
def _binom_pmf(k, n, p):
    k=int(round(k)); n=int(round(n)); return mp.binomial(n,k)*mp.mpf(p)**k*(1-mp.mpf(p))**(n-k)
def _binom_cdf(k, n, p):
    k=int(round(k)); n=int(round(n))
    if k >= n: return mp.mpf(1)
    return mp.betainc(n-k, k+1, 0, 1-mp.mpf(p), regularized=True)
def _geom_pmf(k, p): k=int(round(k)); return (1-mp.mpf(p))**k * p    # FROM-0

REF = {
    "normal_pdf":    (lambda x,mu,s: _norm_pdf(x,mu,s), 1e-2),
    "normal_cdf_at": (lambda x,mu,s: mp.ncdf((x-mu)/s), 1e-2),
    "standard_normal_cdf":         (lambda z: mp.ncdf(z), 1e-2),
    "inverse_standard_normal_cdf": (lambda p: mp.sqrt(2)*mp.erfinv(2*p-1), 1e-2),
    "exponential_pdf": (lambda x,l: l*mp.e**(-l*x) if x>=0 else mp.mpf(0), 1e-2),
    "exponential_cdf": (lambda x,l: 1-mp.e**(-l*x) if x>=0 else mp.mpf(0), 1e-2),
    "gamma_pdf": (lambda x,sh,r: _gamma_pdf(x,sh,r), 1e-2),
    "gamma_cdf": (lambda x,sh,r: mp.gammainc(sh,0,r*x,regularized=True) if x>0 else mp.mpf(0), 1e-2),
    "beta_pdf":  (lambda x,a,b: _beta_pdf(x,a,b), 1e-2),
    "beta_cdf":  (lambda x,a,b: mp.betainc(a,b,0,x,regularized=True), 1e-2),
    "lognormal_pdf": (lambda x,mu,s: _lognorm_pdf(x,mu,s), 1e-2),
    "lognormal_cdf": (lambda x,mu,s: mp.ncdf((mp.log(x)-mu)/s) if x>0 else mp.mpf(0), 1e-2),
    "uniform_pdf": (lambda x,a,b: _uniform_pdf(x,a,b), 1e-2),
    "uniform_cdf": (lambda x,a,b: _uniform_cdf(x,a,b), 1e-2),
    "poisson_pmf": (lambda k,l: _pois_pmf(k,l), 1e-2),
    "poisson_cdf": (lambda k,l: _pois_cdf(k,l), 1e-2),
    "binomial_pmf": (lambda k,n,p: _binom_pmf(k,n,p), 1e-2),
    "binomial_cdf": (lambda k,n,p: _binom_cdf(k,n,p), 1e-2),
    "geometric_pmf": (lambda k,p: _geom_pmf(k,p), 1e-2),
}

def main(require_all=False):
    rows = {}
    tokens = [t for t in sys.stdin.read().split() if not t.startswith("#")]
    i = 0; N = len(tokens)
    while i < N:
        fn = tokens[i]
        if fn not in REF: i += 1; continue
        nargs = int(tokens[i+1])
        args = [bits_to_f64(tokens[i+2+j]) for j in range(nargs)]
        value = bits_to_f64(tokens[i+2+nargs]); i += 2+nargs+1
        ref = float(REF[fn][0](*[mp.mpf(a) for a in args]))
        rel = abs(value-ref)/max(abs(ref),1e-300)
        rows.setdefault(fn, []).append((args, value, ref, rel))
    fail = 0
    print(f"{'function':<26}{'points':>7}{'max_rel_err':>16}  verdict")
    for fn in REF:
        pts = rows.get(fn, [])
        if not pts:
            print(f"{fn:<26}{0:>7}{'NO DATA':>16}  {'FAIL(no-data)' if require_all else 'SKIP'}")
            if require_all: fail = 1
            continue
        worst = max(pts, key=lambda r: r[3]); mre = worst[3]; thr = REF[fn][1]
        ok = mre <= thr
        print(f"{fn:<26}{len(pts):>7}{mre:>16.3e}  {'PASS' if ok else 'FAIL(thr=%.0e)'%thr}")
        if not ok: fail = 1
    print("STATS_DIST_PARITY_OK" if not fail else "STATS_DIST_PARITY_FAIL")
    return fail

def selftest():
    v = float(mp.ncdf(1.0))
    line = f"standard_normal_cdf 1 {_bits(1.0)} {_bits(v)}\n"
    import io; sys.stdin = io.StringIO(line)
    assert main() == 0, "selftest: standard_normal_cdf(1) should pass"
    print("STATS_REF_SELFTEST_OK")

if __name__ == "__main__":
    if "--selftest" in sys.argv: selftest()
    else: sys.exit(main(require_all=("--require-all" in sys.argv)))
