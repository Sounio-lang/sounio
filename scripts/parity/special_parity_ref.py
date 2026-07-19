#!/usr/bin/env python3
"""Compare Sounio special-function emitter output against mpmath (dps=30).
Bit-exact wire on stdin: `<fn> <nargs> <arg_bits...> <val_bits>`, each field a
signed-i64 IEEE-754 bit pattern (Sounio f64_to_bits). No scipy dependency."""
import sys, struct, mpmath as mp
mp.mp.dps = 30

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
}

def main():
    rows = {}
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("#"): continue
        parts = line.split()
        if not parts or parts[0] not in REF: continue
        fn = parts[0]; nargs = int(parts[1])
        args  = [bits_to_f64(parts[2+i]) for i in range(nargs)]
        value = bits_to_f64(parts[2+nargs])
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
