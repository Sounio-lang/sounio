#!/usr/bin/env python3
"""Generate the independent reference corpus for math::qd128 transcendentals.

HOW THIS CORPUS WAS PRODUCED
============================
Oracle:      mpmath 1.3.0 on CPython 3.12.3, mp.prec = 400 binary digits
             (~120 decimal digits), i.e. ~188 bits of headroom over the
             quad-double target of 212 bits.
Independence: nothing in this file calls Sounio. Every `expected` value is
             mpmath's, evaluated at the EXACT binary64/quad-double input that
             the Sounio harness will consume -- inputs are carried as exact
             (mantissa, exponent) integer pairs, never as decimal strings, so
             souc's (not correctly rounded) decimal literal parser cannot
             perturb the comparison.
Emitted:     tests/vectors/qd128_transcendental/qd128_transcendental.jsonl
                 one JSON record per case; record 0 is the `_meta` provenance
                 header. `expected_dec` is the reference value rounded to 90
                 significant decimal digits.
             tests/vectors/qd128_transcendental/gen/qd128_transcendental_harness.sio
                 the generated Sounio program that evaluates each case and
                 prints `<id> <bits(x0)> <bits(x1)> <bits(x2)> <bits(x3)>`.
Checked by:  scripts/dev/qd128_transcendental_accuracy_check.py

Regenerate with:
    python3 tests/vectors/qd128_transcendental/gen/gen_qd128_transcendental_vectors.py
"""
import json
import math
import os
import struct
import sys

from mpmath import mp, mpf, exp, log, sqrt, power

mp.prec = 400

MPMATH_PREC = 400
QD_ULP_EXP = -212  # 4 x 53-bit limbs

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
VEC_DIR = os.path.join(REPO, "tests", "vectors", "qd128_transcendental")
GEN_DIR = os.path.join(VEC_DIR, "gen")


def bits(x):
    return struct.unpack("<q", struct.pack("<d", x))[0]


def me(x):
    """Exact (mantissa, exponent) with |m| < 2^53 and x == m * 2^e."""
    if x == 0.0:
        return (0, 0)
    m, e = math.frexp(x)
    mi = int(m * (1 << 53))
    ee = e - 53
    assert mi * mpf(2) ** ee == mpf(x), (x, mi, ee)
    assert abs(mi) < (1 << 53)
    return (mi, ee)


def split_limbs(v, n=4):
    """Greedy nearest-f64 expansion: v ~ sum(out), each limb non-overlapping."""
    out = []
    r = mpf(v)
    for _ in range(n):
        h = float(r)
        out.append(h)
        r = r - mpf(h)
    return out


def qd_of(v):
    """Exact quad-double representation of an mpf (as 4 f64 limbs)."""
    return split_limbs(v)


def qd_value(limbs):
    t = mpf(0)
    for x in limbs:
        t += mpf(x)
    return t


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
# Each entry: (op, x_limbs, y_limbs_or_None, note)
CASES = []


def add(op, x, y=None, note=""):
    xl = x if isinstance(x, list) else qd_of(mpf(x))
    yl = None
    if y is not None:
        yl = y if isinstance(y, list) else qd_of(mpf(y))
    CASES.append((op, xl, yl, note))


LN2 = log(mpf(2))
LN10 = log(mpf(10))

# --- exp: spread over the whole finite binary64 exponent range -------------
for v, note in [
    ("-700", "near underflow"),
    ("-100", ""),
    ("-10", ""),
    ("-2", ""),
    ("-1", ""),
    ("-0.5", ""),
    ("-0.1", ""),
    ("-1e-5", "small argument"),
    ("-1e-20", "tiny argument"),
    ("1e-20", "tiny argument"),
    ("1e-5", "small argument"),
    ("0.1", ""),
    ("0.5", ""),
    ("1", "e"),
    ("2", ""),
    ("5", ""),
    ("10", ""),
    ("100", ""),
    ("500", ""),
    ("700", "near overflow"),
    ("709", "last decade before binary64 overflow"),
]:
    add("exp", v, note=note)

# reduction-boundary arguments (|r| = ln2/2 exactly after k*ln2 subtraction)
add("exp", qd_of(LN2 / 2), note="reduction boundary +ln2/2 (multi-limb input)")
add("exp", qd_of(-LN2 / 2), note="reduction boundary -ln2/2 (multi-limb input)")
add("exp", qd_of(LN2), note="k=1 exactly, r=0 (multi-limb input)")
add("exp", qd_of(LN2 * 3 / 2), note="reduction boundary 3ln2/2 (multi-limb input)")
add("exp", qd_of(LN2 * 100), note="k=100 exactly (multi-limb input)")

# --- log -------------------------------------------------------------------
for v, note in [
    ("1e-300", "denormal-adjacent decade"),
    ("1e-100", ""),
    ("1e-10", ""),
    ("0.001", ""),
    ("0.1", ""),
    ("0.5", ""),
    ("0.9", ""),
    ("1.5", ""),
    ("2", ""),
    ("3", ""),
    ("7", ""),
    ("10", ""),
    ("100", ""),
    ("1e10", ""),
    ("1e100", ""),
    ("1e300", ""),
]:
    add("log", v, note=note)
add("log", qd_of(exp(mpf(1))), note="log(e) (multi-limb input)")
add("log", "1.0000001", note="near 1: absolute-accuracy regime, see module note")
add("log", "1.5e-8", note="")

# --- log10 -----------------------------------------------------------------
for v, note in [
    ("1000", "exact integer result"),
    ("1e-7", "exact integer result"),
    ("2", ""),
    ("0.5", ""),
    ("1e100", "exact integer result"),
    ("12345.6789", ""),
    ("1.0000001e-30", ""),
]:
    add("log10", v, note=note)

# --- pow -------------------------------------------------------------------
add("pow", "2", "0.5", "sqrt via pow")
add("pow", "2.5", "3.7", "")
add("pow", "0.5", "-1.5", "negative exponent")
add("pow", "1e10", "0.25", "")
add("pow", "7", qd_of(mpf(1) / 3), "cube root (multi-limb exponent)")
add("pow", "10", qd_of(1 / LN10), "")
add("pow", "2", "100", "exact integer exponent -> binary exponentiation")
add("pow", "3", "-7", "negative exact integer exponent")
add("pow", "1.0000000001", "1000", "exact integer exponent, base near 1")
add("pow", "-2", "3", "negative base, odd integer exponent")

# --- Van't Hoff / Arrhenius shaped evaluations -----------------------------
# K(T)/K(T0) = exp(-dH/R * (1/T - 1/T0)), dH = -40000 J/mol,
# R = 8.31446261815324 J/(mol K), T0 = 298.15 K.
R_GAS = "8.31446261815324"
DH = "-40000"
T0 = "298.15"
for T in ["298.15", "310", "323.15", "348", "373.15", "393"]:
    # argument of exp, formed EXACTLY in mpmath then rounded to a qd input
    arg = -mpf(DH) / mpf(R_GAS) * (1 / mpf(T) - 1 / mpf(T0))
    add("exp", qd_of(arg), note="Van't Hoff exponent, T=%s K, dH=%s J/mol" % (T, DH))
# Arrhenius k = A exp(-Ea/(R T)); Ea = 65000 J/mol
for T in ["298.15", "348", "393"]:
    arg = -mpf("65000") / (mpf(R_GAS) * mpf(T))
    add("exp", qd_of(arg), note="Arrhenius exponent, Ea=65000 J/mol, T=%s K" % T)
# pH = -log10[H+]
for c in ["1e-7", "3.98107170553497e-8", "1.2e-3", "5.5e-11"]:
    add("log10", c, note="pH = -log10[H+], [H+]=%s M" % c)
# log K
for kk in ["1e-14", "4.5e-4", "1.75e-5", "6.3e12"]:
    add("log", kk, note="log K, K=%s" % kk)

# --- round trip: log(exp(x)) should return x -------------------------------
for v in ["-500", "-37.5", "-1", "-0.25", "0.25", "1", "37.5", "300", "690"]:
    add("logexp", v, note="round trip log(exp(x)) == x")


# ---------------------------------------------------------------------------
# Reference evaluation
# ---------------------------------------------------------------------------
def reference(op, xv, yv):
    if op == "exp":
        return exp(xv)
    if op == "log":
        return log(xv)
    if op == "log10":
        return log(xv) / LN10
    if op == "pow":
        if xv < 0:
            n = int(yv)
            assert mpf(n) == yv
            return power(xv, n)
        return exp(yv * log(xv))
    if op == "logexp":
        return xv
    raise AssertionError(op)


records = []
meta = {
    "_meta": "qd128 transcendental reference corpus",
    "produced_by": "tests/vectors/qd128_transcendental/gen/gen_qd128_transcendental_vectors.py",
    "oracle": "mpmath",
    "oracle_version": "1.3.0",
    "python": sys.version.split()[0],
    "oracle_precision_bits": MPMATH_PREC,
    "target_precision_bits": 212,
    "qd_ulp": "2^-212 = 1.5192908...e-64",
    "independence": (
        "No Sounio code participates in producing these values. Inputs are "
        "carried as exact (mantissa, exponent) integer pairs so that souc's "
        "decimal literal parser cannot perturb the comparison; the reference "
        "is mpmath evaluated at exactly those quad-double inputs."
    ),
    "expected_dec_digits": 90,
    "ops": sorted(set(c[0] for c in CASES)),
    "case_count": len(CASES),
}
records.append(meta)

for i, (op, xl, yl, note) in enumerate(CASES):
    cid = "qdt_%04d" % (i + 1)
    xv = qd_value(xl)
    yv = qd_value(yl) if yl is not None else None
    ref = reference(op, xv, yv)
    # Regime classification. Two families of case cannot reach 212 bits for
    # reasons that belong to the format / to log itself, not to the routines:
    #   subnormal_tail: |result| < 2^-863, so the low limbs of the quad-double
    #                   are subnormal and the format holds only 1074+E bits.
    #   log_near_one:   log's residual is formed near 1, so a result near 0 has
    #                   quad-double ABSOLUTE, not relative, accuracy.
    # They are measured and reported, never dropped.
    regime = "core"
    fmt_bits = 212
    if ref != 0:
        E = int(mp.floor(log(abs(ref)) / log(mpf(2))))
        if E < -863:
            regime = "subnormal_tail"
            fmt_bits = max(0, 1074 + E)
    if op in ("log", "log10") and abs(xv - 1) < mpf("1e-3"):
        regime = "log_near_one"
    rec = {
        "id": cid,
        "op": op,
        "regime": regime,
        "format_bits_available": fmt_bits,
        "x_bits": [bits(v) for v in xl],
        "x_me": [list(me(v)) for v in xl],
        "x_dec": mp.nstr(xv, 40),
        "expected_dec": mp.nstr(ref, 90),
        "expected_limb_bits": [bits(v) for v in split_limbs(ref)],
        "note": note,
    }
    if yl is not None:
        rec["y_bits"] = [bits(v) for v in yl]
        rec["y_me"] = [list(me(v)) for v in yl]
        rec["y_dec"] = mp.nstr(yv, 40)
    records.append(rec)

os.makedirs(GEN_DIR, exist_ok=True)
jsonl = os.path.join(VEC_DIR, "qd128_transcendental.jsonl")
with open(jsonl, "w") as fh:
    for r in records:
        fh.write(json.dumps(r, sort_keys=True) + "\n")

# ---------------------------------------------------------------------------
# Sounio harness
# ---------------------------------------------------------------------------
L = []
L.append("// GENERATED by tests/vectors/qd128_transcendental/gen/gen_qd128_transcendental_vectors.py")
L.append("// Do not edit by hand. Prints '<id> <bits(x0)> <bits(x1)> <bits(x2)> <bits(x3)>'")
L.append("// for every case in tests/vectors/qd128_transcendental/qd128_transcendental.jsonl.")
L.append("//")
L.append("// Inputs are rebuilt from exact (mantissa, exponent) integer pairs rather than")
L.append("// decimal literals: souc's decimal literal parser is not correctly rounded and")
L.append("// would otherwise feed the harness a different number than the oracle saw.")
L.append("use math::qd128::*")
L.append("")
L.append("fn mk(m: i64, e: i64) -> f64 with Mut {")
L.append("    var r = m as f64")
L.append("    var n = e")
L.append("    while n < 0 {")
L.append("        r = r * 0.5")
L.append("        n = n + 1")
L.append("    }")
L.append("    while n > 0 {")
L.append("        r = r * 2.0")
L.append("        n = n - 1")
L.append("    }")
L.append("    r")
L.append("}")
L.append("")
L.append("fn emit(tag: i64, q: Qd128) with IO, Mut, Panic, Div {")
L.append("    print_int(tag)")
L.append("    print(\" \")")
L.append("    print_int(f64_to_bits(q.x0))")
L.append("    print(\" \")")
L.append("    print_int(f64_to_bits(q.x1))")
L.append("    print(\" \")")
L.append("    print_int(f64_to_bits(q.x2))")
L.append("    print(\" \")")
L.append("    print_int(f64_to_bits(q.x3))")
L.append("    print(\"\\n\")")
L.append("}")
L.append("")
L.append("fn main() -> i32 with IO, Mut, Panic, Div {")


def sio_int(n):
    return str(n) if n >= 0 else "0 - %d" % (-n)


def sio_qd(name, limbs):
    out = ["    let %s = Qd128 {" % name]
    for idx, v in enumerate(limbs):
        m, e = me(v)
        out.append("        x%d: mk(%s, %s)%s" % (idx, sio_int(m), sio_int(e), "," if idx < 3 else ""))
    out.append("    }")
    return out


for i, (op, xl, yl, note) in enumerate(CASES):
    tag = i + 1
    L.append("    // %s  %s" % (op, note))
    L.extend(sio_qd("x%d" % tag, xl))
    if yl is not None:
        L.extend(sio_qd("y%d" % tag, yl))
    if op == "exp":
        L.append("    emit(%d, qd_exp(x%d))" % (tag, tag))
    elif op == "log":
        L.append("    emit(%d, qd_log(x%d))" % (tag, tag))
    elif op == "log10":
        L.append("    emit(%d, qd_log10(x%d))" % (tag, tag))
    elif op == "pow":
        L.append("    emit(%d, qd_pow(x%d, y%d))" % (tag, tag, tag))
    elif op == "logexp":
        L.append("    emit(%d, qd_log(qd_exp(x%d)))" % (tag, tag))
    else:
        raise AssertionError(op)
    L.append("")

L.append("    return 0")
L.append("}")

harness = os.path.join(GEN_DIR, "qd128_transcendental_harness.sio")
with open(harness, "w") as fh:
    fh.write("\n".join(L) + "\n")

print("wrote %s (%d cases)" % (jsonl, len(CASES)))
print("wrote %s" % harness)
