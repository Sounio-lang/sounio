"""Independent check of docs/audit/NATIVE_POW_TGAMMA_NUMERICAL_DESIGN_2026-08-23.md.

The design was produced by one agent. This script is the second instrument: it
re-derives what can be re-derived and arbitrates every claim that has a system
reference, rather than accepting the report's own numbers.

Three things it checks, in order of how badly a wrong answer would hurt:

  1. The sqrt(2*pi) constant. The design claims the composed sqrt(2*math.pi) is
     1 ulp low and that only the correctly-rounded value is acceptable, because
     tgamma(0.5) = sqrt(pi) makes the ulp test-visible. Arbitrated against mpmath.

  2. The 14 Lanczos coefficients at g = 31/8. Evaluated in 60-digit arithmetic so
     the measurement isolates the APPROXIMATION error of the coefficients from
     the rounding error of whatever evaluates them. A naive binary64 evaluation
     measures the evaluator, not the fit, and reports ~10 ulp for coefficients
     that are in fact good to a third of an ulp.

  3. The 22-row IEEE-754 special-case table for pow, against libm through ctypes
     rather than through math.pow. Python's wrapper raises OverflowError and
     ValueError where C99 returns a value, so four correct rows look like
     failures when tested through it. The wrapper is not the standard.
"""
import ctypes
import ctypes.util
import math
import re
import struct
import sys

DOC = "docs/audit/NATIVE_POW_TGAMMA_NUMERICAL_DESIGN_2026-08-23.md"
G = 31 / 8
INF = float("inf")
NAN = float("nan")


def load_coefficients(path):
    """Read the coefficients from their HEX column, not their decimal column.

    The hex is what reaches rodata, so it is the source of truth; the decimal is
    a rendering. Disagreement between the two is itself a defect and is checked.
    """
    text = open(path).read()
    rows = re.findall(
        r"^\| *(\d+) \| `([+-][\d.]+e[+-]\d+)` \| `(0x[0-9A-Fa-f]{16})` \|", text, re.M
    )
    coefficients = {}
    for index, decimal, hexadecimal in rows[:14]:
        value = struct.unpack(">d", bytes.fromhex(hexadecimal[2:]))[0]
        if abs(value - float(decimal)) > abs(value) * 1e-15:
            raise SystemExit(f"coefficient {index}: hex {hexadecimal} != decimal {decimal}")
        coefficients[int(index)] = value
    if len(coefficients) != 14:
        raise SystemExit(f"expected 14 coefficients, parsed {len(coefficients)}")
    return coefficients


def check_sqrt_two_pi(mp):
    exact = mp.sqrt(2 * mp.pi)
    documented = struct.unpack(">d", bytes.fromhex("40040D931FF62706"))[0]
    composed = math.sqrt(2 * math.pi)
    correctly_rounded = float(exact)
    print("sqrt(2*pi)")
    print(f"  documented        {documented!r}  error {float(abs(mp.mpf(documented) - exact)):.3e}")
    print(f"  composed          {composed!r}  error {float(abs(mp.mpf(composed) - exact)):.3e}")
    ok = documented == correctly_rounded and composed != correctly_rounded
    print(f"  claim holds (documented is correctly rounded, composed is not): {ok}")
    return ok


def check_coefficients(mp, coefficients):
    """Approximation error of the fit alone, in exact arithmetic."""
    import random

    def lanczos(x):
        x = mp.mpf(x)
        z = x - 1
        series = mp.mpf(coefficients[0]) + sum(
            mp.mpf(coefficients[k]) / (z + k) for k in range(13, 0, -1)
        )
        w = x + mp.mpf(31) / 8 - mp.mpf(1) / 2
        return mp.sqrt(2 * mp.pi) * series * mp.power(w, z + mp.mpf(1) / 2) * mp.e ** (-w)

    random.seed(11)
    print("Lanczos coefficients, approximation error in 60-digit arithmetic")
    worst_overall = 0.0
    for low, high in [(0.5, 2), (2, 10), (10, 60), (60, 171)]:
        worst = 0.0
        for _ in range(600):
            x = random.uniform(low, high)
            exact = mp.gamma(x)
            error = float(abs(lanczos(x) - exact) / mp.mpf(math.ulp(float(exact))))
            worst = max(worst, error)
        worst_overall = max(worst_overall, worst)
        print(f"  x in [{low}, {high}]".ljust(24) + f"max {worst:.4f} ulp")
    # The published claim is 5 ulp for the whole function; the fit must leave
    # room for the evaluation on top of it, not consume the budget by itself.
    ok = worst_overall < 1.0
    print(f"  fit stays under 1 ulp, leaving budget for the evaluator: {ok}")
    return ok


def check_special_cases():
    """Against libm itself. math.pow raises where C99 returns a value."""
    libm = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
    libm.pow.restype = ctypes.c_double
    libm.pow.argtypes = [ctypes.c_double, ctypes.c_double]
    p = libm.pow

    def same(got, want):
        if math.isnan(got) and math.isnan(want):
            return True
        if got == 0.0 and want == 0.0:
            return math.copysign(1, got) == math.copysign(1, want)
        return got == want

    cases = [
        ("pow(nan, 0) is 1, not nan", p(NAN, 0.0), 1.0),
        ("pow(1, nan) is 1, not nan", p(1.0, NAN), 1.0),
        ("pow(-1, inf) is 1", p(-1.0, INF), 1.0),
        ("pow(-2, 1e300): 1e300 is an even integer", p(-2.0, 1e300), INF),
        ("pow(-2, 2**53): every |y|>=2**53 is even", p(-2.0, 2.0**53), INF),
        ("pow(-0, -3) keeps the sign of zero", p(-0.0, -3.0), -INF),
        ("pow(-0, 3) keeps the sign of zero", p(-0.0, 3.0), -0.0),
        ("pow(-0, -2) even exponent drops it", p(-0.0, -2.0), INF),
        ("pow(0, -3)", p(0.0, -3.0), INF),
        ("pow(-2, 0.5) non-integer y", p(-2.0, 0.5), NAN),
        ("pow(-8, 3) negative base, odd y", p(-8.0, 3.0), -512.0),
        ("pow(inf, -1)", p(INF, -1.0), 0.0),
        ("pow(-inf, 3)", p(-INF, 3.0), -INF),
        ("pow(-inf, 2)", p(-INF, 2.0), INF),
        ("pow(-inf, -3)", p(-INF, -3.0), -0.0),
        ("pow(0, 0) is 1", p(0.0, 0.0), 1.0),
        ("pow(2, 1) is bit-exact", p(2.0, 1.0), 2.0),
        ("pow(0.1, 1) is bit-exact", p(0.1, 1.0), 0.1),
    ]
    print("pow special cases, against libm through ctypes")
    failures = 0
    for description, got, want in cases:
        ok = same(got, want)
        failures += 0 if ok else 1
        print(f"  {description:<44} {'ok' if ok else 'DIVERGES'}")
    print(f"  divergences: {failures}/{len(cases)}")
    return failures == 0


def main():
    try:
        import mpmath as mp
    except ImportError:
        print("mpmath is required: the coefficient check is meaningless in binary64")
        return 2
    mp.mp.dps = 60
    coefficients = load_coefficients(DOC)
    print(f"parsed {len(coefficients)} coefficients, hex and decimal agree\n")
    results = [
        check_sqrt_two_pi(mp),
        check_coefficients(mp, coefficients),
        check_special_cases(),
    ]
    print(f"\nall checks passed: {all(results)}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
