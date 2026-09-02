#!/usr/bin/env python3
"""Compare math::qd128 transcendental output against the mpmath reference corpus.

Usage:
    export SOUNIO_STDLIB_PATH=$PWD/stdlib
    ./bin/souc run tests/vectors/qd128_transcendental/gen/qd128_transcendental_harness.sio \\
        > /tmp/qdt.out
    python3 scripts/dev/qd128_transcendental_accuracy_check.py /tmp/qdt.out

The oracle values were produced once by mpmath (see the corpus `_meta` record
and tests/vectors/qd128_transcendental/gen/). THIS script needs no mpmath: the
harness output is a sum of exact binary64 limbs and the reference is a decimal
string, so the comparison is done in exact rationals with `fractions`.

Reports each case's relative error in quad-double ulps (1 ulp = 2^-212) and
prints the per-op MIN/MAX interval -- an interval, never a single number.

Cases carry a `regime` from the corpus. `core` cases are held to --max-ulps
(default 64). The two non-core regimes are limits of the quad-double format and
of log's formulation rather than of the routines, so they are reported with
their measured numbers and held to their own documented budgets:
  subnormal_tail  the low limbs are subnormal, so the format only holds
                  `format_bits_available` bits; the budget is scaled by
                  2^(212 - format_bits_available).
  log_near_one    log carries ABSOLUTE quad-double accuracy, so this case is
                  checked as an absolute error (--max-abs-ulps, ulps of 1.0).
"""
import argparse
import json
import math
import os
import struct
import sys
from decimal import Decimal
from fractions import Fraction

ULP = Fraction(1, 2 ** 212)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_JSONL = os.path.join(
    REPO, "tests", "vectors", "qd128_transcendental", "qd128_transcendental.jsonl"
)


def b2f(i):
    return struct.unpack("<d", struct.pack("<q", int(i)))[0]


def qval(bit_list):
    return sum((Fraction(b2f(b)) for b in bit_list), Fraction(0))


def to_float(fr):
    """Fraction -> float without OverflowError on wild ratios."""
    try:
        return float(fr)
    except (OverflowError, ValueError):
        return math.inf


def fmt(x):
    if x == 0:
        return "0"
    return "%.4g" % x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("harness_output")
    ap.add_argument("--jsonl", default=DEFAULT_JSONL)
    ap.add_argument("--max-ulps", type=float, default=64.0)
    ap.add_argument("--max-abs-ulps", type=float, default=1.0e6)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    meta, cases = recs[0], recs[1:]
    assert "_meta" in meta, "first corpus record must be the _meta header"

    got = {}
    for line in open(args.harness_output):
        f = line.split()
        if len(f) != 5:
            continue
        try:
            got[int(f[0])] = [int(v) for v in f[1:]]
        except ValueError:
            continue

    rows, missing = [], []
    for idx, rec in enumerate(cases):
        tag = idx + 1
        if tag not in got:
            missing.append(rec["id"])
            continue
        v = qval(got[tag])
        ref = Fraction(Decimal(rec["expected_dec"]))
        err = abs(v - ref)
        rel = err / abs(ref) if ref != 0 else err
        rows.append(dict(rec=rec,
                         ulps=to_float(rel / ULP),
                         abs_ulps=to_float(err / ULP)))

    print("oracle: %s %s at %s bits; corpus = %d cases"
          % (meta["oracle"], meta["oracle_version"],
             meta["oracle_precision_bits"], len(cases)))
    print("relative error in quad-double ulps (1 ulp = 2^-212 = 1.5193e-64)")
    print()

    core = [r for r in rows if r["rec"]["regime"] == "core"]
    per_op = {}
    for r in core:
        per_op.setdefault(r["rec"]["op"], []).append(r)

    print("CORE CORPUS (per-op measured interval)")
    print("  %-8s %4s  %-11s %-11s   %s" % ("op", "n", "min ulp", "max ulp", "worst case"))
    for op in sorted(per_op):
        vals = sorted(per_op[op], key=lambda r: r["ulps"])
        print("  %-8s %4d  %-11s %-11s   %s  x=%s"
              % (op, len(vals), fmt(vals[0]["ulps"]), fmt(vals[-1]["ulps"]),
                 vals[-1]["rec"]["id"], vals[-1]["rec"]["x_dec"]))
    allv = sorted(r["ulps"] for r in core)
    print("  %-8s %4d  %-11s %-11s" % ("ALL", len(core), fmt(allv[0]), fmt(allv[-1])))
    print()

    others = [r for r in rows if r["rec"]["regime"] != "core"]
    if others:
        print("NON-CORE REGIMES (format / formulation limits, measured not hidden)")
        for r in sorted(others, key=lambda r: -r["ulps"]):
            rec = r["rec"]
            print("  %-15s %s  %-8s x=%-24s rel=%-11s abs=%-11s bits=%d"
                  % (rec["regime"], rec["id"], rec["op"], rec["x_dec"],
                     fmt(r["ulps"]), fmt(r["abs_ulps"]), rec["format_bits_available"]))
        print()

    if args.verbose:
        print("worst 12 core cases:")
        for r in sorted(core, key=lambda r: -r["ulps"])[:12]:
            rec = r["rec"]
            print("  %-8s %s  %-12s x=%-26s %s"
                  % (rec["op"], rec["id"], fmt(r["ulps"]), rec["x_dec"],
                     rec.get("note", "")))
        print()

    fails = []
    for r in rows:
        rec = r["rec"]
        regime = rec["regime"]
        if regime == "core":
            if r["ulps"] > args.max_ulps:
                fails.append((rec["id"], "core", r["ulps"], args.max_ulps))
        elif regime == "subnormal_tail":
            budget = args.max_ulps * float(2 ** (212 - rec["format_bits_available"]))
            if r["ulps"] > budget:
                fails.append((rec["id"], "subnormal_tail", r["ulps"], budget))
        elif regime == "log_near_one":
            if r["abs_ulps"] > args.max_abs_ulps:
                fails.append((rec["id"], "log_near_one(abs)", r["abs_ulps"],
                              args.max_abs_ulps))
        else:
            fails.append((rec["id"], "unknown regime %s" % regime, 0.0, 0.0))

    if missing:
        print("MISSING from harness output: %s" % ", ".join(missing))
    if fails:
        for cid, why, got_u, budget in fails:
            print("FAIL %s (%s): %s ulps > budget %s" % (cid, why, fmt(got_u), fmt(budget)))
        return 1
    if missing:
        print("FAIL: missing cases")
        return 1
    print("QD128_TRANSCENDENTAL_ACCURACY_OK "
          "(core %d cases in [%s, %s] ulps; %d non-core within documented budgets)"
          % (len(core), fmt(allv[0]), fmt(allv[-1]), len(others)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
