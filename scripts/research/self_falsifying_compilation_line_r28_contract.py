#!/usr/bin/env python3
"""Self-falsifying compilation, rung R28 — the confidence gate separates almost nothing.

Spec: docs/research/self_falsifying_compilation_line_r28_2026-08-01.md

The compiler carries an epistemic confidence scalar in 0..1000, a gate at 950, and
tiers built on it (PLATINUM == 1000). The question this rung asks is not whether
the number is CALIBRATED -- that would need labelled ground truth this repository
does not have. It asks something prior and cheaper, and the answer decides whether
calibration is even a meaningful thing to attempt:

  how is the scalar DISTRIBUTED over real source?

Measured over 30.6 million expression tokens: the support is genuinely graded --
66 distinct values -- and 99.993% of the mass sits at exactly 0 or exactly 1000.
Fewer than one token in thirty thousand lands strictly between them, and fewer
than three in a million land strictly between 0 and the 950 gate. The threshold's
position is very nearly inert: moving it anywhere in (0, 950] would change the
verdict for 0.003% of decisions.

That is not a claim that the confidence is wrong. It is a claim that an ECE or a
reliability diagram computed on this corpus would be a statement about a
two-valued predictor, and the graded tail it is supposed to describe is 0.007%
of the data.

CLAUSES:
  B1_SUPPORT_IS_GRADED        the scalar takes many distinct values; it is not a
                              boolean by construction.
  B2_MASS_IS_BINARY           yet almost all of it sits at the two extremes.
  B3_GATE_SEPARATES_ALMOST_NOTHING
                              the population strictly between 0 and the gate --
                              the only population whose verdict the threshold's
                              exact value decides.
  B4_SHARED_REDIRECT_INVENTS_VALUES
                              the control, and it is load-bearing. Running the
                              same census in parallel into ONE file tears the
                              output and produces confidences ABOVE 1000, which
                              cannot exist. A census taken that way reports a
                              support that is partly fabricated.
  B5_LIVE_CENSUS_AGREES       a fresh bounded census reproduces B2 on today's
                              tree, so the recorded receipt cannot go stale
                              silently.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "scripts/research/r28/conf_census.json"
LEAN = ROOT / "bin/souc-lean-single-x86_64"
GATE_VALUE = 950
LIVE_FILES = 24          # bounded so the gate stays fast
CONF_RE = re.compile(r'"conf":(\d+)')


def load_receipt():
    d = json.loads(RECEIPT.read_text())
    counts = {int(k): int(v) for k, v in d["counts"].items()}
    return d, counts


def dump_conf(path: Path, out_path: Path) -> None:
    """One input, ONE output file. Never a shared handle -- see B4."""
    with tempfile.NamedTemporaryFile(suffix=".elf", delete=True) as elf:
        with out_path.open("w") as fh:
            subprocess.run([str(LEAN), str(path), elf.name, "--dump-conf-json"],
                           stdout=fh, stderr=subprocess.STDOUT,
                           timeout=120, cwd=str(ROOT))


def clause_b1(counts) -> bool:
    graded = {k: v for k, v in counts.items() if k not in (0, 1000)}
    print(f"B1 distinct confidence values observed: {len(counts)}")
    print(f"    of which strictly between 0 and 1000: {len(graded)}")
    print(f"    sample: {sorted(graded)[:14]} ...")
    print("    The scalar is graded by construction. It is not a boolean.")
    ok = len(graded) >= 10
    print(f"B1_SUPPORT_IS_GRADED {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def clause_b2(counts) -> bool:
    n = sum(counts.values())
    extremes = counts.get(0, 0) + counts.get(1000, 0)
    graded = n - extremes
    pct = 100.0 * extremes / n if n else 0.0
    print(f"B2 tokens measured: {n:,}")
    print(f"    conf == 0     {counts.get(0,0):>12,}")
    print(f"    conf == 1000  {counts.get(1000,0):>12,}")
    print(f"    everything else {graded:>10,}   ({100.0*graded/n:.4f}%)")
    print(f"    mass at the two extremes: {pct:.4f}%")
    print("    Graded in principle; two-valued in practice.")
    ok = n > 1_000_000 and pct > 99.9
    print(f"B2_MASS_IS_BINARY {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def clause_b3(counts) -> bool:
    n = sum(counts.values())
    below = sum(v for k, v in counts.items() if 0 < k < GATE_VALUE)
    within = sum(v for k, v in counts.items() if GATE_VALUE <= k < 1000)
    print(f"B3 the gate sits at {GATE_VALUE}. The only tokens whose verdict its exact")
    print("    position decides are those strictly between 0 and the gate:")
    print(f"    strictly in (0, {GATE_VALUE}):   {below:,}   ({100.0*below/n:.5f}% of all tokens)")
    print(f"    in [{GATE_VALUE}, 1000):        {within:,}")
    print("    Move the threshold anywhere in (0, 950] and the corpus barely notices.")
    ok = n > 0 and below < n / 10_000
    print(f"B3_GATE_SEPARATES_ALMOST_NOTHING {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def clause_b4(files) -> bool:
    """Run the census the WRONG way on purpose and show it invents data."""
    if not LEAN.is_file():
        print("B4 lean_single binary absent; cannot run the control")
        print("B4_SHARED_REDIRECT_INVENTS_VALUES FAIL")
        print()
        return False
    with tempfile.TemporaryDirectory() as td:
        shared = Path(td) / "shared.txt"
        procs = []
        with shared.open("w") as fh:
            for i, f in enumerate(files):
                elf = Path(td) / f"c{i}.elf"
                procs.append(subprocess.Popen(
                    [str(LEAN), str(f), str(elf), "--dump-conf-json"],
                    stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT)))
            for p in procs:
                p.wait(timeout=180)
        vals = [int(m) for m in CONF_RE.findall(shared.read_text(errors="replace"))]
    impossible = sorted({v for v in vals if v > 1000})
    n_bad = len([v for v in vals if v > 1000])
    print(f"B4 {len(files)} compilers writing into ONE file, concurrently:")
    print(f"    values above 1000, which the scalar cannot take: {n_bad}")
    print(f"    e.g. {impossible[:6]}")
    print("    Interleaved writes tear the JSON mid-number. A census taken this way")
    print("    reports a support that is partly fabricated -- and the fabricated")
    print("    values land in the graded tail, which is exactly what B1 measures.")
    ok = n_bad > 0
    print(f"B4_SHARED_REDIRECT_INVENTS_VALUES {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def clause_b5(files) -> bool:
    """Fresh bounded census, one file per output, must reproduce B2."""
    if not LEAN.is_file():
        print("B5 lean_single binary absent")
        print("B5_LIVE_CENSUS_AGREES FAIL")
        print()
        return False
    c = Counter()
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(files):
            out = Path(td) / f"{i}.out"
            try:
                dump_conf(f, out)
            except Exception:
                continue
            c.update(int(m) for m in CONF_RE.findall(out.read_text(errors="replace")))
    n = sum(c.values())
    impossible = [k for k in c if k > 1000]
    extremes = c.get(0, 0) + c.get(1000, 0)
    pct = 100.0 * extremes / n if n else 0.0
    print(f"B5 live census over {len(files)} files, one output each: {n:,} tokens")
    print(f"    mass at {{0, 1000}}: {pct:.4f}%")
    print(f"    values above 1000: {len(impossible)} (must be 0 when written correctly)")
    ok = n > 0 and pct > 99.0 and not impossible
    print(f"B5_LIVE_CENSUS_AGREES {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def main() -> int:
    print("Self-falsifying compilation R28 -- the confidence gate separates almost nothing")
    print("=" * 78)
    print()
    if not RECEIPT.is_file():
        print(f"missing receipt: {RECEIPT.relative_to(ROOT)}")
        print("SELF_FALSIFYING_R28_VERDICT INCONCLUSIVE")
        return 1
    meta, counts = load_receipt()
    print(f"receipt: {meta['files_measured']} files, {meta['tokens']:,} tokens")
    print(f"method:  {meta['method']}")
    print()

    files = sorted((ROOT / "tests/run-pass").glob("*.sio"))[:LIVE_FILES]
    if not files:
        print("no run-pass corpus to census")
        print("SELF_FALSIFYING_R28_VERDICT INCONCLUSIVE")
        return 1

    ok1 = clause_b1(counts)
    ok2 = clause_b2(counts)
    ok3 = clause_b3(counts)
    ok4 = clause_b4(files)
    ok5 = clause_b5(files)

    ok = ok1 and ok2 and ok3 and ok4 and ok5
    verdict = ("CONFIDENCE_IS_GRADED_IN_PRINCIPLE__BINARY_IN_PRACTICE"
               if ok else "INCONCLUSIVE")
    print("-" * 78)
    print("The scalar can take dozens of values and almost never does. The tier")
    print("boundary the compiler advertises is exercised by a population three")
    print("orders of magnitude smaller than the rounding error of the corpus. Before")
    print("asking whether 950 is calibrated, one has to notice that almost nothing")
    print("is ever weighed against it.")
    print()
    print(f"SELF_FALSIFYING_R28_VERDICT {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
