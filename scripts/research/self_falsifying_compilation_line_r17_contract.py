#!/usr/bin/env python3
"""Self-falsifying compilation, rung R17 — witness binding, in the compiler.

Spec: docs/research/self_falsifying_compilation_line_r17_2026-07-28.md

R15 measured the limit of R2's contribution: a verdict token binds the
PROPOSITION a check reports, so its resolution is bounded by the invariance group
of that proposition. R16 identified the group — maps acting within the blocks of
a classification — and both rungs proposed the repair: bind a WITNESS of the
proposition rather than its truth value, verified in Python as discriminating.

R17 implements it in the compiler. First compiler change since R2, ten rungs ago.

WHAT WAS ADDED, confined to self-hosted/compiler/claim_executor.sio:
  * a claim may declare `witness = "<fingerprint>"`;
  * the gate's captured output is read for `<PREFIX>_WITNESS <fingerprint>`;
  * mismatch (6) and absence (7) refuse codegen, after the token decision.
The parser needed no change: claim field names are not allowlisted.

THE PROBE THAT MATTERS is W2. Its gate exits 0 AND emits exactly the declared
verdict token, so exit-code gating passes it and token binding passes it. The
build is refused anyway, because the witness differs. Nothing below this
mechanism can distinguish that case from a good build.

CLAUSES:

  X1_EXECUTOR_SURFACE
      The mechanism is present in the executor source: the field, a shared
      extractor (not a duplicated one), the outcome function, both failure arms.

  X2_NO_DUPLICATED_DERIVATION
      The token and witness readers share one implementation. Writing the
      extractor twice would be the exact failure R6 measures, inside the arc
      that measures it — so it is checked, not trusted.

  X3_BEHAVIOUR_RECEIPT
      Source surface is not behaviour. R2 learned this the hard way: its
      contract certified the mechanism as implemented while the compiler built
      from that source segfaulted on every claim. This clause requires a receipt
      that W1-W4 and the R2/R0-R1 regressions were observed, bound to the
      executor's sha256.

Pure Python 3.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXECUTOR = REPO / "self-hosted/compiler/claim_executor.sio"
RECEIPT = REPO / "artifacts/self_falsifying_r17_receipt.txt"
FIX = REPO / "scripts/ci/fixtures"

SURFACE = [
    ("witness field read",        r'ce_name_eq_str\(f\.name,\s*"witness"\)'),
    ("witness line convention",   r'ce_extract_after\(out,\s*"_WITNESS ",\s*9\)'),
    ("outcome function",          r"fn ce_witness_outcome\("),
    ("mismatch code 6",           r"6\s*//\s*CLAIM_GATE_WITNESS_MISMATCH"),
    ("absent code 7",             r"7\s*//\s*CLAIM_GATE_WITNESS_ABSENT"),
    ("mismatch arm",              r'print\("CLAIM_WITNESS_MISMATCH "\)'),
    ("absent arm",                r'print\("CLAIM_WITNESS_ABSENT "\)'),
    ("fresh-variable discipline", r"var settled = decided"),
]

RECEIPT_ROWS = ["W1_MATCH_PASSES", "W2_DRIFT_BLOCKS", "W3_ABSENT_BLOCKS",
                "W4_BACKWARD_COMPAT", "R2_REGRESSION_PASS",
                "R2_REGRESSION_DRIFT", "R2_REGRESSION_ABSENT",
                "R01_REGRESSION_EXITCODE", "R01_REGRESSION_NOOP"]

FIXTURES = ["self_falsifying_witness_pass.sio", "self_falsifying_witness_drift.sio",
            "self_falsifying_witness_absent.sio", "self_falsifying_witness_compat.sio",
            "self_falsifying_witness_match.sh", "self_falsifying_witness_drift.sh",
            "self_falsifying_witness_absent.sh"]


def main() -> int:
    src = EXECUTOR.read_text()
    sha = hashlib.sha256(EXECUTOR.read_bytes()).hexdigest()

    print("R17 — witness binding, in the compiler")
    print("=" * 72)
    print(f"executor {EXECUTOR.relative_to(REPO)}  sha256 {sha[:16]}...")
    print()

    # ---- X1 -----------------------------------------------------------------
    missing = []
    for label, pat in SURFACE:
        hit = re.search(pat, src) is not None
        print(f"  [{'OK' if hit else 'MISSING'}] {label}")
        if not hit:
            missing.append(label)
    for f in FIXTURES:
        if not (FIX / f).exists():
            missing.append(f"fixture {f}")
            print(f"  [MISSING] fixture {f}")
    x1 = not missing
    print(f"X1_EXECUTOR_SURFACE {'PASS' if x1 else 'FAIL'}  "
          f"{len(SURFACE) - len([m for m in missing if not m.startswith('fixture')])}"
          f"/{len(SURFACE)} surface elements, {len(FIXTURES)} fixtures")
    print()

    # ---- X2 -----------------------------------------------------------------
    # One extractor body, two thin readers delegating to it.
    bodies = re.findall(r'ce_find_sub\(out, needle, 0\)', src)
    tok_delegates = re.search(
        r"fn ce_extract_verdict_token\(out: string\) -> string[^\n]*\n\s*"
        r'ce_extract_after\(out, "_VERDICT ", 9\)', src) is not None
    wit_delegates = re.search(
        r"fn ce_extract_witness\(out: string\) -> string[^\n]*\n\s*"
        r'ce_extract_after\(out, "_WITNESS ", 9\)', src) is not None
    x2 = len(bodies) == 1 and tok_delegates and wit_delegates
    print(f"  [{'OK' if len(bodies) == 1 else 'FAIL'}] exactly one scan body "
          f"(found {len(bodies)})")
    print(f"  [{'OK' if tok_delegates else 'FAIL'}] token reader delegates")
    print(f"  [{'OK' if wit_delegates else 'FAIL'}] witness reader delegates")
    print(f"X2_NO_DUPLICATED_DERIVATION {'PASS' if x2 else 'FAIL'}")
    print()

    # ---- X3 -----------------------------------------------------------------
    if not RECEIPT.exists():
        print("X3_BEHAVIOUR_RECEIPT FAIL  no receipt")
        print("  Source surface is not behaviour. Build a witness-binding")
        print("  compiler and run the probe suite (see the spec's Reproduce),")
        print("  which writes artifacts/self_falsifying_r17_receipt.txt.")
        x3 = False
        rec = ""
    else:
        rec = RECEIPT.read_text()
        rsha = re.search(r"executor_sha256=([0-9a-f]{64})", rec)
        sha_ok = bool(rsha) and rsha.group(1) == sha
        rows_ok = [r for r in RECEIPT_ROWS if r in rec]
        drift_ok = re.search(r"W2_DRIFT_BLOCKS\s+rc=1\s+elf=no\s+CLAIM_WITNESS_MISMATCH",
                             rec) is not None
        print(f"  [{'OK' if sha_ok else 'STALE'}] receipt is bound to THIS executor source")
        if not sha_ok:
            print("      the executor changed since the probes ran — rerun them")
        print(f"  [{'OK' if len(rows_ok) == len(RECEIPT_ROWS) else 'MISSING'}] "
              f"{len(rows_ok)}/{len(RECEIPT_ROWS)} observations recorded")
        print(f"  [{'OK' if drift_ok else 'FAIL'}] W2: gate exits 0, token correct, "
              f"build REFUSED")
        x3 = sha_ok and len(rows_ok) == len(RECEIPT_ROWS) and drift_ok
        print(f"X3_BEHAVIOUR_RECEIPT {'PASS' if x3 else 'FAIL'}")
    print()

    ok = x1 and x2 and x3
    verdict = ("WITNESS_BINDING_IMPLEMENTED__REFUSES_ON_PRESERVED_PROPOSITION"
               if ok else "SURFACE_ONLY__NOT_CERTIFIED")
    print("-" * 72)
    print("R2 bound the build to the proposition a check reports. R15/R16 showed")
    print("that is blind to anything preserving the proposition's truth. This")
    print("binds the witness, and refuses a build whose grounds were replaced.")
    print()
    print(f"SELF_FALSIFYING_R17_VERDICT {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
