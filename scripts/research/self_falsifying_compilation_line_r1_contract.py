#!/usr/bin/env python3
"""Self-falsifying compilation, rung R1 — corpus binding.

Spec: docs/research/self_falsifying_compilation_line_r1_2026-07-26.md

R0 measured that the mechanism guarded nothing: 0 native claims outside tests
and CI fixtures. R1 binds real CI gates to native claims in a real source file
and walks into the module-closure wall on purpose.

Clauses (all static; the compile-time measurements are re-run by the gate's
optional compile arm, see the spec's Reproduce section):

  B1_MANIFEST_BOUND    a non-test, non-fixture source carries native claims,
                       every one bound to a real CI gate (no fixtures)
  B2_GATES_EXIST       every bound gate path exists and is executable
  B3_MODULE_CLOSURE    the module-closure probe fixtures exist and are shaped
                       so that the importer's compile outcome is decisive
  B4_TIMEOUT_BUDGET    no gate known to exceed the executor's wall-clock budget
                       is bound

Pure Python 3 + git. No third-party dependencies.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

MANIFEST = "examples/epistemic/rupture_claims_verified.sio"
MC_LIB = "scripts/ci/fixtures/self_falsifying_modclosure_lib.sio"
MC_MAIN = "scripts/ci/fixtures/self_falsifying_modclosure_main.sio"

# Per-gate wall-clock budget enforced by claim_executor.sio
# (CLAIM_GATE_TIMEOUT_MS = 30000).
EXECUTOR_BUDGET_MS = 30000

# Gates measured 2026-07-26 to exceed a 45 s probe, hence over the 30 s
# executor budget. They are excluded from binding by construction. Measured,
# not assumed: see the spec's sample table. A gate here is NOT asserted to
# fail — only to cost more than the executor will wait.
OVER_BUDGET_GATES = [
    "scripts/ci/compiler_lane_status_gate.sh",
    "scripts/ci/heuristic_firewall_gate.sh",
    "scripts/ci/knowledge_context_static_gate.sh",
    "scripts/ci/zd_qec_prediction_gate.sh",
    "scripts/ci/falsification_ledger_gate.sh",
]

# Gates measured 2026-07-26 to MUTATE THE WORKING TREE when run. Binding one
# would make every compile dirty the repository and make the build
# non-idempotent, so they are excluded regardless of speed or colour. Found by
# running each candidate and diffing `git status --porcelain` before/after.
NON_HERMETIC_GATES = [
    # rewrites results/associator_gum_variance/{RUNLOG.txt,receipt.v1.json}
    # with the current timestamp and git SHA on every run.
    "scripts/ci/associator_gum_variance_gate.sh",
]

# Paths under which a gate writing anything is presumed non-hermetic. Used for
# the static half of B5; the dynamic half is the probe described in the spec.
# The scan regex is DERIVED from this tuple — adding a prefix here is enough,
# with no second pattern to keep in sync by hand.
TREE_WRITE_PREFIXES = ("results/", ".sounio/", "artifacts/")
TREE_WRITE_RE = re.compile(
    "(?:" + "|".join(re.escape(p) for p in TREE_WRITE_PREFIXES) + r")[A-Za-z0-9_./-]*"
)

CLAIM_BLOCK_RE = re.compile(r"^claim\s+(\w+)\s*\{(.*?)^\}", re.MULTILINE | re.DOTALL)
GATE_FIELD_RE = re.compile(r'gate\s*=\s*"([^"]+)"')


def read(rel: str) -> str:
    try:
        return (REPO / rel).read_text(errors="replace")
    except OSError:
        return ""


def claims_in(rel: str) -> list[tuple[str, str]]:
    """Return [(claim_name, body)] for native claim blocks in a file."""
    return CLAIM_BLOCK_RE.findall(read(rel))


# ---------------------------------------------------------------- B1


def clause_b1() -> tuple[bool, list[tuple[str, str]]]:
    ok = True
    if not (REPO / MANIFEST).exists():
        print(f"B1_MANIFEST_BOUND FAIL — manifest missing: {MANIFEST}")
        return False, []

    if MANIFEST.startswith(("tests/", "scripts/ci/fixtures/")):
        print(f"B1_MANIFEST_BOUND FAIL — manifest is a test/fixture: {MANIFEST}")
        ok = False

    blocks = claims_in(MANIFEST)
    if not blocks:
        print(f"B1_MANIFEST_BOUND FAIL — no native claim blocks in {MANIFEST}")
        return False, []

    bound: list[tuple[str, str]] = []
    for name, body in blocks:
        m = GATE_FIELD_RE.search(body)
        if not m:
            print(f"  {name}: no gate field")
            ok = False
            continue
        gate = m.group(1)
        if "/fixtures/" in gate:
            print(f"  {name}: bound to a FIXTURE gate ({gate}) — not a real gate")
            ok = False
            continue
        bound.append((name, gate))

    print(f"B1_MANIFEST_BOUND {len(bound)} claims bound to real CI gates in {MANIFEST}")
    for name, gate in bound:
        print(f"    {name} -> {gate}")
    print(f"B1_MANIFEST_BOUND {'PASS' if ok else 'FAIL'}")
    return ok, bound


# ---------------------------------------------------------------- B2


def clause_b2(bound: list[tuple[str, str]]) -> bool:
    ok = True
    for name, gate in bound:
        path = REPO / gate
        if not path.exists():
            print(f"  {name}: gate does not exist: {gate}")
            ok = False
        elif not os.access(path, os.X_OK):
            print(f"  {name}: gate not executable: {gate}")
            ok = False
    print(f"B2_GATES_EXIST {'PASS' if ok else 'FAIL'} — "
          f"{len(bound)} bound gate paths exist and are executable")
    return ok


# ---------------------------------------------------------------- B3


def clause_b3() -> bool:
    """The module-closure probe must be shaped so its outcome is decisive.

    The imported module must carry a claim whose gate ALWAYS FAILS, and the
    importer must import it. Then the importer compiling cleanly can only mean
    the imported claim was never executed.
    """
    ok = True
    lib, main = read(MC_LIB), read(MC_MAIN)

    if not lib or not main:
        print("B3_MODULE_CLOSURE FAIL — probe fixtures missing")
        return False

    lib_claims = claims_in(MC_LIB)
    if not lib_claims:
        print(f"B3_MODULE_CLOSURE FAIL — {MC_LIB} carries no claim")
        ok = False
    else:
        gates = [GATE_FIELD_RE.search(b) for _, b in lib_claims]
        fails = [g.group(1) for g in gates if g and "gate_fail" in g.group(1)]
        if not fails:
            print(f"B3_MODULE_CLOSURE FAIL — {MC_LIB}'s claim is not bound to an "
                  f"always-failing gate, so the probe would not be decisive")
            ok = False

    lib_module = Path(MC_LIB).stem
    if f"use {lib_module}::" not in main:
        print(f"B3_MODULE_CLOSURE FAIL — {MC_MAIN} does not import {lib_module}")
        ok = False

    if not claims_in(MC_MAIN):
        print(f"B3_MODULE_CLOSURE FAIL — {MC_MAIN} carries no claim of its own "
              f"(needed to prove verification ran at all)")
        ok = False

    print(f"B3_MODULE_CLOSURE {'PASS' if ok else 'FAIL'} — probe is decisive: "
          f"importer compiling clean ⇒ imported claim never executed")
    print(f"B3_MODULE_CLOSURE   recorded outcome: MODULE_CLOSURE_PASSES "
          f"(measured 2026-08-01, R29: VERIFY_CLAIMS_SCOPE modules=2, "
          f"CLAIM_FAIL mcl_library_claim_that_is_false, VERIFY_CLAIMS_FALSIFIED "
          f"fail=1, no ELF). Supersedes MODULE_CLOSURE_BLOCKS (measured "
          f"2026-07-26: VERIFY_CLAIMS_OK pass=1, ELF emitted, imported false "
          f"claim invisible) — a true reading of the compiler as it then stood, "
          f"changed by R29 rather than corrected. Re-measure with the gate's "
          f"compile arm — this clause checks the probe's shape, not the run.")
    return ok


# ---------------------------------------------------------------- B4


def clause_b4(bound: list[tuple[str, str]]) -> bool:
    bound_gates = {g for _, g in bound}
    violations = sorted(bound_gates & set(OVER_BUDGET_GATES))
    ok = not violations
    for v in violations:
        print(f"  bound gate exceeds the {EXECUTOR_BUDGET_MS} ms executor budget: {v}")
    print(f"B4_TIMEOUT_BUDGET {'PASS' if ok else 'FAIL'} — "
          f"{len(OVER_BUDGET_GATES)} gates known over budget, none bound")
    return ok


# ---------------------------------------------------------------- B5


def clause_b5(bound: list[tuple[str, str]]) -> bool:
    """No bound gate may mutate the working tree.

    Two halves: the known-non-hermetic list (measured by the probe in the spec),
    and a static scan for writes under results/ or .sounio/ that catches a gate
    added later without re-running the probe.
    """
    bound_gates = {g for _, g in bound}
    ok = True

    known = sorted(bound_gates & set(NON_HERMETIC_GATES))
    for g in known:
        print(f"  bound gate is known non-hermetic: {g}")
        ok = False

    for gate in sorted(bound_gates - set(known)):
        hits = sorted(set(TREE_WRITE_RE.findall(read(gate))))
        if hits:
            print(f"  bound gate references tree-write paths {hits}: {gate}")
            ok = False

    if ok:
        print(f"B5_HERMETIC PASS — {len(NON_HERMETIC_GATES)} gates known to mutate "
              f"the tree, none bound; static scan of {len(bound_gates)} bound gates clean")
    else:
        print(f"B5_HERMETIC FAIL — a bound gate mutates the working tree "
              f"(see above); binding it would make every compile dirty the repo")
    return ok


# ---------------------------------------------------------------- main


def main() -> int:
    print("SELF-FALSIFYING COMPILATION R1 — corpus binding")
    print("=" * 72)

    b1, bound = clause_b1()
    print()
    b2 = clause_b2(bound)
    print()
    b3 = clause_b3()
    print()
    b4 = clause_b4(bound)
    print()
    b5 = clause_b5(bound)
    print()

    print("=" * 72)
    if not (b1 and b2 and b3 and b4 and b5):
        print("SELF_FALSIFYING_R1_VERDICT INCOMPLETE")
        return 1

    # The token carries the bound COUNT only. An earlier draft of this rung's
    # verdict form embedded the gate-population denominator (BOUND_N_OF_294);
    # that denominator moves whenever any gate is added, so the token would
    # have drifted without the claim changing — the very sub-token failure this
    # line documents. Corrected here and in R0's spec §5.
    # Flipped by R29 (2026-08-01): the closure walk landed, so the wall this
    # token recorded is gone. The token tracks the measurement; it is never
    # edited to keep a gate green.
    token = f"BOUND_{len(bound)}__MODULE_CLOSURE_PASSES"
    print(f"  claims bound to real gates : {len(bound)}")
    print(f"  module closure             : PASSES (imported claims execute — R29)")
    print(f"  gates excluded, over budget: {len(OVER_BUDGET_GATES)}")
    print(f"  gates excluded, non-hermetic: {len(NON_HERMETIC_GATES)}")
    print(f"SELF_FALSIFYING_R1_VERDICT {token}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
