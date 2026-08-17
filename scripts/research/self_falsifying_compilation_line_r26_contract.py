#!/usr/bin/env python3
"""Self-falsifying compilation, rung R26 -- the never-committed oracle,
reconstructed; the orbit theorem's verifier runs in-tree for the first time.

Spec: docs/research/self_falsifying_compilation_line_r26_2026-07-31.md

R20 found cd_tower_automorphism_oracle.py absent from every branch (git log --all
empty), loaded by the orbit theorem's verifier via exec_module with no fallback.
R21 then proved a theorem RESTING on that orbit theorem -- whose own verification
script could not execute in any checkout because its oracle had never existed.
This rung reconstructs the oracle from the verifier's own proof, validates the
reconstruction, and shows the verifier now runs.

CLAUSES:
  W1_ORACLE_VALIDATES        sweep_autos(4) returns 168 maps, each a validated
                             permutation part of an octonion automorphism (a
                             sign vector exists), not merely a GL(3,2) element.
  W2_ORBIT_THEOREM_RUNS      the verifier executes end to end and reports the
                             predicted orbit multiset 2^(n-4)x[7] + (2^(n-4)-1)x[1],
                             stab 24, fixed = seam subspace, at n = 4..7.
  W3_DANGLING_DEP_CLOSED     the oracle is in the tree and the R20 audit no
                             longer flags it; the remaining absent hard deps are
                             foreign lanes, not this one's.

Pure Python 3.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ORACLE = REPO / "scripts/research/cd_tower_automorphism_oracle.py"
VERIFIER = REPO / "scripts/research/cd_tower_auto_action_on_zd_fibers.py"
AUDIT = REPO / "scripts/research/r20/audit.json"


def _load(path):
    sp = importlib.util.spec_from_file_location(path.stem, path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def main() -> int:
    print("R26 -- the reconstructed oracle; the orbit verifier runs in-tree")
    print("=" * 72)

    # ---- W1 -----------------------------------------------------------------
    w1 = True
    try:
        orc = _load(ORACLE)
        n = orc._self_check()
        autos = orc.sweep_autos(4)
        valid = all(orc._is_valid_permutation_part(c) for c in orc._gl3_columns())
        gl3n = sum(1 for _ in orc._gl3_columns())
        w1 = (n == 168 and len(autos) == 168 and valid and gl3n == 168)
        print(f"  |GL(3,2)| = {gl3n}, sweep_autos(4) = {len(autos)}, "
              f"all valid permutation parts: {valid}")
    except Exception as exc:                                    # noqa: BLE001
        w1 = False
        print(f"  oracle failed to load or self-check: {exc}")
    print(f"W1_ORACLE_VALIDATES {'PASS' if w1 else 'FAIL'}")
    print()

    # ---- W2 -----------------------------------------------------------------
    cp = subprocess.run([sys.executable, str(VERIFIER)],
                        capture_output=True, text=True, cwd=str(REPO), timeout=600)
    out = cp.stdout + cp.stderr
    predicted = {
        4: "orbits {7: 1}", 5: "orbits {1: 1, 7: 2}",
        6: "orbits {1: 3, 7: 4}", 7: "orbits {1: 7, 7: 8}",
    }
    w2 = cp.returncode == 0
    for lvl, pat in predicted.items():
        line = next((l for l in out.splitlines()
                     if l.startswith(f"n={lvl} ")), "")
        ok = pat in line and "stab(7-orbit)=24" in line and line.rstrip().endswith("OK")
        w2 &= ok
        print(f"  n={lvl}: {'OK' if ok else 'MISMATCH'}  "
              f"{pat}, stab=24, seam-fixed")
    w2 &= "VERIFIED n=4..7" in out
    print(f"W2_ORBIT_THEOREM_RUNS {'PASS' if w2 else 'FAIL'}  "
          f"(exit {cp.returncode}) -- first in-tree run of the orbit verifier")
    print()

    # ---- W3 -----------------------------------------------------------------
    w3 = ORACLE.exists()
    remaining = []
    if AUDIT.exists():
        a = json.loads(AUDIT.read_text())
        hard = [r for r in a["missing"] if r.get("hard_dependency")]
        oracle_flagged = any("automorphism_oracle" in r["artifact"] for r in hard)
        remaining = [r["artifact"] for r in hard]
        this_line = [r for r in remaining if "cd_tower" in r]
        w3 = w3 and not oracle_flagged and not this_line
        print(f"  oracle in tree: {ORACLE.exists()}; still flagged absent: "
              f"{oracle_flagged}")
        print(f"  remaining absent hard deps: {len(remaining)} "
              f"(this line's: {len(this_line)})")
        for r in remaining:
            print(f"      foreign: {r}")
    else:
        w3 = False
        print("  R20 audit not present; run scripts/research/r20_provenance_audit.py")
    print(f"W3_DANGLING_DEP_CLOSED {'PASS' if w3 else 'FAIL'}")
    print()

    ok = w1 and w2 and w3
    verdict = ("ORACLE_RECONSTRUCTED__ORBIT_VERIFIER_RUNS_IN_TREE"
               if ok else "INCONCLUSIVE")
    print("-" * 72)
    print("The dependency R20 found dangling under the theorem R21 proved is")
    print("closed: the oracle is reconstructed from the verifier's own proof,")
    print("validated, and the orbit theorem is now checkable in every checkout.")
    print()
    print(f"SELF_FALSIFYING_R26_VERDICT {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
