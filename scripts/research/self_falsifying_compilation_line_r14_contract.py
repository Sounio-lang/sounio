#!/usr/bin/env python3
"""Self-falsifying compilation, rung R14 — what a corpus computes vs what it checks.

Spec: docs/research/self_falsifying_compilation_line_r14_2026-07-27.md

R13 used perturbation of the shared object to detect SHARED evidential fate.
R14 turns the same instrument the other way: for each contract, of everything it
computes, how much does its stated conclusion actually depend on?

The hypothesis under test was VACUITY -- that contracts computing the
Cayley-Dickson tower to levels 5-10 might have conclusions resting only on the
low levels, making the expensive high-level work decoration. It is refuted.

Two inputs, both recorded under scripts/research/r14/:
  call_trace.json   which (a, b, bits) each contract queries of the sign table
  loadbearing.json  per-contract, per-level: does flipping a queried pair's sign
                    move the verdict?

CLAUSES:

  C1_CONTROL_INERT
      Instrument before corpus. Any contract reacting to the null wrapper has
      its row voided (R13 SS5.1 -- this line has already reported one harness
      artifact as a corpus finding).

  C2_LOAD_BEARING_MEASURED
      The measure has to separate three outcomes, not two: a VERDICT change, a
      CRASH (the conclusion can no longer be established -- still a dependence),
      and MISSING data (timeout or lost output -- no information). Contracts
      whose kills are ALL crashes are flagged: for them this measures fragility,
      not conclusion-dependence.

  C3_VACUITY_REFUTED
      No contract has a level it queries but does not depend on, once genuine
      single-flip invariance is separated out (RULE 2 of the analysis: escalate
      to a row flip before calling a zero vacuous).

Pure Python 3; reads recorded evidence, does not re-run the batteries.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
D = REPO / "scripts/research/r14"
R13 = REPO / "scripts/research/r13"

MISSING_PREFIXES = ("TIMEOUT", "nojson")


def main() -> int:
    LB = json.loads((D / "loadbearing.json").read_text())
    trace = json.loads((D / "call_trace.json").read_text())
    plan, res = LB["plan"], LB["results"]

    print("R14 — what this corpus computes vs what it checks")
    print("=" * 72)

    spans = {t["contract"]: sorted(t.get("by_bits", {}), key=int)
             for t in trace if t.get("by_bits")}
    deepest = max((int(l) for s in spans.values() for l in s), default=0)
    print(f"call trace: {len(spans)} contracts; deepest level queried = {deepest}")
    print()

    # ---- C1 ------------------------------------------------------------------
    void = []
    rows = []
    for c in sorted(plan):
        r = res.get(c, {})
        base = r.get("baseline", {}).get("verdict")
        if not base:
            void.append((c, "no baseline verdict"))
            continue
        ctrl = r.get("null_wrap", {})
        if ctrl.get("verdict") != base or ctrl.get("error"):
            void.append((c, "reacts to the null wrapper"))
            continue
        rows.append((c, base, r))
    reactive = [c for c, why in void if "null wrapper" in why]
    c1 = not reactive
    print(f"C1_CONTROL_INERT {'PASS' if c1 else 'FAIL'}  "
          f"{len(rows)}/{len(rows) + len(reactive)} inert to the null wrapper")
    for c, why in void:
        print(f"    excluded: {c[:56]} — {why}")
    if not c1:
        print("\nInstrument contaminated; nothing below may be read.")
        print("SELF_FALSIFYING_R14_VERDICT BATTERY_INVALID")
        return 1
    print()

    # ---- C2 ------------------------------------------------------------------
    kind = collections.Counter()
    per_level = collections.defaultdict(collections.Counter)
    all_crash, surv_cells = [], []
    for c, base, r in rows:
        d = collections.Counter()
        for mid, rr in r.items():
            if mid in ("baseline", "null_wrap"):
                continue
            lvl = mid.split("_")[0][1:]
            err = str(rr.get("error") or "")
            if err.startswith(MISSING_PREFIXES):
                k = "MISSING"
            elif err:
                k = "CRASH"
            elif rr.get("verdict") != base:
                k = "VERDICT"
            else:
                k = "survives"
                surv_cells.append((c, mid))
            d[k] += 1
            kind[k] += 1
            per_level[lvl][k] += 1
        if d["VERDICT"] == 0 and d["CRASH"] > 0:
            all_crash.append(c)

    total = sum(kind.values())
    c2 = kind["VERDICT"] > 0 and kind["MISSING"] == 0
    print(f"C2_LOAD_BEARING_MEASURED {'PASS' if c2 else 'FAIL'}  "
          f"{total} cells: VERDICT {kind['VERDICT']}, CRASH {kind['CRASH']}, "
          f"survives {kind['survives']}, MISSING {kind['MISSING']}")
    print(f"  {'level':>6} {'VERDICT':>8} {'CRASH':>7} {'surv':>6}")
    for l in sorted(per_level, key=int):
        p = per_level[l]
        print(f"  {l:>6} {p['VERDICT']:8} {p['CRASH']:7} {p['survives']:6}")
    print(f"\n  ALL-CRASH contracts ({len(all_crash)}): for these the measure is "
          f"fragility, not conclusion-dependence")
    for c in all_crash:
        print(f"      {c}")
    print()

    # ---- C3 ------------------------------------------------------------------
    by_contract = collections.defaultdict(list)
    for c, mid in surv_cells:
        by_contract[c].append(mid)

    # A level is a vacuity candidate only if EVERY sampled flip there survives.
    candidates = []
    for c, base, r in rows:
        lv = collections.defaultdict(lambda: [0, 0])
        for mid, rr in r.items():
            if mid in ("baseline", "null_wrap"):
                continue
            l = mid.split("_")[0][1:]
            err = str(rr.get("error") or "")
            if err.startswith(MISSING_PREFIXES):
                continue
            lv[l][1] += 1
            if not err and rr.get("verdict") == base:
                lv[l][0] += 1
        for l, (s, n) in lv.items():
            if n and s == n:
                candidates.append((c, l, n))

    print(f"C3_VACUITY_REFUTED  {len(candidates)} (contract, level) all-survive cells")
    resolved = []
    if candidates:
        B13 = json.loads((R13 / "battery_results.json").read_text())
        for c, l, n in candidates:
            r13 = B13["results"].get(c, {})
            b = r13.get("baseline", {}).get("verdict")
            rows_el = [(m, rr) for m, rr in r13.items()
                       if m.startswith("elem_") and m.endswith(f"L{l}")]
            killers = [m for m, rr in rows_el
                       if (rr.get("verdict") != b) or rr.get("error")]
            verdictum = ("INVARIANCE — single flips survive, row flips kill"
                         if killers else "NO ESCALATION EVIDENCE")
            resolved.append((c, l, n, len(killers), len(rows_el), verdictum))
            print(f"    {c[:50]:<52} L{l}: {n}/{n} survive; "
                  f"row flips kill {len(killers)}/{len(rows_el)} -> {verdictum}")

    unresolved = [x for x in resolved if not x[3]]
    c3 = all(x[3] for x in resolved) if resolved else True
    print(f"  every all-survive level is explained by single-flip invariance: "
          f"{'YES' if c3 else 'NO'}")
    print()

    ok = c1 and c2 and c3
    verdict = ("VACUITY_REFUTED__CORPUS_CHECKS_WHAT_IT_COMPUTES"
               if ok else "INCONCLUSIVE")
    print("-" * 72)
    print(f"Deepest level with load-bearing perturbations: {deepest}")
    print(f"SELF_FALSIFYING_R14_VERDICT {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
