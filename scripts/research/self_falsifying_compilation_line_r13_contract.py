#!/usr/bin/env python3
"""Self-falsifying compilation, rung R13 — co-sensitivity vs structural distance.

Spec: docs/research/self_falsifying_compilation_line_r13_2026-07-27.md

R12 established, from someone else's study, that structural diversity does not
predict failure independence, and narrowed C6 to a one-sided reading. R13 asks
whether that holds for THIS corpus's own internal checks, and answers it by
measurement rather than by transfer.

THE MEASUREMENT. Perturb the shared mathematical object — flip the
Cayley–Dickson sign on a targeted base pair at a targeted level — and record
which contracts change their verdict token. Both derivations in this corpus
(`cds`, iterative; `cd_sigma`, recursive; R6 similarity 0.507) take (a, b, bits)
and return ±1, so the SAME conceptual perturbation crosses both. That is what
makes the experiment non-tautological: mutating the *source* of `cds` could only
ever reach `cds` users, which would re-derive R6's structural partition by
construction.

Two contracts are CO-SENSITIVE when they are killed by the same perturbations.
The question is whether co-sensitivity tracks structural distance. Answer: no.

CLAUSES:

  C1_CONTROL_INERT
      The null-wrap control installs identical wrapper machinery with a
      condition that can never fire. Every usable contract must be inert to it.
      This is checked FIRST and fails the rung outright, because the first
      battery was contaminated by exactly this failure (§5.1 of the spec) and
      reported it as a corpus finding.

  C2_BATTERY_DISCRIMINATES
      A mutant is informative if it kills >10% and <90% of usable contracts.
      Pre-registered floor: at least 8, else the battery is degenerate and no
      partition may be read from it.

  C3_IDENTICAL_FATE_BELOW_THRESHOLD
      The finding: pairs whose kill sets are IDENTICAL over all 36 perturbations
      while their R6 structural similarity is below the 0.90 independence
      threshold. R6 calls such pairs independent evidence. They are not: they
      share evidential fate exactly.

Reads the recorded battery; does not re-run it (that needs a large machine —
see the spec's Reproduce section). Pure Python 3.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
D = REPO / "scripts/research/r13"

CONTROL = "null_wrap"
INFORMATIVE_FLOOR = 8          # pre-registered
R6_THRESHOLD = 0.90            # R6's own DUP_THRESHOLD, not re-tuned here


def load():
    B = json.loads((D / "battery_results.json").read_text())
    S = json.loads((D / "structural_similarity.json").read_text())
    fam = {c["file"]: c["fn"] for c in json.loads((D / "manifest.json").read_text())}
    return B, S, fam


def main() -> int:
    B, S, fam = load()
    res = B["results"]
    muts = [m for m in B["mutants"] if m not in ("baseline", CONTROL)]
    base = {c: res[c].get("baseline", {}).get("verdict") for c in res}
    usable = [c for c in res if base[c]]

    def killed(c, m):
        # A CRASH IS A KILL (the conclusion can no longer be established); a
        # TIMEOUT or a lost output file is MISSING DATA and must not be scored
        # as one. Added in R14; the 21 pairs are identical under either
        # convention, which was checked rather than assumed.
        r = res[c].get(m, {})
        err = str(r.get("error") or "")
        if err.startswith(("TIMEOUT", "nojson")):
            return False
        return (r.get("verdict") != base[c]) or bool(err)

    print("R13 — does co-sensitivity track structural distance?")
    print("=" * 72)
    print(f"contracts probed {len(res)}; usable {len(usable)}; "
          f"mutants {len(muts)} + baseline + control")
    print()

    # ---- C1: the instrument, before the corpus -------------------------------
    reactive = [c for c in usable
                if res[c].get(CONTROL, {}).get("verdict") != base[c]
                or res[c].get(CONTROL, {}).get("error")]
    c1 = not reactive
    for c in reactive:
        print(f"  [REACTS TO HARNESS] {c}")
    print(f"C1_CONTROL_INERT {'PASS' if c1 else 'FAIL'}  "
          f"{len(usable) - len(reactive)}/{len(usable)} inert to the null wrapper")
    if not c1:
        print("\nInstrument contaminated. Nothing below may be read.")
        print(f"SELF_FALSIFYING_R13_VERDICT BATTERY_INVALID")
        return 1
    print()

    # ---- C2: does the battery discriminate at all? ---------------------------
    infor = [m for m in muts
             if 0.10 < sum(killed(c, m) for c in usable) / len(usable) < 0.90]
    c2 = len(infor) >= INFORMATIVE_FLOOR
    print(f"C2_BATTERY_DISCRIMINATES {'PASS' if c2 else 'FAIL'}  "
          f"{len(infor)}/{len(muts)} mutants informative "
          f"(floor {INFORMATIVE_FLOOR}, pre-registered)")

    kills = {c: frozenset(m for m in muts if killed(c, m)) for c in usable}
    patterns = {}
    for c in usable:
        patterns.setdefault(kills[c], []).append(c)
    print(f"  {len(patterns)} distinct kill patterns over {len(usable)} contracts "
          f"— the resolution this measure actually has")
    print()

    # ---- C3: the finding -----------------------------------------------------
    ident = []
    for a, b in itertools.combinations(sorted(usable), 2):
        s = S.get(f"{a}|{b}", S.get(f"{b}|{a}"))
        if s is None or s >= R6_THRESHOLD:
            continue
        if kills[a] == kills[b]:
            ident.append((a, b, s))

    c3 = bool(ident)
    print(f"C3_IDENTICAL_FATE_BELOW_THRESHOLD {'PASS' if c3 else 'FAIL'}  "
          f"{len(ident)} pairs")
    if ident:
        lo = min(s for _, _, s in ident)
        hi = max(s for _, _, s in ident)
        cross = sum(fam[a] != fam[b] for a, b, _ in ident)
        print(f"  R6 structural similarity {lo:.3f}–{hi:.3f}, all below "
              f"{R6_THRESHOLD} => R6 calls every one INDEPENDENT evidence")
        print(f"  cross-derivation (cds vs cd_sigma): {cross}/{len(ident)}")
        for a, b, s in sorted(ident, key=lambda t: t[2])[:4]:
            print(f"    sim {s:.3f}  {len(kills[a])} kills, identical")
            print(f"        {a}")
            print(f"        {b}")

    # ---- the direction, reported whichever way it falls ----------------------
    def jac(x, y):
        u = kills[x] | kills[y]
        return len(kills[x] & kills[y]) / len(u) if u else 1.0

    lo_s, hi_s = [], []
    for a, b in itertools.combinations(sorted(usable), 2):
        s = S.get(f"{a}|{b}", S.get(f"{b}|{a}"))
        if s is None:
            continue
        (lo_s if s < R6_THRESHOLD else hi_s).append(jac(a, b))
    if lo_s and hi_s:
        ml, mh = sum(lo_s) / len(lo_s), sum(hi_s) / len(hi_s)
        print()
        print(f"  mean kill-set agreement, R6-INDEPENDENT pairs (n={len(lo_s)}): {ml:.3f}")
        print(f"  mean kill-set agreement, R6-SHARED      pairs (n={len(hi_s)}): {mh:.3f}")
        print(f"  gap {mh - ml:+.3f} — a measure predicting shared fate would be "
              f"strongly positive")

    ok = c1 and c2 and c3
    verdict = ("STRUCTURAL_INDEPENDENCE_DOES_NOT_IMPLY_INDEPENDENT_FATE"
               if ok else "INCONCLUSIVE")
    print()
    print("-" * 72)
    print(f"SELF_FALSIFYING_R13_VERDICT {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
