# Phase D — search for a 6-chromatic unit-distance graph (χ(plane) ≥ 6)

**This is the open frontier and the long-odds gamble**, stated plainly: no 6-chromatic
unit-distance graph is known; de Grey, Heule, Parts and the entire Polymath16 effort
(2018–2021) searched hard and never reached χ ≥ 6; χ(plane) might genuinely be 5. A
session will not settle this. The deliverable here is the **verified discovery engine**
plus an **honest bounded negative** — not a claim of progress on #508.

## The search (genuine, not theater)
HeuleGraph510 is a 5-chromatic *subgraph* of a larger ambient unit-distance graph in
ℚ(√3,√5,√11). The real question (the de Grey→6 strategy): does a *larger finite chunk* of
that same graph become non-5-colourable? `scripts/research/heule510_phaseD_search.py`:

1. extracts the 72 distinct **unit-edge direction vectors** of the certified
   HeuleGraph510 (the exact "moves" of the ambient graph);
2. grows the vertex set by exact closure under those moves (radius/vertex capped),
   deduplicated exactly in the field;
3. computes the exact unit-distance edges of the chunk (numeric grid prefilter, then
   **exact** `‖·‖²=1` verification in ℚ(√3,√5,√11) — no float decisions on edges);
4. SAT-tests 5-colourability (PySAT + Cadical). **UNSAT ⇒ χ ≥ 6** (and would be
   drat-trim-verified, like Phase C); **SAT ⇒ 5-colourable** (negative).

## Result (bounded negative)
Closure to **2600 vertices / 11545 exact unit edges** (radius 6) → **5-colourable**.
No 6-chromatic graph in this region. (A larger 6000-vertex / radius-9 sweep built the
closure but its exact edge computation did **not** finish in the session time budget, so
there is **no colourability verdict at 6000** — the substantive, completed negative is the
2600-vertex result above; the 6000 attempt only confirms the closure scales and that the
exact all-pairs edge step is the bottleneck to optimise.) The 2600 outcome is the expected
one and is consistent with every published result: the ambient de Grey graph's finite
chunks remain 5-colourable.

## What a real attempt requires (and a session cannot do)
- **Sustained compute** over far larger chunks / many seed structures (Polymath tested
  graphs orders of magnitude larger).
- **ML-guided generation** (FunSearch-style: propose candidate vertex families /
  gluings, GPU-evaluate, keep what raises the SAT lower bound) — the discovery half of
  the program. This harness is the evaluator/certifier that such a loop would call; the
  generator + compute budget is the missing, expensive piece.
- A genuinely different geometric idea may simply be required; χ ≥ 6 has resisted brute
  enlargement.

## Honest status
- Phase A/B/C: done (spindle exact; HeuleGraph510 exact unit-distance kernel; χ=5).
- **Phase D: discovery engine built; bounded search negative.** χ ≥ 6 / Hadwiger–Nelson
  remains open. Finding a 6-chromatic graph would be a landmark; this is not it.
