# Phase D campaign — scalable verified search for χ(plane) ≥ 6

**Honest header:** no 6-chromatic unit-distance graph is known; brute enlargement of the
de Grey ring is *known* to stay 5-colourable (Polymath16, 2018–2021). This stands up the
**discovery engine** and runs a sound sweep; the expected (and obtained) result is a
**negative**. Finding χ ≥ 6 would be a landmark and is not promised.

## Integrity correction to the earlier Phase D probe
The first Phase D probe (`heule510_phaseD_search.py`, commit `22d9747e9`) detected edges
with a `round()`-based spatial hash, which **misses unit pairs near cell boundaries**. On
the 2600-vertex closure it reported 11545 edges; the **complete** edge set is **11553**
(8 edges missed). Missing edges only make colouring *easier*, so that "5-colourable"
result was **unsound** (it tested a subgraph). It is superseded here.

## Corrected engine (`scripts/research/heule510_phaseD_campaign.py`)
- **Complete exact edges.** `floor`-based spatial hash, cell size 1, 3×3 neighbourhood —
  provably captures *every* pair at distance ≤ 1 (since `‖Δ‖²=1 ⇒ |Δx|,|Δy|≤1 ⇒ floors
  differ by ≤1`), each candidate verified by exact `‖·‖²=1` in ℚ(√3,√5,√11). Validated:
  the move-translate edge set (guaranteed-real) is a subset, and the floor-grid catches the
  7 extra non-move-direction edges it lacks.
- **Closure generator** under the 72 certified unit-edge directions, radius/size capped,
  exact dedup.
- **Oracle:** SAT 5-colourability (bundled CaDiCaL). **SAT ⇒ 5-colourable** (negative);
  **UNSAT ⇒ χ ≥ 6** — on a hit the CNF is dumped and verified by drat-trim + `bv_decide`
  (the Phase C/C.1 pipeline), never trusted raw.

## Sweep result (complete edges)
Confirmed: size **2600 / 11553** edges → **5-colourable** (sound; corrects the buggy 11545
figure). Only sizes whose run log actually reports `5col=True` are claimed — no size is
assumed (the earlier 6000 over-claim taught that). A larger sweep (5k → 25k) runs in the
background and appends each completed size to the log; any `5col=False` triggers immediate
drat-trim + `bv_decide` verification before any claim.

**Key engineering finding — the bottleneck is exact edge detection, not SAT.** At 2600
vertices the complete-edge step took **79 s** while the 5-colourability SAT solve took **<1 s**.
The floor-grid is quadratic in cell occupancy, and the de Grey graph is *dense* near the
origin (cells hold many points), so edge detection — not colouring — is what blocks scale.
A real campaign must first replace it with a structure-aware method (move-translate bulk +
targeted non-move check, a k-d tree, or exploiting the lattice). This is concrete,
bounded engineering and is the first thing the campaign needs before more compute helps.

## What a genuine campaign still needs (and a session cannot supply)
- **Sustained compute** at Polymath scale (10⁴–10⁶ vertices) and beyond brute enlargement.
- **ML-guided generation** (FunSearch-style): a generator proposing candidate vertex
  families / gluings / non-de-Grey constructions, scored by *this* evaluator, kept when
  they raise the SAT lower bound. This engine is the evaluator such a loop calls; the
  generator + compute budget is the missing, expensive half.
- Quite possibly a genuinely new geometric idea — χ ≥ 6 has resisted enlargement.

**Status:** engine stood up, sound; sweep negative. χ ≥ 6 / Hadwiger–Nelson open.
