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

## Edge-detection bottleneck — FIXED
The first cut did exact field arithmetic on *every* candidate pair in dense cells: **79 s**
at 2600 vertices. Fix: a **cheap float screen before the exact check** — a true unit pair
has `float ‖Δ‖² = 1 ± ~1e-12`, so screening `|float‖Δ‖² − 1| < 1e-6` (tolerance ≫ float
error, so **no true edge is skipped**) sends only near-unit pairs to the exact `‖·‖²=1`
verifier. Validated: identical edge set to the all-exact method (11553 @ 2600), and the
exact verifier is now called only ~once per real edge.

Result: **79.2 s → 0.61 s at 2600 (≈130×)**; the cost is now the SAT solve, as it should be.

## Sweep result (complete edges, fast)
Confirmed `5col=True` (sound, complete edges) — only sizes the run log actually reports:

| vertices | edges | edge time | SAT time | 5-colourable |
|---|---|---|---|---|
| 2 600 | 11 553 | 1 s | <1 s | yes |
| 5 000 | 21 313 | 2 s | <1 s | yes |
| 10 000 | 56 925 | 6 s | 24 s | yes |

Larger sizes (20k, 40k, 70k) continue in the background; each is appended only when the log
confirms it. Any `5col=False` triggers immediate drat-trim + `bv_decide` verification before
any claim. With edges fixed, **SAT 5-colourability is now the scaling cost** (24 s at 10k and
growing) — the next bottleneck a real campaign hits.

## What a genuine campaign still needs (and a session cannot supply)
- **Sustained compute** at Polymath scale (10⁴–10⁶ vertices) and beyond brute enlargement.
- **ML-guided generation** (FunSearch-style): a generator proposing candidate vertex
  families / gluings / non-de-Grey constructions, scored by *this* evaluator, kept when
  they raise the SAT lower bound. This engine is the evaluator such a loop calls; the
  generator + compute budget is the missing, expensive half.
- Quite possibly a genuinely new geometric idea — χ ≥ 6 has resisted enlargement.

**Status:** engine stood up, sound; sweep negative. χ ≥ 6 / Hadwiger–Nelson open.
