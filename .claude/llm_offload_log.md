# LLM Offload Log

## 2026-05-26: A1 probe math-review (168_regime_a1.sio)

- **Task**: math-review
- **Provider**: xai / **Model**: grok-4.3
- **Tokens**: prompt=1576, completion=270 (reasoning=513), total=2359
- **Cost**: $0.0379 (37931000 usd_ticks)
- **Target**: Mathematical claims in `examples/erdos/168_regime_a1.sio` and `docs/research/locus-coeruleus-surgical-controller-sounio-note.md §5(c)`

### Verdict

```
[OK]         42 vars from 14×3 encoding — correct
[OK]         56 coloring-base clauses (14×3 + 42×2) — correct
[OK]         151 + 3e formula and five ratios — arithmetic holds
[OVERREACH]  e≥9 → UNSAT: no proof/citation that graphs are non-3-colorable
[OVERREACH]  above-threshold → shorter refutation: known only for uniform random 3-SAT; structured clauses + LCG background invalidate extrapolation
[TIGHTENABLE] regime_recent_hardness tracks conflict count: non-standard metric, unvalidated in probe
[TIGHTENABLE] "CONFIRMED" at margin 0.01 (0.06>0.05) with n=4 for e=18: statistically fragile
[WRONG]      "ZD surgery edge structure correlates with epistemic regime signal": rests on the two OVERREACH claims; not established at probe level
```

### Action required (original)

- §5(c) and A1 probe status header must be downgraded from "CONFIRMED" to "directional probe / math review flags two overreaches"
- UNSAT claim requires either: (a) cite χ>3 for specific 14-vertex unit-distance graphs, or (b) add runtime SAT/UNSAT check to the probe
- Phase-transition extrapolation must be flagged as heuristic only (not derived from mixed-formula theory)
- n=4 for e=18 is insufficient; note recommends denser surgery scan

### Resolution (Phase 0 + full 84-surgery Phase 1 run, 2026-05-26)

Added Phase 0 to `examples/erdos/168_regime_a1.sio`: pure coloring solver (no background)
for each distinct edge-count group. Result: **r=1, confl=0 for ALL groups** (e=8,10,11,12,18).

**The 14-vertex unit-distance graphs ARE 3-colorable (χ≤3). UNSAT interpretation definitively
refuted.** The CDCL phase-transition framing (shorter UNSAT refutation → fewer conflicts →
lower hardness) does not apply. Directional signal re-framed as SAT-search difficulty:
more edge constraints → fewer valid colorings → CDCL converges faster. This is also heuristic.

**Full 84-surgery Phase 1 run completed 2026-05-26** (first machine-verified run):
- Added `decision_limit` (default 100,000) to `SmtContext` to bound solver runtime
- Root cause of prior hang: stale binary (old smt.sio, not recompiled after struct change)
- All 84 surgeries completed in <1s total

Results:
```
e    mean_hard  n   col_r
8    0.45     40   1 (SAT)
10   0.44     32   1 (SAT)
11   0.51     4    1 (SAT)
12   0.55     4    1 (SAT)
18   0.50     4    1 (SAT)
```
**UNEXPECTED direction:** dense graphs (e=12) are HARDER than sparse (e=8). 
The original hypothesis (more edges → more constraints → shorter UNSAT refutation) is WRONG.
All instances are SAT; the solver is searching for a valid 3-coloring, not proving UNSAT.
The "hardness" signal reflects SAT-search difficulty, not UNSAT refutation speed.
Higher edge count → fewer valid colorings → solver backtracks more to find one = more conflicts.

**Primary finding:** ZD surgery edge density has a weak positive effect on SAT-search hardness
(0.55 at e=12 vs 0.44 at e=10) but the signal is small (0.11) and the sample sizes are small
(n=4 for e=11/12/18). This is a directional observation, not a statistically robust finding.

---

## 2026-05-26: GPU Bridge Validation (sinkhorn16)

- **Task**: Validate sinkhorn16 K-AXI kernel against CPU LSE for hyperbolic semantic networks ORC
- **Provider**: N/A (internal validation, no external math claims)
- **Outcome**: PASS — all tests agree within 1e-6 for epsilon ≥ 0.5
- **Speedup**: 37× over CPU serial on RTX A5000
- **Blocker resolved**: lambda=epsilon mapping, log2-marginal input, inactive padding
- **Remaining**: kernel size limit (16×16) prevents N=100 k>15 use cases

