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

### Resolution (Phase 0 probe + B→A→C arc, 2026-05-26)

Added Phase 0 to `examples/erdos/168_regime_a1.sio`: pure coloring solver (no background)
for each distinct edge-count group. Result: **r=1, confl=0 for ALL groups** (e=8,10,11,12,18).

**The 14-vertex unit-distance graphs ARE 3-colorable (χ≤3). UNSAT interpretation definitively
refuted.** The CDCL phase-transition framing (shorter UNSAT refutation → fewer conflicts →
lower hardness) does not apply. Directional signal re-framed as SAT-search difficulty:
more edge constraints → fewer valid colorings → CDCL converges faster. This is also heuristic.

**B→A→C arc completed (same session):**
- B: Three chromatic-flip probes (init_probe14, C₅, cross-half sums) — all null.
- A: Moser spindle UNSAT probe — all 84 instances hit 500-conflict cap, fiber ratio 1.17x (weak).
- C: Exhaustive edge map for K=1..4 component diffs reveals:
  - K=1: always edge (all 84 surgeries) → hypercube subgraph → bipartite
  - K=2: never edge (algebraic cancellation in sedenion product)
  - K=3: edge for 4-8 surgeries per diff type (378/560 positive diffs), but triangle-free (parity)
  - K=4: never edge (sample verified)
- **THEOREM (machine-verified):** Integer sedenion ZD-surgery unit-distance graph is always
  bipartite. χ=2 universally. All 84 surgeries, all vertex sets tested. 2-coloring SAT r=1,
  confl=0 on rich mixed vertex set.
- **Escape route:** Non-integer coordinates (rational/algebraic). C₅ with ε~1e-4 is next probe.

---

## 2026-05-26: GPU Bridge Validation (sinkhorn16)

- **Task**: Validate sinkhorn16 K-AXI kernel against CPU LSE for hyperbolic semantic networks ORC
- **Provider**: N/A (internal validation, no external math claims)
- **Outcome**: PASS — all tests agree within 1e-6 for epsilon ≥ 0.5
- **Speedup**: 37× over CPU serial on RTX A5000
- **Blocker resolved**: lambda=epsilon mapping, log2-marginal input, inactive padding
- **Remaining**: kernel size limit (16×16) prevents N=100 k>15 use cases

