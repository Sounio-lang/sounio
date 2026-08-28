# Temporal discrete-curvature of the individual affect network — real MDD case (Kossakowski 2017)

**Date:** 2026-05-27 · **Single-case method demonstration (n=1 patient), not a population claim.**
Pipeline: `scripts/research/esm_affect_network_fixture.py` (I/O) +
`examples/hyperbolic_semantic_networks/affect_net_curvature.sio` (all geometry/stats, native Sounio).
Extends today's `affect_network_orc.sio` (4 independent per-node temporal self-transitions) to the TRUE
between-item network — the "full network ORC over all mood items" named as future work.

## Construction
Real Kossakowski (2017) n=1 ESM data: 1476 beeps, 12 mood items, 5 phases (phase 3→4 = medication
reduction, the depressive-transition onset). Sliding window over the series; within each window, the
12-item **Pearson correlation network** (edge if |r|≥0.25, weight=|r|). Per window we compute:
- **Forman-Ricci curvature** (weighted, Sreejith et al.; node weights 1) — NEW tool, exact, O(edges·deg),
  scales to full networks, complementary to Ollivier-Ricci. Mean over edges = network curvature.
- **λ₂** (algebraic connectivity, the original-KEC coherence) via Jacobi (verified K₁₂/C₈/K₂₀).
Compared against classical critical-slowing-down (CSD) baselines: within-window variance, lag-1 AC.

## Result — curvature anti-tracks CSD, robust
| statistic | overlapping (W=50,step=25, n=58) | **non-overlapping (step=50, n=29)** |
|---|---|---|
| corr(Forman, lag-1 autocorrelation) | −0.453 | **−0.495** (p≈0.006) |
| corr(Forman, variance) | −0.325 | −0.349 |
| corr(Forman, AC) — λ₂≥0.01 windows only | −0.447 | **−0.550** |
| phase-4 Forman shift (baseline-phase-1/2 SD) | −0.32 SD | **−0.72 SD** |
| phase-4 λ₂ shift | +1.62 SD | +2.48 SD |

**Interpretation.** As the affect system slows (lag-1 AC and variance rise — the textbook CSD signature
of approaching a critical transition), the network's Forman-Ricci curvature **drops** (becomes more
negative / more hyperbolic). This is the geometric early-warning prediction, and it holds on real MDD
data. The relationship:
- **survives de-overlapping** (r −0.45→−0.50 at independent n=29) — not an autocorrelated-window artifact;
- **survives excluding near-disconnected windows** (λ₂<0.01; r −0.45→−0.55) — not driven by the phase-5
  decoupling;
- is directionally coherent with both classical CSD indicators (negative with variance AND AC).
Phase-4 (post-reduction depressive band) shows lower curvature and higher λ₂ vs the phase-1/2 baseline.

This is the opposite outcome to the DCT/KEC-α and induced-subgraph nulls — a genuine, robustness-checked
signal — because the substrate is right: ORC-style curvature on a real individual network that changes
over time, the original hyperbolic-semantic-networks instinct.

## Honest scope / limits
- **n=1.** This is *the* canonical MDD-EWS single case (well-precedented in the CSD literature); it is a
  single-case demonstration of the method, NOT evidence about depression in general.
- **Structural ≠ temporal.** What is measured here is the *structural* curvature of the within-window
  correlation network. It is NOT the *temporal-transition* curvature (per-node/joint window-to-window
  distribution shift) that `affect_network_orc.sio` and the CSD lineage compute. Both are legitimate;
  the temporal-transition full-network ORC remains untested and is the matched next comparison.
- One free knob: correlation threshold |r|≥0.25. λ₂ hits ~0 in phase-5 windows (network decouples) — real
  or threshold-sensitive; the headline survives excluding them but threshold-sensitivity is untested.
- Eigensolver self-verified each run (λ₂ = 12 / 0.586 / 20 on K₁₂/C₈/K₂₀).
- Significance at n=29 is nominal single-case; overlapping-window p-values are inflated and not used.

## Matched comparison: temporal-transition full-network ORC (the 4-node heir)
Ran the *other* object — the direct generalization of `affect_network_orc.sio` from 4 to all 12 items:
per consecutive window-pair, per-node temporal Ollivier-Ricci of the binarized marginal shift
(t→t+1) via the verified forward-AD Sinkhorn `orc2x2`, mean κ̄(t) over 12 nodes with GUM σ.
Files: `affect_temporal_orc_main.sio` + fixture `--mode temporal`. (W=50, step=50, 28 pairs.)

| statistic | temporal-transition κ̄ (12-node) | structural Forman (above) |
|---|---|---|
| phase-4 shift (baseline SD) | **+0.03 SD (flat)** | −0.72 SD |
| corr(·, lag-1 AC) | **+0.22** | −0.50 |
| corr(·, variance) | +0.28 | −0.35 |

**The temporal-transition curvature does NOT carry the EWS signal.** κ̄ is flat across the phase-4
depressive band (0.929→0.930) and co-moves *positively* (weakly) with the CSD indicators — opposite
sign to the structural curvature. The 4-node file's headline drop (κ̄ 0.82→0.59) **does not survive
expansion to all 12 mood items**: averaging the full item set washes it out, so that drop was specific
to the 4 cardinal-circumplex items (satisfi/enthus/down/lonely), not a property of the full affect
network's temporal geometry.

**Disambiguation:** the geometric early-warning signal on this case lives in the **structural curvature
of the correlation network** (how the connectivity geometry deforms), not in the **temporal-transition
curvature** (how fast per-node marginals move). Two distinct objects; only the former tracks CSD here.
This is why the matched comparison was run rather than assumed.

## Threshold robustness (structural Forman)
Sweep |r|≥thr ∈ {0.15..0.40}, non-overlapping: corr(Forman, AC1) stays in **[−0.41, −0.58]** (corr with
variance strengthens for sparser graphs, −0.27→−0.55); phase-4 drop always negative (−0.66 to −1.35 SD).
Not a threshold artifact.

## Exact-OT Ollivier-Ricci (the original instrument) — and a confound it exposes
Computed κ_OR(u,v)=1−W1(m_u,m_v)/d(u,v) per edge (weighted random-walk measure, idleness 0; hop-distance
metric; entropic Sinkhorn ε=0.05, 200 iters; native Sounio), mean over edges per window. File
`affect_orc_exact.sio`.

| | exact-OT Ollivier-Ricci | combinatorial Forman |
|---|---|---|
| corr with lag-1 AC | **+0.52** | −0.50 |
| corr with variance | +0.49 | −0.35 |
| corr(ORC, Forman) | **−0.89** | — |

**The two curvatures disagree in sign relative to CSD.** ORC *rises* as the classical fragility markers
rise; Forman *falls*. They are themselves strongly anti-correlated (−0.89). The most likely common driver
is **network density**: in high-CSD windows the mood items co-move, |r| edges proliferate → denser graph
→ higher Ollivier-Ricci (more neighbour overlap = positive curvature) but more-negative weighted Forman
(higher degrees). So the apparent "geometric early-warning" may be largely a **density proxy**, and the
*sign* of the EWS depends entirely on which curvature is used.

**This tempers the earlier headline.** The robust empirical fact is: a discrete-curvature summary of the
affect network co-varies with classical CSD on this case (|r|≈0.5 either curvature). Whether that is a
genuinely *geometric* early-warning beyond density was then tested directly.

### Density-confound test (Spearman + first-order partials) — VERDICT: reduces to density
Density covariates: per-window edge count and threshold-free **mean|r|** over the full correlation matrix.

| | value |
|---|---|
| premise corr(mean\|r\|, AC1) | **+0.70** |
| premise corr(n_edges, AC1) | +0.46 |
| Spearman Forman vs AC1 | −0.44 |
| Spearman ORC vs AC1 | +0.56 |
| partial(Forman, AC1 \| n_edges) | **+0.01** (collapses) |
| partial(Forman, AC1 \| mean\|r\|) | +0.31 (weakens, sign-flips) |
| partial(ORC, AC1 \| n_edges) | +0.36 |
| partial(ORC, AC1 \| mean\|r\|) | **−0.36** (sign-flips vs base) |

**Premise confirmed:** network density tracks CSD strongly (mean|r| vs AC1 = +0.70) — as the affect
system slows, mood items synchronize and the correlation network densifies. **Neither curvature retains a
stable partial** with AC1 once density is removed: Forman collapses to ~0 (edge-count control) or flips to
a weak +0.31 (mean|r|); ORC flips sign between the two density controls (+0.36 vs −0.36). Sign-flips and
disagreement across density measures are the signature of confounding, not a direct geometric effect.

**Honest verdict (single case).** The discrete-curvature "geometric early-warning" of the affect network
**reduces to network density**: the items synchronize as autocorrelation rises, and both curvatures are
oppositely-signed restatements of that densification rather than an addition to it. The genuine EWS-like
quantity here is the simpler one — **affect-network density (mean|r|) rises with critical slowing down**
(a known psychometric-network EWS) — and curvature does not improve on it on this case.

**Consequence:** Ricci-flow community detection (planned step 3) was NOT built — with no curvature signal
surviving the density control, there is no geometric structure to flow; building it would dress up a
density proxy. Deferred unless a multi-subject cohort shows a density-independent curvature effect.

### Caveat retained either way
The *sign* of any curvature-EWS here is curvature-flavour-dependent (Forman −, Ollivier +; the two are
anti-correlated at −0.89) — there is no universal "curvature drops at transitions" on this substrate.

## Warranted next steps (now that both robustness tests passed)
1. **Temporal-transition full-network ORC** — the direct extension of the 4-node work and the matched
   comparison to this structural result (joint/per-node distribution shift between consecutive windows).
2. **Ricci-flow community detection / surgery** on the affect network over time — does community structure
   reorganize at the transition?
3. **Exact-OT Ollivier-Ricci** (reuse repo Sinkhorn-LSE) per edge, alongside Forman, on the same networks.
4. Threshold sensitivity sweep; replication on other ESM cohorts when available.

Reproduce: `python3 scripts/research/esm_affect_network_fixture.py --window 50 --step 50` then
`cat affect_net_curv_data.sio affect_net_curvature.sio | bin/souc run`.
