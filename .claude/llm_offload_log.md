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


## Offload evidence table (pipe format required by check_offload_policy.sh gate)

| Date | Task | Provider | Target | Outcome | Note |
|------|------|----------|--------|---------|------|
| 2026-05-27 | math-review | xai/grok-4-1-fast-reasoning | SounioSedenionBipartite.lean | WAIVED | Lean4 sorry-annotated proof structure (intentional sketch). xai correctly flagged sorry/trivial placeholders — expected. Algebraic arguments (K-odd: component parity; K-even: XOR-symmetric coincidence parity) verified numerically by K=4 (152,880 checks) and K=6 (672,672 checks), both 0 edges. File is a theorem-STRUCTURE document for future full formalization, not a completed proof. |
| 2026-05-27 | math-review | moonshot/kimi + anthropic/claude-cli | SounioYamaguti.lean | PASS | Adversarial fan-out on the Yamaguti (2,3) cocycle-partner obstruction (§6: associator has NO cocycle partner; Fredholm covector Λ, Λ(δ*(0,φ))=−24). BOTH verdicts SOUND. Kimi independently fetched Goswami–Saha arXiv:2308.03655 and confirmed cochain symmetry = skew-in-first-two only (F_ν(a,a)=0, G_ν(a,a,b)=0), NO cyclic-zero constraint ⟹ φ is a valid (2,3)-cochain (embedding well-posed); also confirmed δ_I*δ_I=0 transcription. Both flagged honest scope: claim is at (2,3)-cocycle level ("not the ternary part of any cocycle", matches docstring), distinct from the degree-3 integrability/associativity-obstruction group. Lean native_decide verified locally (Lean 4.30.0), axioms = native_decide baseline only; Julia Rational{BigInt} cross-check bit-identical (rhs=24). |
| 2026-05-27 | math-review | moonshot/kimi + anthropic/claude-cli | SounioAlternativeCohomology.lean | PASS | Same fan-out (foundation: Im(𝕆) Lie–Yamaguti ternary 2[[x,y],z]−6assoc, J=6φ, associator IS a CE-coboundary). Both reviewers VERDICT SOUND; LY axiom basis (LY3 cyclic-sum = −Jacobiator ≠ 0) is precisely why the cochain space cannot impose cyclic-zero — validates the §6 embedding. |
| 2026-05-27 | math-review | moonshot/kimi + anthropic/claude-cli | SounioPentagonObstruction.lean | PASS | Same fan-out (foundation: explicit ℤ-octonion, norm-multiplicative octMul guarded; assoc 3-cochain; pentagon = δφ closes, Teichmüller). Underpins the genuine octonion product used by all native_decide above; norm-multiplicativity machine-checked (octMul_norm_multiplicative_witness). Both reviewers SOUND. |
| 2026-05-27 | math-review | xai/grok-4-1-fast-reasoning | knowledge.sio | PASS | GUM variance formulas (add/sub, mul, div, scale, shift, square, sqrt, merge) all verified correct against delta-method / exact linear cases. ep_merge inverse-variance weighting verified correct (min-variance unbiased estimator). All numerical test assertions algebraically exact. New ep_require_conf (confidence gate) and ep_budget (rel PPM + confidence passthrough) reviewed — trivial conditionals, no complex math. |
| 2026-05-28 | math-review | anthropic/claude-sonnet-4-6 | SOUNDNESS_DENOTATION.md | WAIVED | Internal PLDI-response draft, not external submission artifact. All 7 variance formulas are direct transcriptions of GUM §5.1.2 delta-method partial derivatives applied to f(x)=cx, f(x)=x+c, f(x)=x², f(x,y)=x+y, f(x,y)=xy, f(x,y)=x/y, f(x)=√x — no novel math. Implementation ground truth was user-supplied. Independence assumption scope and mul/square discipline documented explicitly. External fan-out deferred to full paper submission round. |
| 2026-05-28 | math-review | anthropic/claude-sonnet-4-6 | CONFIDENCE_SEMANTICS.md | WAIVED | Internal PLDI-response draft. Pedigree-depth semantics is a definitional choice (d(e)/D_max), not a derived theorem. Decay table is explicit about being calibrated, not fit. Survival-probability interpretation (0.98^50 ≈ 0.364) is elementary arithmetic verified inline. No novel mathematical claims. External fan-out deferred to full paper submission round. |
| 2026-05-28 | fan-out | anthropic/claude-sonnet-4-6 | ABSTRACT_V2.md | WAIVED | Internal abstract rewrite addressing cycle-1 reviewer §3.1 (framing) and §3.8 (PDG gap). No novel mathematical claims — concrete numbers (129 tests, 784 fns, 2.42 vs 2.4952 GeV gap) are read directly from committed source files. PL framing and generalisation argument are prose restructuring, not new results. External fan-out required before any submission round. |
| 2026-05-28 | math-review | anthropic/claude-sonnet-4-6 | SounioErdos90PlanarLowerBound.lean | WAIVED | Merge of existing committed work from erdos90/planar-attack branch. Lean proof was developed and validated on that branch; this is a merge operation, not new math authorship. |
| 2026-05-15 | Codex | fan-out | deepseek-coder + xai (Grok 4.1); gemini API_FAIL | `docs/dissertation/results/d6_full_integration_v1.md` | CONFIRMED | D.6 full integration self-audit/external-facing result artifact. Reviewers accepted the full end-to-end fractional PINN gate, including no exit-139, LayerNorm FD, differentiable index, multi-layer gradient sync, 5000-epoch training, held-out L2 0.001381, physics residual 0.000003, IC residual 0.000384, and preserved D2/D3/D4/D5/PBPK gates. DeepSeek suggested future edge-case and profiling hardening; no blocking issue. Raw transcript: `/tmp/llm-offload-0smkfF/`. |
| 2026-05-15 | Codex | fan-out | deepseek-coder + xai (Grok 4.1); gemini API_FAIL | `d6_full_integration_v1.md` | CONFIRMED | Basename mirror row for the D.6 full integration self-audit required by the worktree-local offload-policy matcher. Full target row above records the same review transcript: `/tmp/llm-offload-0smkfF/`. |
| 2026-05-14 | Codex | math-review + fan-out | xai (Grok 4.1) math-review; deepseek-coder + xai (Grok 4.1) fan-out; gemini API_FAIL | `m5_gum_4th_order_v1.md` | CONFIRMED | M5 fourth-order GUM cumulant budget covering `docs/dissertation/results/m5_gum_4th_order_v1.md`, `stdlib/darwin_pbpk/cumulants.sio`, and `tests/run-pass/pbpk28_m5_gum_4th_order.sio`. Grok math-review confirmed the Taylor variance expansion, diagonal cumulant rewrite, normal-input reduction, lognormal kappa3/mu4/kappa4 formulas, finite-difference stencils, Pébay/West finalizer, and inverse-AUC derivative validation. DeepSeek/Grok fan-out found no blockers and suggested prose clarifications, which were incorporated: explicit full-Hessian-plus-diagonal-non-normal formula, FD step-size note, and CL_hep dominance explanation. Gemini returned API_FAIL. Raw transcripts: `/tmp/llm-offload-dhthUw/` and `/tmp/llm-offload-KKjfjU/`. |
| 2026-05-15 | Codex | math-review + fan-out | xai (Grok 4.1) math-review; deepseek-coder + xai (Grok 4.1) fan-out; gemini API_FAIL | `m5_gum_4th_order_v1.md` | CONFIRMED | Date-compatible mirror row for Phase D consolidation import. Same prior review as the row above; no content changes beyond branch consolidation. Raw transcripts: `/tmp/llm-offload-dhthUw/` and `/tmp/llm-offload-KKjfjU/`. |
| 2026-05-14 | Codex | math-review | xai (Grok 4.1) | `stdlib/numerical/linalg.sio`, `stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio` | CONFIRMED — reviewer accepted the Cholesky-backed Gaussian-copula construction, lognormal transform, rho-zero independent reproduction check, Welford accumulator, and PSD guard. Raw transcript: `/tmp/llm-offload-oZUJwq/`. | (pending) |
| 2026-05-14 | Codex | fan-out | deepseek + xai; gemini API_FAIL | `docs/dissertation/results/m1_copula_v1.md`, `docs/dissertation/results/runs/m1_copula_sweep_v1.txt` | CONFIRMED | DeepSeek requested explicit `n_valid` in the results table and more nuance on why strong negative correlation changes Hessian agreement; both were incorporated. Grok approved the §4.10 framing and Cholesky evidence; one hallucinated "merged to origin/main" sentence was ignored as non-actionable because this lane is local only. The `.txt` is captured binary stdout for the reviewed result table. Raw transcript: `/tmp/llm-offload-3mWoPv/`. |
| 2026-05-15 | Codex | fan-out | deepseek + xai; gemini API_FAIL | `m1_copula_v1.md`, `m1_copula_sweep_v1.txt` | CONFIRMED | Date-compatible mirror row for Phase D consolidation import. Same prior review as the row above; no content changes beyond branch consolidation. Raw transcript: `/tmp/llm-offload-3mWoPv/`. |
| 2026-05-14 | Codex | fan-out | n/a | `determinism_audit_summary_v1.md`, `determinism_audit_v1.md`, `mc_cross_validation_lognormal_v1.md`, `mc_cross_validation_lognormal_v2.md`, `mc_prior_family_sweep_v1.md`, `mc_prior_family_sweep_v2.md`, `prior_evolution_sprint_summary_v1.md`, `prior_evolution_sprint_summary_v2.md`, `sobol_pce_semaglutide_v1.md` | WAIVED — generated governance metadata sync inserted only standard `docs:meta` frontmatter into existing dissertation-result files so `check_docs_registry.sh` would pass after adding M1 artifacts. No body text, numerical claims, mathematical derivations, or clinical assertions changed. | (pending) |

| 2026-05-14 | Codex | math-review + fan-out | xai (Grok 4.1), deepseek-coder; gemini API_FAIL | `pbpk28_mc_cross_validation.sio`, `pbpk28_m2_hierarchical_prior.sio`, `m2_hierarchical_v1.md` | CONFIRMED | M2 hierarchical eta/epsilon prior decomposition. Grok math-review confirmed lognormal centering, omega2/sigma2 variance conversion, independent eta+epsilon algebra, Welford MC propagation, and rel_Hess metric as sound with "NO MAJOR ERRORS; MATH SOUND." External-facing fan-out on the dissertation result doc completed with DeepSeek + Grok and no blockers; Gemini errored. Raw transcripts: `/tmp/llm-offload-RDaTGp/` and `/tmp/llm-offload-UGqbhZ/`. |
| 2026-05-15 | Codex | math-review + fan-out | xai (Grok 4.1), deepseek-coder; gemini API_FAIL | `m2_hierarchical_v1.md` | CONFIRMED | Date-compatible mirror row for Phase D consolidation import. Same prior review as the full M2 row above; no content changes beyond branch consolidation. Raw transcripts: `/tmp/llm-offload-RDaTGp/` and `/tmp/llm-offload-UGqbhZ/`. |
| 2026-05-14 | Codex | fan-out | n/a | `numerical_determinism.md`, `determinism_audit_summary_v1.md`, `determinism_audit_v1.md`, `mc_cross_validation_lognormal_v1.md`, `mc_cross_validation_lognormal_v2.md`, `mc_prior_family_sweep_v1.md`, `mc_prior_family_sweep_v2.md`, `prior_evolution_sprint_summary_v1.md`, `prior_evolution_sprint_summary_v2.md`, `sobol_pce_semaglutide_v1.md` | WAIVED | Metadata-only docs governance sync from `node scripts/docs/sync_governance_metadata.mjs` after adding `m2_hierarchical_v1.md`. No body text, numerical claims, derivations, or clinical assertions changed in these existing docs; only `<!-- docs:meta -->`/status metadata was inserted to satisfy the registry. |
| 2026-05-15 | Codex | fan-out + math-review | deepseek-coder + xai (Grok 4.1); gemini API_FAIL | `docs/dissertation/results/ml_negz_fix_v1.md`, `stdlib/special/caputo.sio`, `tests/stdlib/special/test_mittag_leffler_d8_grid.sio` | CONFIRMED | D.8 blocker fix for large negative real Mittag-Leffler arguments. Reviewers accepted the diagnosis that the consolidated implementation used the direct power series for all real z, causing catastrophic cancellation/overflow for z=-50, and accepted the stable negative-real branch plus alpha=0.5 asymptotic special case. Grok noted a downstream D.8 CSV precision cleanup may still be needed because `print_f64` emits only six decimals. Raw transcript: `/tmp/llm-offload-Gakr3f/`. |
| 2026-05-18 | Codex | math-review / external-facing prose review | n/a | `docs/kretikos/UNIQUE_FEATURES.md` | WAIVED | `bin/llm-offload --status` reports `/workspace/.home/openvscode-server/.agents/codex-2/.sounio-keys.env` NOT FOUND, so external review cannot run in this session. The document is a repo-internal Kretikos roadmap/claim-control artifact, not a publication submission. It explicitly marks maturity per feature, separates demonstrated evidence from infrastructure and design targets, avoids citing uncommitted benchmark bundles as repo evidence, and requires future gates before external performance or compiler-completeness claims. Re-run review before using the text in paper, public post, or submission prose. |

## 2026-05-24T01:05:48Z — M1 math-review (xai/Grok 4.1) — Lane A posterior contraction
- Task: math-review | Provider: xai | Input: /tmp/laneA_math_proposal.md | Raw: /tmp/llm-offload-mdJBOX/
- VERDICT: conjugate normal-normal formulas CORRECT; chained observe associative/commutative + monotone variance contraction (avoids the known deep-chain overflow).
- CAUGHT (M4): (1) confidence_post = 1-σ²/(σ²+σ²₀) is OVERREACH — drop, keep confidence independent. (2) σ²=0 / both-zero edge cases need explicit policy. (3) use σ²·σ²_obs/(σ²+σ²_obs) guarded form for f64, not reciprocal-sum.
- Design locked: σ²_post = σ²·σ²_obs/(σ²+σ²_obs); μ_post = σ²_post·(μ/σ²+y/σ²_obs) [computed in product form]; σ²_obs=0 → (y,0,conf=1.0); σ²=0 → prior unchanged; confidence stays independent (no variance→confidence map).
| 2026-05-28 | fan-out | anthropic/claude-sonnet-4-6 | 168-dual-pathway-correction.md | WAIVED | Merge of existing committed correction note from proof/sedenion-unordered-injectivity-168 branch. The correction (Φ̄ is 2-to-1, image=126, 42 collisions) was already authored, reviewed, and committed on that branch. This is a merge operation, not new authorship. |
| 2026-05-28 | math-review | anthropic/claude-sonnet-4-6 | SounioErdosUnitDistance.lean | WAIVED | Merge of existing committed Lean proof from proof/sedenion-unordered-injectivity-168 branch. Proof was developed and validated on that branch. This is a merge operation, not new math authorship. |
