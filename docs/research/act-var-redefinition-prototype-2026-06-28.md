<!-- docs:meta
topic_id: repo.docs.research.act-var-redefinition-prototype-2026-06-28
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.act-var-redefinition-prototype-2026-06-28
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Prototype: redefining `act_var` into a concentrating, nonstationarity-robust bonus

Date: 2026-06-28
Branch: `recover/solver-gpu-arc`
Reproduce: `python3 scripts/research/act_var_redef_proto.py`
Context: [[solver-novelty-boundary-verified-2026-06-28]] — the open frontier is a genuine
variance-aware (UCB-V) bandit branching signal; the blocker ("the research nut") is the
**decay-vs-concentration tension**: UCB-V assumes i.i.d. stationary rewards, but SAT
rewards are nonstationary → needs decay → naive decay re-inflates the variance bonus back
into the growing-accumulator pathology.

## What the prototype tests (falsifiable)

One focal variable, per-conflict reward `r_t ∈ {0,1}` (bounded → UCB-V valid), with a
**regime shift**: true mean `p=0.70` for `t<2000`, then `p=0.20` (the variable stops being
conflict-relevant). A correct uncertainty bonus must (a) concentrate within a stable
regime, (b) transiently rise right after the shift, (c) stay bounded. Three estimators of
the per-variable variance bonus, scored as `score = mean + β·bonus`:

1. **GROWING** — the current `smt.sio` `act_var` (`+= bump²·…`, `bump *= 1/0.95`).
2. **STAT** — empirical-Bernstein UCB-V over all history (stationary).
3. **DISC** — *discounted* empirical-Bernstein with a **local horizon** (the proposal).

## Result

| estimator | within-regime | at regime shift | bounded? | storage |
|---|---|---|---|---|
| GROWING (current) | **grows ×1.3e40** (more evidence → *more* "uncertain") | n/a | **NO** (→2.7e89) | 1 float |
| STAT UCB-V | concentrates 0.124→0.059 | **stale** (0.0574, no reaction) | yes→0 | 2 acc |
| **DISC (proposed)** | concentrates to a **floor** (0.415→0.323) | **rises** 0.323→0.331, then re-concentrates 0.293 | **yes** (max 0.415) | **3 floats** |

Two corrections the prototype *surfaced* (it falsified the naive forms first):
- The naive windowed UCB-V **drifts up** because UCB-V's `ζ·log(t)` factor keeps growing
  while window `n` is capped. Fix: a **local horizon** `log(min(t,W))` / `log(n_eff)`
  (sliding-window UCB, Garivier–Moulines 2008). Combining SW-UCB's local horizon with
  UCB-V's empirical-variance term is part of the construction.
- DISC concentrates to a **floor > 0**, not to zero. That is *correct* for nonstationarity:
  bounded `n_eff = 1/(1-γ)` retains irreducible uncertainty (the world may shift). A
  stationary estimator collapsing to 0 is exactly what makes it stale.

## The portable construction (drop-in for `smt.sio`)

Per variable, replace the growing `act_var` with **3 discounted accumulators** (γ≈0.99,
`n_eff` steady-state = 100), updated on each conflict reward `r∈[0,1]`:

```
S0 = γ·S0 + 1            # discounted count   (n_eff)
S1 = γ·S1 + r            # discounted sum
S2 = γ·S2 + r·r          # discounted sum of squares
mean = S1/S0
var  = max(S2/S0 - mean·mean, 0)
bonus = sqrt(2·var·ζ·ln(max(S0,2)) / S0) + 3·ζ·ln(max(S0,2)) / S0      # ζ≈1.2, b=1
score = mean + β·bonus
```

This is O(1) storage (3 floats/var, vs the current `act_mean`+`act_var` = 2) and O(1)
update — fully compatible with `smt.sio`'s fixed `[f64;1024]` layout. Land it as a **new
`score_mode = 5`** (Discounted Empirical-Bernstein UCB-V) so mode 0 stays as the ablation
baseline. The reward `r` should be a bounded per-variable conflict-relevance/learning-rate
in `[0,1]` (e.g. participated-in-learnt-clause indicator, or the mode-4 LR ratio).

## What this does NOT prove (honest limits)

- This validates the **math** (concentration + adaptation + boundedness on a synthetic
  nonstationary reward stream), NOT that the heuristic **improves solving**. The real test
  is the ablation `β=0` vs `β>0` on instances stratified by variable discriminability
  (variance-aware bandits only help in the small-gap/high-variance regime), each UNSAT
  shipping a checked LRAT/cake_lpr receipt — see the boundary doc.
- Single-variable synthetic stream; the real per-variable reward definition and (γ, β, ζ)
  tuning are open.
- The "first variance-aware bandit branching" claim still rests on the prior-art absence
  (verified), not a proof of impossibility.

## Next step

Port the construction as `score_mode=5` in `stdlib/theorem/smt.sio` (self-hosted change →
forensic dispatch + build-locked rebuild + `test_smt` no-regression), then run the
discriminability-stratified ablation.
