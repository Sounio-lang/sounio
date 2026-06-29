# Pre-registration — Variance-aware branching (score_mode=5) ablation

Date: 2026-06-29
Branch: `research/novelty-ablation` (base `d7bf33814`, madaros `cb2f7685…`)
Status: **PRE-REGISTERED — written before any conflict numbers were observed.**

This document fixes the design of the β-ablation experiment **before the first run**,
so the harness cannot quietly degrade into a search for a flattering configuration.
It is the integrity backbone for the narrow novelty claim recorded in the verified
claim boundary (see `docs/research/solver-novelty-boundary-verified-2026-06-28.md`).

## Background and the single open question

The narrow novelty rests on a genuinely open frontier (verified across 2 surveys +
focused searches): **no SAT/SMT/CP solver has used empirical-variance (UCB-V /
UCB1-Tuned) exploration for variable selection.** Sounio's `score_mode=5` implements a
*discounted empirical-Bernstein* UCB-V variable score whose bonus SHRINKS as evidence
accrues (a genuinely concentrating uncertainty), unlike the legacy growing `act_var`
(which is an overclaim and is NOT under test here).

The prototype (`scripts/research/act_var_redef_proto.py`,
`docs/research/act-var-redefinition-prototype-2026-06-28.md`) validated the **math
only** (the discounted form concentrates to a bounded floor on a regime-shift stream).
It did **not** validate any solving benefit. This experiment tests solving benefit.

UCB-V theory predicts the variance bonus helps **only** when arms (variables) have
**small gaps** between their true means — i.e. variables are hard to tell apart. Where
one variable is clearly best (large gap), a first-moment score already picks it and the
variance term is wasted or harmful.

## Hypothesis (directional, regime-specific)

> Turning the empirical-variance bonus ON (`disc_beta_scale = 1.0`, "β>0") rather than
> OFF (`disc_beta_scale = 0.0`, "β=0", which reduces score_mode=5 to its discounted
> mean) **reduces conflicts-to-completion in the small-gap / high-variance stratum, and
> does NOT reduce them in the large-gap / low-variance stratum.**

The regime-specificity is the real test. A uniform improvement across both strata would
*disconfirm* the mechanism (it would mean something other than variance-awareness is
acting). A reduction confined to the high-variance stratum is the predicted signature.

## Strata (a priori, never measured post-hoc)

Generated uniform random **3-SAT** at controlled clause/variable ratio α — a standard
a-priori knob for cross-variable symmetry. No scraped instances (the harness caps —
`n_vars ≤ 500`, `n_clauses ≤ 4096`, ≤8 lits/clause — admit too few real CNFs).

- **S_high** (small-gap / high-variance): α = 4.26, the 3-SAT phase transition, where
  instances are maximally symmetric and hardest — variables are least distinguishable.
- **S_low** (large-gap / low-variance): α = 6.0, well overconstrained, where UNSAT is
  forced early through few highly-constrained variables — large score gaps.

n_vars = 80 for both strata. Only **UNSAT** instances are kept (clean conflicts-to-
refutation metric and a future cert path). 20 instances per stratum, fixed seeds
`1000..1019` (S_high) and `2000..2019` (S_low), generated identically except α.

**Stratification-validity check (pre-run gate, secondary):** the phase-transition
ratio is a *proxy* for cross-variable small gaps; v1 does not instrument per-decision
score gaps directly (a stated v1 limitation). As a sanity proxy we require S_high to be
substantially harder than S_low under β=0 (mean conflicts S_high ≫ S_low). If this
ordering fails, the strata are not behaving as designed and results are uninterpretable.

## Conditions

`score_mode = 5` throughout. To isolate the **variable-selection** variance axis (the
primary novel axis), hold `phase_mode = 0` (saved-phase) and `restart_budget = 0`.
Single factor under test:

- **β=0**:  `disc_beta_scale = 0.0`
- **β>0**:  `disc_beta_scale = 1.0`

`score_mode=5` is deterministic given an instance (no RNG; the score is a function of
the disc accumulators), so one run per (instance, condition); the instance population
supplies the variance. (The joint var+polarity+regime conjunction — the full claim — is
a *later* stage; this stage isolates the one axis that is genuinely open.)

## Metric and success criterion

- Primary metric: **conflicts-to-completion** per (instance, condition).
- Per-instance paired effect: `Δ = conflicts(β=0) − conflicts(β>0)` (positive = β>0 helps).
- Aggregate per stratum: mean Δ with a bootstrap 95% CI; paired sign test across the 20
  instances (p < 0.05).

**Success (mechanism confirmed):** mean Δ > 0 with p < 0.05 in **S_high**, AND mean Δ
not significantly > 0 in **S_low**.

**Pre-committed outcomes:**
- Confirmed → proceed to the full conjunction stage + verified-receipt (LRAT/cake_lpr) half.
- **Null (Δ ≈ 0 in S_high) → report the null.** This is a legitimate, shippable result
  and is explicitly an acceptable outcome of this experiment. The variance-aware axis
  would then be honestly logged as "no measured solving benefit at this scale," not quietly
  dropped or re-searched for a better-looking config.
- Uniform (Δ > 0 in both strata) → mechanism disconfirmed; investigate confound.

## Correctness gate (no measuring conflicts on wrong answers)

Every instance's Sounio SAT/UNSAT verdict is cross-checked against **z3** (the only
external solver on PATH) via emitted DIMACS. Instances where Sounio disagrees with z3
are excluded and flagged as solver-correctness bugs (a separate issue, not ablation data).

## Known v1 limitations (stated up front)

1. Synthetic generated instances only — this is the **mechanism test**, not the
   external-benchmark requirement (#3 of the readiness doc). Not a SOTA claim.
2. No direct per-decision score-gap instrumentation; α is the a-priori proxy.
3. No verified UNSAT certificate yet (solver emits no DRAT today; receipt half deferred).
4. Pilot scale (20 instances/stratum, n_vars=80).

Nothing here may be relabeled as a public novelty result. Safe wording stays:
"candidate variance-aware (UCB-V) variable-selection experiment with regime-gated
exploration, pilot scale, mechanism test only."

---

## ADDENDUM — capability-pilot findings & design revision (2026-06-29, BEFORE any outcome run)

A capability pilot (`benchmarks/solver/cap_pilot.sio`, n∈{20..50}, β=0) was run to
validate the substrate and stratification **before** building the real matrix. It
revised the design. This revision is made on pilot/capability data only — NO β=0-vs-β>0
*outcome* comparison has been observed. (Pilot-informed pre-registration; revising the
plan after seeing outcome data would be the violation, and is not what happened.)

**Findings:**
1. Substrate green: `score_mode=5` runs bounded/correct. Solver verdict **agrees with z3**
   on a byte-identical DIMACS instance (`z3_check_probe.sio`: Sounio SAT == z3 SATISFIABLE).
   No solver-correctness problem at this scale.
2. **The α=4.26 stratification premise is INVERTED for this solver.** This is a bounded
   **DPLL(T) without clause learning**, not CDCL. At α=4.26 instances are mostly SAT and
   found with **0 conflicts** (trivial descent). The conflict-bearing search — where
   variable-selection heuristics can matter — lives in the **overconstrained UNSAT regime
   (α≈6.0)**: there instances are UNSAT with 150–1700 conflicts. The textbook
   "phase-transition = hard" does not transfer to a non-CDCL DPLL core.
3. A multi-instance harness memory bug (n-grid array clobbered as ctx allocations
   accumulate) must be fixed; the single-instance path is clean.

**Revised design (supersedes Strata / Conditions / success-criterion above):**
- **Regime under test = overconstrained UNSAT** (α≈6.0), the conflict-bearing regime.
  Instances kept only if Sounio returns definitive `0` (UNSAT), agrees with z3, and is
  uncensored (decisions < cap). SAT or censored instances are excluded.
- **Primary outcome (clean, unconfounded):** within the UNSAT set, paired
  `Δ = conflicts(β=0) − conflicts(β>0)` per instance; mean Δ with bootstrap 95% CI and a
  paired sign test (p<0.05). H1: β>0 reduces conflicts-to-UNSAT. **Null is shippable.**
- **Secondary/exploratory (per advisor BLOCK 2 — confound-aware):** does Δ vary with a
  cross-variable-discriminability proxy (e.g. α swept 5.0/6.0/7.0, or n)? Only interpreted
  where every cell has real headroom (β=0 conflict count well above single digits).
- Capability gate from the pilot: pick the largest n that yields a usable population of
  uncensored UNSAT instances in reasonable wall-time (pilot suggests n≈50–87 at α=6.0
  completes with hundreds–low-thousands of conflicts — ample headroom under the 100000
  decision cap).

Stratification by the textbook transition is retired for this solver; the honest axis is
conflict-bearing UNSAT hardness. The variance-aware *mechanism* claim is unchanged; only
the instance regime that exercises it is corrected.
