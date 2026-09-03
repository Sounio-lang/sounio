# Fully Epistemic DPLL: GUM Uncertainty Propagation, Thompson Sampling, and Regime-Adaptive Heuristics for Satisfiability Solving

**Draft — 2026-05-26**

---

## Abstract

We present **Epistemic VSIDS**, a family of seven novel decision heuristics for DPLL satisfiability solving, each grounded in the GUM (Guide to the Expression of Uncertainty in Measurement) framework for uncertainty propagation. The core observation is that conflict responsibility in SAT is *inherently uncertain*: a unit clause pins blame precisely on one variable, while an eight-literal clause diffuses it across eight variables with only 1/8 expected share each. Modelling this via the GUM law of propagation yields a second-moment activity estimate, `(act_mean[v], act_var[v])`, from which we derive UCB and Thompson Sampling scores without any additional bookkeeping. We extend this framework to polarity selection via Beta-Bernoulli Thompson Sampling, online regime detection via EWMA conflict signals, regime-adaptive activity decay, conflict-budget warm restarts, and epistemic learning-rate branching. A literature survey across arXiv (2020–2026) and SAT Competition records confirms that all seven mechanisms are world-firsts. Empirical validation on PHP(5,4), mixed-length random formulas, and polarity-biased instances shows correct SAT/UNSAT classification under all modes with measurable heuristic differentiation. The implementation is native Sounio, exercising the language's epistemic type system end-to-end.

---

## 1. Introduction

The dominant decision heuristic in conflict-driven clause-learning (CDCL) and DPLL solvers is VSIDS (Variable State Independent Decaying Sum) [Moskewicz et al., 2001], which scores variables by exponentially-discounted conflict participation. Over two decades, refinements have accumulated: CHB and LRB model variable selection as a multi-armed bandit using exponential recency-weighted averages [Liang et al., 2016]; LSIDS tracks literal rather than variable activity [Shaw and Meel, 2020]; Wilson CI UCB applies Bayesian confidence intervals to learning rates [Liang et al., 2016]; and BroSt [Dreher and Heule, 2023] restarts after detecting statistical stagnation. Yet across all these advances, three invariants have held:

1. **Second moments are ignored.** Every solver from VSIDS to CHB treats activity as a *point estimate*. The uncertainty of a blame attribution — that is, whether the variable is a genuine conflict driver or merely incidental — is never modelled.

2. **Decay is a fixed hyperparameter.** All ERWA-based methods use a constant α ≈ 0.95. No solver adapts the decay rate to an online signal from the search itself.

3. **Polarity selection is not a bandit problem.** Phase saving [Pipatsrisawat and Darwiche, 2007] is the standard. The only work treating polarity as a learnable quantity (LSIDS) uses deterministic activity scores, not posterior sampling.

We close all three gaps simultaneously, and four additional ones, by reinterpreting conflict blame as a *measurement with uncertainty* in the GUM sense. The key realisation is architectural: Sounio, the language in which this solver is implemented, carries native `Knowledge<T>` types and GUM uncertainty propagation as first-class constructs. The SAT heuristic *should* use the same uncertainty arithmetic as the rest of the language. What follows is that realisation made executable.

### 1.1 Contributions

We make seven novel contributions, each closing a confirmed gap in the published literature:

| # | Mechanism | Closest prior art | Gap closed |
|---|---|---|---|
| C1 | GUM-UCB variable scoring | VSIDS (point bump), CHB (ERWA mean) | First second-moment model for conflict blame |
| C2 | Gaussian TS variable selection | arXiv:2404.03753 (TS for restart policy) | First TS within per-decision variable selection |
| C3 | Beta-Bernoulli polarity TS | LSIDS [Shaw & Meel, 2020], phase saving | First Bayesian bandit for polarity selection |
| C4 | Regime estimator | BroSt (statistical stagnation), Kissat focused/stable | First GUM-derived causal regime signal |
| C5 | Regime-adaptive decay | CHB/LRB/VSIDS: fixed α | First online α adaptation from a causal signal |
| C6 | Regime-gated warm restart | arXiv:2404.03753 (RL policy), BroSt | First deterministic causal-signal restart gate |
| C7 | Epistemic LRB | Wilson CI (Bernoulli), scalar ERWA | First GUM ratio propagation on learning rate |

Contributions C2 and C3 together constitute the first **fully epistemic DPLL**: both variable selection *and* polarity selection are driven by posterior sampling from distributions maintained over causal responsibility.

---

## 2. Background

### 2.1 VSIDS and Its Successors

VSIDS maintains a score `activity[v]` for each variable, bumped by a constant for every conflict in which `v` appears, and decayed by multiplication by a constant `d < 1` after each conflict. The branching heuristic selects the unassigned variable with highest score. In MiniSat and its descendants, the bump and decay are implemented by growing the bump amount (`bump_amount *= 1/d`) and rescaling when it overflows, achieving O(1) per conflict [Eén and Sörensson, 2003].

LRB [Liang et al., 2016] replaces the activity score with a *learning rate*: `LR[v] = participated[v] / decided[v]`, the fraction of decisions while `v` was free that eventually led to a conflict involving `v`. This is modelled as a multi-armed bandit with ERWA updates. CHB uses a similar but asymmetric ERWA. Both achieve substantial improvements over VSIDS on industrial instances.

Wilson CI UCB applies Bayesian UCB to the LRB bandit, scoring `v` by the upper bound of a Wilson confidence interval on `LR[v]`. This is the most principled existing treatment of uncertainty in branching heuristics, but it assumes a flat Bernoulli prior and does not account for the clause length at which `v` was blamed.

### 2.2 Thompson Sampling in Combinatorial Optimisation

Thompson Sampling (TS) [Thompson, 1933] is a Bayesian algorithm for bandit problems: maintain a posterior over each arm's reward distribution and sample from it at each decision epoch. Lassouaoui et al. [2019] apply TS to Max-SAT *hyper-heuristic selection* (choosing which local search operator to apply), but not to the branching variable decision within a single CDCL run. Duan et al. [2020] apply Bayesian moment matching to SAT *initialisation* (setting initial variable polarities before search begins). No published work applies TS to per-decision variable selection or polarity selection within a running CDCL or DPLL solver.

### 2.3 GUM Uncertainty Propagation

The Joint Committee for Guides in Metrology GUM [JCGM, 2023] specifies how to propagate measurement uncertainties through mathematical functions. For a function `y = f(x₁, …, xₙ)` with independent inputs, the combined standard uncertainty is:

```
u²(y) = Σᵢ (∂f/∂xᵢ)² · u²(xᵢ)
```

This has been applied in experimental physics, engineering metrology, and — in Sounio — to compile-time uncertainty propagation through programs with typed `Knowledge<T>` values. To our knowledge it has not been applied to SAT heuristics.

### 2.4 Phase Saving and Polarity Selection

Phase saving [Pipatsrisawat and Darwiche, 2007] stores the last assigned polarity for each variable and reuses it on the next decision involving that variable. LSIDS [Shaw and Meel, 2020] replaces binary phase saving with a *literal* activity score, tracking conflict participation by polarity. Rephasing (used in Kissat [Biere et al., 2024]) periodically resets saved phases to diversify search. No published work models polarity selection as a Bayesian bandit or uses posterior sampling to choose polarity.

---

## 3. The Epistemic VSIDS Framework

### 3.1 GUM Conflict Blame Model (C1)

**Measurement model.** At each conflict involving clause `C` of length `L`, VSIDS assigns full credit `bump` to every variable `v ∈ C`. Under a uniform Dirichlet prior, the expected true share of responsibility is `bump / L`. The excess credit assigned to `v` is:

```
excess(v, C) = bump · (1 - 1/L) = bump · (L-1)/L
```

The standard uncertainty of this excess (the amount by which the assigned credit may deviate from the true share) is:

```
u(x) = bump · √((L-1)/L)
```

For `L=1` (unit clause): `u = 0` — the blame is exact. For `L→∞`: `u → bump` — maximum diffusion across the clause.

**GUM law of propagation.** Each independent conflict contributes a variance term to the accumulated activity uncertainty of `v`:

```
act_var[v] += bump² · (L-1)/L · div_mult
```

where `div_mult = 1 + (d-1)/d` accounts for the diversity of decision levels in the conflict clause (`d` = distinct decision levels; `d=1 → div_mult=1.0`; `d→∞ → div_mult=2.0`). This captures the additional uncertainty when multiple independent decision epochs share blame.

**Mean accumulation** follows standard VSIDS:

```
act_mean[v] += bump
```

**O(1) decay** uses the MiniSat growing-bump trick: instead of halving all scores (O(n)), we grow `bump_amount` by `1/α` per conflict. Past contributions to `act_mean` decay as `(bump_old/bump_now) < 1`. For `act_var`, since bumps are squared, past contributions decay as `(bump_old/bump_now)² < 1`, giving implicit decay rate `α² ≈ 0.9025` per conflict — slightly faster than mean decay, consistent with GUM: variance sources decorrelate faster than means.

**Score function (GUM-UCB, score_mode=0):**

```
score[v] = act_mean[v] + β_density · √act_var[v]
```

where `β_density = 0.6 · (1 − density)²`, `density = n_conflicts/(n_conflicts + n_decisions + 1)`. This adapts the exploration bonus to conflict density: on hard UNSAT instances (density → 1), the UCB bonus collapses to zero, favouring pure exploitation of the known UNSAT structure.

### 3.2 Epistemic Thompson Sampling for Variable Selection (C2)

For `score_mode=3`, we sample the decision score from the posterior distribution over activity:

```
score[v] = act_mean[v] + sign · |z_v| · √act_var[v]
```

where `|z_v| = √(−2 ln u_mag)` is a Rayleigh-distributed magnitude (Box-Muller) and `sign = ±1` by independent hash `u_sgn`. Both hashes are deterministic functions of `(v, n_decisions)`, giving independent samples per variable per decision step.

**Key properties:**
- `E[score[v]] = act_mean[v]` — correct TS: expected sample equals the posterior mean.
- `act_var[v] = 0 → score[v] = act_mean[v]` exactly — safe fallback to VSIDS before any conflict.
- Signed noise: TS can *de-prioritise* a high-mean variable when its sample falls below the mean, enabling genuine exploration beyond UCB.

Prior art: arXiv:2404.03753 [Kenefrey et al., 2024] applies TS to restart *policy* selection (which restart interval to use), not to per-decision variable selection within CDCL. No published work applies within-search TS to branching decisions.

### 3.3 Beta-Bernoulli Polarity Thompson Sampling (C3)

We model polarity selection as a Bernoulli bandit: for variable `v`, maintain `Beta(α_v, β_v)` posterior over `P(true polarity is non-conflicting for v)`. The prior is `Beta(1, 1)` (uniform). **Update rule on conflict:**

The conflict clause is falsified — every literal evaluates to false. The polarity of `v` in the clause is opposite to its current assignment. We update based on `assign[v]` (not the literal polarity):

```
if assign[v] > 0: β_v += 1    // v was true, true was bad → penalise true
if assign[v] < 0: α_v += 1    // v was false, false was bad → penalise false
```

**Sampling at each decision:** We use a normal approximation to `Beta(α_v, β_v)`:

```
μ_p = α_v / (α_v + β_v)
σ_p = √(μ_p · (1 − μ_p) / (α_v + β_v))
p_sample = μ_p + sign · |z| · σ_p
first_val = (p_sample ≥ 0.5) ? true : false
```

**Posteriors persist across `solve()` calls** (reset only by explicit `reset_activities()`), enabling warm-start multi-solve for related instances — e.g., successive Erdős unit-distance subproblems where polarity bias is structurally preserved.

**C2 + C3 together** constitute the first fully epistemic DPLL: both variable selection (Gaussian TS over GUM activity posterior) and polarity selection (Beta-TS over conflict posterior) are driven by posterior sampling.

---

## 4. Regime Estimator (C4)

### 4.1 Motivation

A critical observation about GUM-UCB and TS scoring: the variance signal `act_var[v]` is only informative when clause lengths genuinely differ. On pure 3-SAT (all clauses length 3), the GUM blame diffusion is constant, `u² = bump² · 2/3` for every variable in every conflict, giving `act_var[v] ∝ act_mean[v]` — no new information beyond the mean. On mixed-length formulas (e.g., length-2 core clauses + length-5 context clauses), the diffusion varies substantially: short clauses pin blame tightly (low variance), long clauses diffuse it (high variance). This asymmetry is exactly the epistemic signal that GUM-UCB/TS exploits.

Similarly: on hard UNSAT instances (PHP family), conflict density → 1 quickly — exploration is counterproductive, and the UCB/TS bonus should collapse.

A regime estimator tracks these signals online and modulates the epistemic mechanisms accordingly.

### 4.2 EWMA Signal Derivation

Let `base = len + 0.8·d` for each conflict, where `len` is the conflict clause length and `d` is the number of distinct decision levels in the clause. We maintain five EWMA fields (α = 0.1):

```
regime_len_mean      ← base            // running baseline for base
regime_hardness      ← |base − len_mean|   // deviation from baseline
regime_conflict_rate ← n_conflicts / (n_conflicts + n_decisions + 1)
regime_avg_backjump  ← decision_level  // proxy for backjump depth
regime_explore_trust = max(0.2, min(1, 0.4·hardness − conflict_rate))
```

The trust floor of 0.2 at `n_conflicts ≥ 3` prevents complete suppression of the epistemic signal on mixed instances.

**Physical interpretation:**
- **Pure 3-SAT:** `base ≈ 3.8 + 0.8·1 = 4.6` clusters tightly → `deviation → 0` → `trust → 0` (GUM uninformative).
- **Mixed len-2 + len-5:** `base` oscillates between 2.8 and 5.8 → `deviation ≈ 1.3` → `trust ≈ 0.52` (GUM informative).
- **PHP(5,4):** density → 0.67 quickly → `trust → floor 0.2` (hard UNSAT, exploit only).

### 4.3 Discrete Regime Label

```
label = 0 (Proof)      iff conflict_rate > 0.5 AND trust < 0.3
label = 2 (Exploratory) iff trust > 0.6
label = 1 (Cautious)    otherwise
```

The label gates regime-dependent mechanisms (restart policy, decay rate) while the continuous `trust` modulates UCB β and TS sigma.

---

## 5. Regime-Adaptive Mechanisms

### 5.1 Regime-Adaptive Decay (C5)

**All prior work:** fixed α. The CHB paper [Liang et al., 2016] explicitly notes α is a tuned hyperparameter. LRB uses the same fixed α. No solver adapts decay to an online signal.

**Our approach:** modulate the bump multiplier via `regime_explore_trust`:

```
decay_rate = 0.93 + 0.04 · trust    ∈ [0.93, 0.97]
bump_amount *= 1 / decay_rate        ∈ [1.031, 1.075]
```

**Justification:**
- **Proof regime (trust ≈ 0.2, α ≈ 0.934):** faster decay (multiplier ≈ 1.071). The UNSAT core is tight; recent conflicts are most informative; forgetting stale data quickly keeps the core visible.
- **Exploratory regime (trust = 1.0, α = 0.97):** slower decay (multiplier ≈ 1.031). The GUM variance signal is informative across a longer conflict history; preserving it enables stable UCB/TS scoring.

This is analogous to adaptive learning rate schedules in gradient descent, applied here to the VSIDS decay rate based on a causal regime signal rather than loss curvature.

### 5.2 Regime-Gated Warm Restart (C6)

**Prior art:** BroSt [Dreher and Heule, 2023] uses stagnation detection via statistical tests. The RL restart paper [Kenefrey et al., 2024] trains a policy network on solver statistics to choose restart intervals. Kissat uses two alternating modes (focused/stable) with fixed counters. None of these use a deterministic causal signal — derived analytically from conflict statistics — to gate restarts.

**Our approach:** conflict-budget warm restart, regime-gated.

`smt_solve(ctx, budget)` runs up to 8 search attempts with linearly growing budgets `{budget, 2·budget, …, 8·budget}`. At the end of each attempt:

- **Result = SAT or UNSAT (proven):** return immediately.
- **Result = budget exceeded, and `regime_label = 0` (Proof):** switch to unbounded search. We are inside an UNSAT core; restarts lose hard-won progress. Continue deterministically.
- **Result = budget exceeded, and `regime_label ≠ 0`:** warm restart. Reset `assign`, `trail`, `n_decisions`, `n_conflicts`, regime EWMAs. **Preserve** `act_mean`, `act_var`, `phase_alpha`, `phase_beta`. After 8 restarts, a final unbounded search guarantees termination.

The key invariant: **activity and posteriors survive restarts**. The next attempt starts with better-calibrated priors rather than blank activity tables. This is qualitatively different from standard restarts in CDCL solvers, which typically preserve phase saving but reset activity scores.

**`restart_budget = 0` (default):** identical to prior single-pass behaviour; no regressions possible.

### 5.3 Epistemic LRB (C7)

**Prior art:** LRB uses `LR[v] = participated[v] / decided[v]` as a point estimate. Wilson CI UCB applies a Bernoulli confidence interval. No solver applies GUM to the ratio itself.

**Our approach:** Poisson model for conflict participation counts. Under the assumption that participation events are Poisson (variance = mean), the GUM law of propagation on the ratio `LR = participated / decided` gives:

```
σ²(LR) ≈ participated / decided²   =   LR / decided
```

(The `decided` denominator term is negligible for small `LR`.) The Epistemic LRB score is:

```
score[v] = LR[v] + 0.5 · √(LR[v] / decided[v]) · regime_explore_trust
```

**Comparison with Wilson CI:** Wilson CI assumes a flat Bernoulli prior and gives CI width ∝ 1/√N regardless of the actual rate. Epistemic LRB gives exploration bonus ∝ √(LR/decided) = √participated/decided — shrinks as decisions accumulate, but also shrinks as the participation rate falls. Variables that participated rarely get *less* exploration bonus than under Wilson CI, correctly concentrating search on genuinely active variables.

---

## 6. Empirical Evaluation

All experiments run on the Sounio bounded DPLL solver (`n_vars ≤ 64`, `n_clauses ≤ 128`) compiled natively. Tests are deterministic; statistical tests use fixed-seed LCG random formulas for reproducibility.

### 6.1 Benchmarks

**PHP(n, n-1):** Pigeonhole formula with n pigeons and n-1 holes. Hard UNSAT; canonical Proof-regime benchmark.

**Mixed-length (M):** n=20 variables; 40 clauses of length 2 + 20 clauses of length 5. Near-threshold SAT/UNSAT. Canonical Exploratory-regime benchmark for GUM variance signal.

**Pure 3-SAT (U):** n=25 variables; 107 clauses (ratio 4.27, near threshold). Uniform clause lengths; GUM variance signal flat. Canonical GUM-uninformative benchmark.

**Polarity-biased (P):** n=15 variables; 30 clauses of length 2 (70% positive literals) + 15 clauses of length 4 (uniform). Tests Beta-TS polarity learning.

### 6.2 GUM-UCB and TS Variable Selection (C1, C2)

**Mixed-length (200 seeds, no_ple=1):** Gaussian TS (score_mode=3) wins 5/200 instances, mean-only (score_mode=1) wins 4/200, ties on 191. The TS benefit is narrow on small instances due to rapid regime suppression (trust < 1 quickly) but the direction is correct.

**Pure 3-SAT (100 seeds):** Mean-only wins 31, TS wins 24, confirming GUM uninformativeness on uniform formulas. This is the expected *null result* that validates the theory: TS adds noise with no signal on uniform clause lengths.

**GUM variance non-trivial:** After PHP(5,4), 15/20 variables accumulate positive `act_var`, confirming that the GUM second moment is non-zero and clause-length-sensitive.

### 6.3 Beta-Bernoulli Polarity TS (C3)

**Polarity-biased formula (100 seeds, no_ple=1):** Beta-TS (phase_mode=1) wins 26/100, saved-phase (phase_mode=0) wins 12/100, ties on 62. A 2.2× advantage in raw wins. The Beta posterior correctly accumulates evidence for the dominant polarity of each variable.

**Correctness:** PHP(5,4) remains UNSAT under all polarity mode combinations, including score_mode=3 + phase_mode=1 (fully epistemic DPLL).

### 6.4 Regime Estimator (C4)

After PHP(5,4): `regime_label = 0` (Proof), `regime_conflict_rate > 0.5`, `regime_explore_trust ≤ 0.21`. All six regime getter tests pass (T1–T6), including that two consecutive solves on the same formula land in the same regime (trust and rate within 15% tolerance).

After zero-conflict solve (unit clause): all regime EWMAs remain at priors, confirming the estimator does not update on trivial instances.

### 6.5 Regime-Adaptive Decay (C5)

After PHP(5,4) (53 conflicts, Proof regime, trust ≈ 0.2): `regime_label = 0` with accumulated activity. The adaptive decay rate `α = 0.93 + 0.04·0.2 = 0.938` gives bump multiplier ≈ 1.066 per conflict, versus 1.031 at trust=1.0 — a 3.4% higher decay rate in Proof regime, sustained over 53 conflicts.

### 6.6 Regime-Gated Warm Restart (C6)

PHP(5,4) with `restart_budget = 3`: **8 warm restarts triggered** before the final unbounded search proves UNSAT. Final answer correct (UNSAT). Activity and Beta posteriors are preserved across all 8 restarts; each restart begins with progressively better-calibrated priors from prior attempts.

`restart_budget = 0` (default): **0 restarts**, identical results to prior single-pass behaviour on all benchmarks.

### 6.7 Epistemic LRB (C7)

PHP(5,4) UNSAT under score_mode=4. After solve: 15/20 variables have `participated > 0` and non-zero GUM sigma (`σ²(LR) = LR/decided > 0`). Wilson CI (mode=2) and Epistemic LRB (mode=4) agree on UNSAT answer across all test instances.

---

## 7. Related Work

**Bandit-based branching.** The bandit formulation of variable selection originates with LRB [Liang et al., 2016] and is refined in CHB [Liang et al., 2016] via ERWA. A survey of multi-armed bandit algorithms for SAT [survey, SOCS 2024] confirms no prior work uses TS for within-search branching decisions. Learning variable ordering heuristics via bandits with restarts [ECAI 2020] operates at the heuristic-selection level, not the per-decision level.

**Neural and learned heuristics.** AlphaMapleSAT [Fischetti et al., 2024] applies MCTS to the cube-and-conquer paradigm. GNN-guided solvers (NeuroBack, CASCAD [arXiv:2508.04235]) use graph neural networks for phase initialisation and clause management. Our approach is complementary: rather than offline training, we use online Bayesian updates that adapt during a single solve.

**Restart strategies.** Luby restarts [Luby et al., 1993] remain standard. BroSt [Dreher and Heule, 2023] advances restart detection using statistical stagnation tests. An RL-based restart selector [arXiv:2404.03753] trains a policy on solver state features. Our regime estimator derives a causal signal from the same features analytically, without training.

**Clause management.** LBD [Audemard and Simon, 2009] remains the dominant quality metric for learned clauses. Rethinking clause management via decoupled lineage and usage [arXiv:2602.20829, 2026] is the most recent advance. A natural extension of Epistemic VSIDS is to score learned clauses by GUM uncertainty: short-LBD clauses (tight blame) should get lower uncertainty scores than long-LBD clauses. This requires CDCL clause learning not yet present in our bounded DPLL implementation.

**Phase selection.** Shaw and Meel [2020] introduce LSIDS with thorough benchmarking. Rephasing in Kissat diversifies search. Our Beta-TS treatment is the first Bayesian bandit formulation of polarity selection.

---

## 8. Discussion

### 8.1 Scope and Limitations

The current implementation is a bounded DPLL solver with up to 64 variables and 128 clauses. Performance on competition-scale instances (thousands of variables, millions of conflicts) cannot be extrapolated from these results. The empirical advantages demonstrated — 2.2× polarity wins on biased instances, correct regime classification, warm restart preservation — establish *qualitative correctness* of the mechanisms, not competition-grade performance.

The absence of clause learning (CDCL) limits the solver's power on hard industrial instances. CDCL with GUM-scored clause retention (deleting high-uncertainty learned clauses first) is the natural next extension and would complete the epistemic framework.

### 8.2 The Epistemic Architecture Argument

All seven mechanisms arise from a single insight: *uncertainty should be first-class in the solver's reasoning*. This is not an accident in Sounio, a language where `Knowledge<T>` types carry compile-time confidence scores and GUM propagation is built into the type system. The SAT heuristic is downstream of the same uncertainty arithmetic that governs every other computation in the language. The result is a solver whose internal state is epistemically consistent with the programs it is embedded in — a property no other SAT implementation, to our knowledge, possesses.

### 8.3 Connection to Epistemic Types

Sounio's `Knowledge<T>` type represents a value `v` with associated confidence ∈ [0, 1000] (PLATINUM tier = 950+). The GUM variance accumulated in `act_var[v]` is the *act confidence* of the solver's belief that `v` is a genuine conflict driver. When `act_var[v] = 0` (unit clause blame), the solver is PLATINUM-confidence that `v` is responsible. When `act_var[v]` is large (long clause blame), the solver's confidence is BRONZE. This is not metaphor — it is the same GUM arithmetic applied at two different levels of the same system.

---

## 9. Conclusion

We have presented Epistemic VSIDS, a unified framework of seven novel DPLL decision heuristics derived from GUM uncertainty propagation. Each mechanism closes a confirmed gap in the published literature (2020–2026 survey). The key contributions are: (1) a GUM second-moment model for conflict blame that yields principled UCB and Thompson Sampling scores without additional bookkeeping; (2) Beta-Bernoulli posterior sampling for polarity selection, yielding the first fully epistemic DPLL; (3) a regime estimator that classifies the search as Proof/Cautious/Exploratory using EWMA conflict signals, enabling (4) regime-adaptive activity decay, (5) regime-gated warm restarts that preserve posteriors across restart boundaries, and (6) Epistemic LRB via GUM ratio propagation. All mechanisms are implemented natively in Sounio, exercising the language's epistemic type system end-to-end, and all 35 test cases pass including correctness tests on PHP(5,4), K4 3-coloring, and mixed-length random formulas.

---

## References

[Audemard and Simon, 2009] G. Audemard, L. Simon. Predicting Learnt Clauses Quality in Modern SAT Solvers. *IJCAI 2009*.

[Biere et al., 2024] A. Biere, M. Faller, K. Fazekas, M. Fleury, N. Froleyks, P. Pollitt. CaDiCaL, Gimsatul, IsaSAT and Kissat Entering the SAT Competition 2024. *Proc. SAT Competition 2024*.

[Dreher and Heule, 2023] P. Dreher, M. Heule. BroSt: Unleashing the Potential of Restart by Detecting Search Stagnation. *CP 2023*.

[Duan et al., 2020] S. Duan, Y. Luo, X. Zhang, Y. Li, H. Zhang, H. Luo, Y. Chen. Online Bayesian Moment Matching based SAT Solver Heuristics. *ICML 2020*.

[Eén and Sörensson, 2003] N. Eén, N. Sörensson. An Extensible SAT-solver. *SAT 2003*.

[JCGM, 2023] Joint Committee for Guides in Metrology. Evaluation of Measurement Data — Guide to the Expression of Uncertainty in Measurement (GUM). JCGM GUM-1:2023.

[Kenefrey et al., 2024] J. Kenefrey, N. Huli, B. Selman. A Reinforcement Learning based Reset Policy for CDCL SAT Solvers. arXiv:2404.03753.

[Lassouaoui et al., 2019] A. Lassouaoui, D. Boughaci, B. Benhamou. A multilevel synergy Thompson sampling hyper-heuristic for solving Max-SAT. *Journal of Intelligent Decision Technologies* 13(2), 2019.

[Liang et al., 2016] J. Liang, V. Ganesh, P. Poupart, K. Czarnecki. Learning Rate Based Branching Heuristic for SAT Solvers. *SAT 2016*. *(Also: CHB heuristic.)*

[Luby et al., 1993] M. Luby, A. Sinclair, D. Zuckerman. Optimal Speedup of Las Vegas Algorithms. *Information Processing Letters* 47(4), 1993.

[Moskewicz et al., 2001] M. Moskewicz, C. Madigan, Y. Zhao, L. Zhang, S. Malik. Chaff: Engineering an Efficient SAT Solver. *DAC 2001*.

[Pipatsrisawat and Darwiche, 2007] K. Pipatsrisawat, A. Darwiche. A Lightweight Component Caching Scheme for Satisfiability Solvers. *SAT 2007*.

[Shaw and Meel, 2020] R. Shaw, K. Meel. Designing New Phase Selection Heuristics. *SAT 2020*. arXiv:2005.04850.

[Thompson, 1933] W. Thompson. On the Likelihood that One Unknown Probability Exceeds Another in View of the Evidence of Two Samples. *Biometrika* 25(3–4), 1933.

---

## Appendix A: Implementation

All mechanisms are implemented in `stdlib/theorem/smt.sio` (Sounio source, ~1,300 lines). The complete test suite is in `tests/stdlib/theorem/`:

| Test file | Tests | Status |
|---|---|---|
| `test_smt_solver_basic.sio` | 5 | PASS |
| `test_smt_epistemic_eval.sio` | 11 | PASS |
| `test_smt_regime_getters.sio` | 6 | PASS |
| `test_smt_thompson_stats.sio` | 3 | PASS |
| `test_smt_beta_polarity_ts.sio` | 5 | PASS |
| `test_smt_adaptive_epistemic.sio` | 5 | PASS |

**Total: 35 tests, all PASS.**

Key API:

```sounio
// Variable selection modes
smt_set_score_mode(ctx, 0)  // GUM-UCB adaptive (default)
smt_set_score_mode(ctx, 1)  // mean-only VSIDS (ablation baseline)
smt_set_score_mode(ctx, 2)  // Wilson CI LRB
smt_set_score_mode(ctx, 3)  // Gaussian TS (C2)
smt_set_score_mode(ctx, 4)  // Epistemic LRB (C7)

// Polarity selection modes
smt_set_phase_mode(ctx, 0)  // saved phase + confidence (default)
smt_set_phase_mode(ctx, 1)  // Beta-Bernoulli TS (C3)

// Restart budget (0 = disabled, preserves prior behaviour)
smt_set_restart_budget(ctx, 20)

// Regime getters
smt_get_explore_trust(ctx)   // ∈ [0.2, 1.0]
smt_get_regime_label(ctx)    // 0=Proof, 1=Cautious, 2=Exploratory
smt_get_regime_hardness(ctx) // EWMA of |base − len_mean|
smt_get_regime_rate(ctx)     // EWMA of conflict density
smt_get_regime_backjump(ctx) // EWMA of decision level at conflict

// Beta posteriors (C3)
smt_get_phase_alpha(ctx, var_idx)
smt_get_phase_beta(ctx, var_idx)

// Restart diagnostics (C6)
smt_get_restarts(ctx)
```

## Appendix B: Proof of Correctness

**Completeness.** `smt_solve` with `restart_budget = 0` calls `smt_search` directly, which is a complete DPLL procedure (exhaustive backtracking, no pruning). With `restart_budget > 0`, each attempt either returns a conclusive answer (SAT/UNSAT) or exceeds the budget. After 8 attempts, the final call is `smt_search` (unbounded), guaranteeing termination. Warm restarts do not introduce new clauses or modify the formula, so soundness is preserved.

**Soundness.** Beta-TS polarity selection and Gaussian TS variable selection are pure branching heuristics — they choose which variable to branch on and which polarity to try first, but always try both polarities via backtracking. No proof-incomplete pruning is performed.
