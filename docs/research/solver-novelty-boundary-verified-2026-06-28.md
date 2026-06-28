# Solver Novelty Boundary — Verified

Date: 2026-06-28
Branch: `recover/solver-gpu-arc`
Method: two adversarial deep-research passes (`wf_86c396d4-e10`, `wf_4a066e7f-8fc`;
~200 agents total, 3-vote verification, 2/3-to-kill). This supersedes the aspirational
"FIRST-IN-CLASS / GUM uncertainty" wording previously in `stdlib/theorem/smt.sio`.

## The only defensible claim is the CONJUNCTION

Every individual ingredient of the Sounio SMT heuristic is established prior art:

| Ingredient | Prior art (primary source) |
|---|---|
| bandit / learning-rate **variable** selection | LRB (SAT 2016) + CHB (AAAI 2016), both ERWA, per-decision |
| joint **variable+polarity** Bayesian learning | Online Bayesian Moment Matching (Duan et al., ICML 2020) + SAT/SETTA 2022 in-processing follow-up |
| **Thompson sampling** inside CDCL | Li et al. arXiv 2404.03753 — but only the binary RESET decision |
| **regime-gated** heuristic switching | Cherif et al. CP 2021 (VSIDS↔CHB at restart); vivification-ratio gating arXiv 2112.06917 |

**SAFE wording:** "a *per-decision* (not initialization, not in-processing, not
restart-gated) bandit/Thompson-sampling mechanism applied *jointly* to BOTH variable and
phase/polarity selection, with a regime-gated exploration coefficient, in one bounded
DPLL(T) core." Novelty rests on cadence + joint axis + integration; a hostile reviewer can
compress it to "a cadence variant of OBMM/LRB + restart-gated switching," so any
published claim must be backed by an ablation isolating each conjunct's marginal value.

**UNSAFE wording (all refuted by primary sources — do not use):** "first bandit / first
Bayesian / first Thompson-sampling / first uncertainty / first epistemic heuristic in
SAT," "fully epistemic DPLL," any "SOTA," and **any "GUM / epistemic posterior variance"
framing for the score** (see below).

## The variance critique is VALID — `act_var` (mode 0) is not a posterior variance

No surveyed SAT/SMT heuristic scores variables with a second-moment `mean + β·sqrt(var)`
term. UCB1-in-SAT is a count-based confidence width; VSIDS is a first-moment EMA; BMM uses
the second moment only to fit a Beta and ranks by posterior CERTAINTY (which CONCENTRATES
as evidence accrues). Sounio's mode-0 `act_var` accumulator **GROWS** with conflicts → it
is functionally a recency-weighted clause-length signal (Jeroslow-Wang / MOM lineage),
**not** a concentrating posterior variance. The "GUM law of propagation" derivation in
`smt.sio` is design *motivation*, not a verified property. (Mode 4's `σ²(LR)=LR/decided`
already shrinks as `1/sqrt(decided)` and is closer to a genuine concentrating form.)

## The open frontier to break: variance-aware (UCB-V) bandit branching

Confirmed genuinely open: **no SAT/SMT/CP solver has ever used UCB-V (Audibert/Munos/
Szepesvári 2009) or UCB1-Tuned (Auer 2002) empirical-variance exploration for variable or
polarity selection** (the Fröhlich SLS-bitvector counterexample was refuted 0-3).

UCB-V gives the principled, concentrating template: `score = mean + sqrt(2·V·ζ·log t / n)`
with `V` = empirical variance; the bonus SHRINKS as `n` grows. **Actionable redefinition:**
replace growing `act_var` with a per-variable Normal-Gamma / empirical posterior over a
BOUNDED conflict-relevance/learning-rate reward in `[0,1]`:
`score = mean + β·sqrt(emp_var · log(t) / n_v)`.

**The core technical risk (the actual research nut):** UCB-V assumes i.i.d. STATIONARY
rewards; SAT rewards are NONSTATIONARY → needs decay; but decay RE-INFLATES the variance
bonus and can reintroduce the growing-accumulator pathology. Resolving that tension
(sliding-window / discounted empirical-Bernstein UCB-V) IS the genuine contribution.
**Proof experiment:** ablation `β=0` vs `β>0` on instances stratified by variable
discriminability (variance-aware bandits only help in the small-gap/high-variance regime),
each UNSAT shipping a checked LRAT receipt.

## Certificate path for a credible Level-3 result

- **SAT (cheapest, mature):** emit DRAT → `drat-trim` elaborate to LRAT → check with
  **`cake_lpr`** (formally verified to x64 machine code, CakeML/HOL4; SAT-Competition
  checker). UNSAT-only; SAT certified by a model; LRAT-direct emit skips drat-trim.
- FRAT = lower-overhead intermediate (~6% solver slowdown), later upgrade. VeriPB/CakePB =
  no benefit unless cardinality/Gaussian/symmetry reasoning.
- **SMT:** Carcara checks Alethe; QF_LRA gets a cheap **Farkas** receipt (`la_generic`,
  GMP rationals) but Carcara is UNVERIFIED Rust (trust via Isabelle/Coq), not cake_lpr
  calibre. **QF_LIA is an open gap** (`lia_generic` is a "hole" Carcara skips) → restrict
  Sounio's verified-receipt claims to **QF_LRA / Farkas**, not QF_LIA.
- **Benchmark framing:** a fixed ~50–100 instance slice of SAT-Comp 2024/25 main-track
  within the 1024-var / 4096-clause cap + structured families, each UNSAT shipping a
  checked LRAT receipt; report as "honest ablation with checked certificates," NOT SOTA
  speed (the bounded solver will lose on raw speed vs CaDiCaL/Kissat/z3/cvc5).

## Primary sources
LRB cs.uwaterloo.ca/~ppoupart/...; CHB AAAI 10439/10440; OBMM mlr v119/duan20c +
Springer 978-3-031-21213-0_14; TS-reset arXiv 2404.03753; Cherif CP 2021 LIPIcs.CP.2021.20;
vivification arXiv 2112.06917; VSIDS-as-EMA arXiv 1506.08905; MAB-for-SAT survey SoCS 2025;
UCB-V inria-00203487; UCB1-Tuned Auer 2002; cake_lpr SATComp2025 + TACAS21; FRAT arXiv
2109.09665; Carcara TACAS 2023.
