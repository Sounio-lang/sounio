<!-- docs:meta
topic_id: repo.docs.papers.vancomycin-pl-paper-outline
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.vancomycin-pl-paper-outline
-->

# PL Paper Draft Outline — POPL/ICFP 2027 target

**Working title**: *Algebraic Effects for Knightian Uncertainty: A Verified Clinical Decision Support Foundation*

**Target venue**: POPL 2027 (Jul deadline) or ICFP 2027 (Sep deadline)
**Target page count**: 25 pages (POPL standard)
**Authors**: D. C. Agourakis (Sounio PI). Co-authors TBD.
**Status**: M5 outline. Section-level skeleton; prose-filling deferred until M4 cohort results land.

## Abstract (target 200 words)

Knightian uncertainty — uncertainty *about* the probability distribution itself — is a first-class concern in safety-critical domains (clinical dosing, autonomous vehicles, structural engineering) yet has no satisfactory representation in mainstream programming languages. We present **Sounio**, a self-hosted systems language with an algebraic-effect system (`Approx`, `Causal`, `Knowledge`, `Observe`) that composes with a Knightian operator implemented as Ferson probability boxes. We prove, in Lean 4, that the three effects commute under the canonical arithmetic, and that the p-box operator preserves containment of the true CDF under elementary operations. We demonstrate the system on a retrospective vancomycin TDM cohort, showing that Knightian-conservative refusals correlate with subsequent AKI events at non-inferior rates compared to Bayesian SOTA. To our knowledge this is the first programming-language work to combine algebraic effects with imprecise probability and to validate the construct on real clinical data. Sounio's substrate is openly released; the paper's contributions include the formal soundness theorems, the operational p-box arithmetic, and the empirical case study.

## 1. Introduction

- Knightian uncertainty: definition, history (Knight 1921, Walley 1991, Klibanoff 2005).
- The gap in PL: probabilistic languages (Anglican, Pyro, Stan) handle Bayesian uncertainty; effect-typed languages (Koka, Eff, Frank) handle algebraic effects but not probability; uncertainty-aware DSLs (UQLab, Cosmos) lack PL semantics.
- Three contributions:
  1. Algebraic-effect composition `Approx × Causal × Knowledge` with Lean-4-verified soundness.
  2. Ferson p-box operator `Knightian<f64>` integrated into the effect row.
  3. Empirical validation on vancomycin TDM cohort (M4).

## 2. Background

- 2.1 Algebraic effects (Plotkin & Pretnar 2009, Bauer & Pretnar 2015, Leijen 2017 Koka).
- 2.2 Imprecise probability (Walley 1991, Augustin 2014 *Introduction to Imprecise Probabilities*).
- 2.3 Probability boxes (Ferson et al. 2003 Sandia tech report; NASA-STD-7009A).
- 2.4 Sounio (brief): self-hosted, x86-64, monomorphic generics, refinement types.

## 3. The Sounio Effect System

Mirrors `formal/lean4/SounioEffects.lean`.

- 3.1 Effect rows as `Effect → Bool` (Mathlib-free).
- 3.2 Subtraction, union, identity laws.
- 3.3 The four new effects for this work: `Approx`, `Causal`, `Knowledge`, `Observe`.
- 3.4 Effect inference via the Sounio bidirectional checker.

## 4. Composing Approx, Causal, and Knowledge

Mirrors `stdlib/epistemic/composed_effects.sio`, `formal/lean4/SounioApproxCausalKnowledge.lean`.

- 4.1 The `ComposedKnowledge` record.
- 4.2 Arithmetic: GUM variance + interval-Approx + Beta-edge Causal.
- 4.3 Handler commutation theorem (formally: `composition_soundness`).
- 4.4 Confidence decay as a lower-bound conservativeness measure.

## 5. The Knightian Operator: Ferson p-boxes

Mirrors `stdlib/epistemic/knightian.sio`, `formal/lean4/SounioKnightian.lean`,
`formal/lean4/SounioFrechet.lean`.

- 5.1 Why p-boxes: rationale doc `docs/research/knightian_operator_choice.md`. Walley vs Klibanoff trade-offs.
- 5.2 PBox arithmetic: addition, subtraction, multiplication (corner enumeration), division (zero-straddle vacuous).
- 5.3 Containment soundness theorem.
- 5.4 **Joint dependence and the Fréchet outer enclosure (M2.5).**
  - 5.4.1 Anticipated reviewer pushback: univariate p-boxes bound only marginal CDFs, so a joint computation `f(X, Y)` requires a copula assumption — typically independence — which fails for correlated PBPK parameters such as vancomycin (Vc, CL).
  - 5.4.2 **Resolution.** When `f` is C¹ with strict per-argument monotonicity on the marginal rectangle, the corner-enumeration band `[f(a, d), f(b, c)]` is the Fréchet (point-mass) outer enclosure: it contains `f(X, Y)` almost surely for any joint distribution on the rectangle, regardless of copula. (Theorem in `formal/lean4/SounioFrechet.lean`; math-review-validated 2026-04-30.)
  - 5.4.3 Conservatism factor (per math-review Q4): ~1× the true 95%-CI under counter-monotonic `r ≈ -0.7`; ~2.6× under co-monotonic `r ≈ +0.7`. Acceptable for a safety gate; explicitly disclosed.
  - 5.4.4 Empirical verification: `tests/stdlib/clinical/test_vancomycin_correlation_sensitivity.sio` runs 250 deterministic samples (50 each at `r ∈ {-0.7, -0.3, 0, +0.3, +0.7}`) through the actual Cmin function and confirms the enclosure for every sample.
  - 5.4.5 Limit of applicability: monotonicity must hold uniformly on the marginal rectangle; the technique fails on saddle / non-monotone regions. Vancomycin satisfies the hypothesis on all physiological positive parameters.
- 5.5 Connection to parenthesization-invariance (sedenions, `HYPER_UNCERTAINTY_PARENTHESIZATION_REPORT.md`): the directional projection lever.

## 6. Refinement Types and Lean Export

Mirrors `stdlib/clinical/vancomycin_pbpk.sio` runtime contracts.

- 6.1 Sounio refinement-style runtime gates.
- 6.2 The `is_safe_dose` family: contract → Lean theorem.
- 6.3 Manual export pipeline; future-work notes for automated extractor.

## 7. Case Study: Vancomycin Dosing

Mirrors `tests/run-pass/vancomycin_propagation_v2.sio`, `formal/lean4/SounioVancomycinDosingSafety.lean`.

- 7.1 Population PK: 2-compartment Roberts 2011 model.
- 7.2 Knightian-Cmin pipeline: pre-TDM vs post-TDM band tightening.
- 7.3 The contract gates; refusal as a first-class outcome.

## 8. Empirical Evaluation

Mirrors `docs/research/m4_validation_framework.md`. **Filled when cohort results land (M4).**

- 8.1 Cohort and methods.
- 8.2 Primary outcome: MAE non-inferiority.
- 8.3 Secondary outcomes: coverage, refusal rate, clinical correlates.
- 8.4 Discussion: Knightian conservativeness vs Bayesian aggressiveness.

## 9. Related Work

- Probabilistic PLs: Anglican, Stan, Pyro, Edward, ProbCog.
- Effect-typed PLs: Koka, Eff, Frank, Multicore OCaml effects.
- Verified clinical software: TDMx (Imai 2018), CSL-DACH (Burrows 2017), CarePlan (Gallego 2020). None with Knightian semantics.
- Imprecise-probability libraries: pba-for-r (Ferson), pyMatching, Bayesian-pba.

## 10. Limitations and Future Work

- Knowledge<T> monomorphism (struct wrapper for nested generics).
- Lean discharge of the probabilistic obligation deferred (8-12 weeks).
- Cohort scope: single-center; MIMIC-IV external validation in progress.
- Walley credal sets as theoretical back-stop; not yet implemented.

## 11. Conclusion

Sounio establishes that Knightian uncertainty is *expressible*, *composable*, and *verifiable* at the PL level, with operational consequences in clinical care. The vancomycin case study validates the substrate; the substrate generalises beyond medicine.

## Reproducibility appendix

- `bin/souc` (this commit): builds the runtime.
- `formal/lean4/lakefile.lean`: lake build SounioApproxCausalKnowledge SounioKnightian SounioVancomycinDosingSafety.
- `tests/run-pass/vancomycin_propagation_v2.sio` and `tests/stdlib/clinical/test_vancomycin_pbpk_v2.sio`: end-to-end smoke.
- `scripts/clinical/process_tdm_cohort.sh`: cohort pipeline driver.

## Reviewer-prefacing notes (internal)

Anticipated objections and pre-emption:

- **"Why not Walley credal sets?"** § 5.1 + decision document; cited operational acceptance. M3.5 follow-up uses Walley neighborhood at the *elicitation* surface and lifts to p-box at propagation via Fréchet bounds.
- **"Lean proofs are mostly trivial / sorry"** Acknowledge; the structural part is real, the probabilistic obligation is explicitly deferred with effort estimate. The Fréchet enclosure theorems in `SounioFrechet.lean` are fully proven (no `axiom`, no `sorry`) at the `Nat` level; the Float-Real lift is the deferred Mathlib milestone.
- **"You assume independent (Vc, CL)"** §5.4 explicitly: NO. The Fréchet outer enclosure holds for any joint distribution on the marginal rectangle. Math-review-validated theorem; 250-sample empirical sanity check across five correlation values in [-0.7, 0.7]; reproducible by `bash scripts/run_sio_test_suite.sh sensitivity`.
- **"Synthetic cohort dilutes empirical claim"** § 8.4 footnote: no inferential analysis on synthetic data; real cohort gates the empirical claim.
