<!-- docs:meta
topic_id: repo.docs.research.octonion-pbpk
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.octonion-pbpk
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Octonion PBPK — Non-Associative Drug Interaction as a Dissertation Contribution

**Date**: 2026-04-13
**Status**: Spec (γ-thread). Not yet implemented.
**Scope**: A candidate 4th novel contribution for the Master's dissertation in biomaterials + regenerative medicine. Unifies the non-associative connectomics research with the paid dissertation work.
**Advisor buy-in required before committing.**

## Motivation

Classical PBPK with DDI models drug-drug interaction as modifier factors on clearance:
```
dC_A/dt = −CL_A · (1 − I_B(C_B)) · C_A + ...
```
Order of administration matters — giving drug B first pre-induces enzymatic inhibition, altering drug A's later exposure. But this order-dependence emerges from the *integrated temporal dynamics*, not from a *structural* invariant of the compartment graph. There is no single number one can read off a patient's regimen that quantifies "how much does order matter for this patient's three co-administered drugs."

**Claim**: the order-dependence of DDI has a natural non-associative algebraic structure that is masked when we model each drug's state as a scalar. Lifting the state to an octonion captures the order-dependence as a first-class geometric object — the associator `[L_A, L_B, L_C] = (L_A · L_B) · L_C − L_A · (L_B · L_C)` — whose squared norm is a scalar "DDI severity" invariant.

This is exactly the same algebraic machinery as the non-associative connectomics work (`experiments/non_assoc_connectomics/`) applied to a different graph (compartments instead of brain regions). One codebase, two clinical domains.

## Classical formulation (recap)

Let `Ω = {drugs, enzymes, transporters}` be the state space. For compartment `p` (liver, gut, blood, brain, ...), the classical state is a vector `C^p ∈ ℝ^{|Ω|}`. Transfer through `p` is a matrix multiplication `C^p' = M_p · C^p` where `M_p` encodes tissue partitioning, enzymatic processing, DDI modifiers. Sequential exposure to compartments `p_1, p_2, p_3` yields `M_{p_3} · M_{p_2} · M_{p_1} · C^0`. Matrix multiplication is associative, so this factorization is order-invariant; DDI order-dependence enters only through *nonlinear* modifier functions, not through the algebra itself.

## Octonion reformulation

Embed the compartment state as an **octonion** `L^p ∈ 𝕆`, where the 8 components encode:

| Slot | Canonical meaning |
|------|-------------------|
| 0 | Scalar: total free drug concentration |
| 1 | Drug A free |
| 2 | Drug B free |
| 3 | Drug C free |
| 4 | CYP3A4 active fraction |
| 5 | P-gp active fraction |
| 6 | Plasma protein bound fraction |
| 7 | Tissue-sequestered fraction |

Compartment transfer becomes **octonion multiplication**: `L^{p,t+1} = L^p · L^{p,t}` where `L^p` is a compartment-specific octonion operator. For rapamycin alone (scalar-only state), `L^p` lives in the ℝ-subalgebra and `L^p · L^{p,t}` reduces to classical scalar flow. For triples including enzymatic interactions, `L^p` has nonzero components on the Fano-plane generators {e₁, e₂, e₃} (drugs) and {e₄, e₅} (enzymes), and multiplication is non-associative.

**The key identity**:
```
classical scalar PBPK = projection of octonion PBPK onto the ℝ-subalgebra
```
In other words: everything the existing Darwin PBPK code does remains correct; octonion PBPK is a *strict extension* that adds structure in the non-scalar components. When only drug concentrations matter and enzymes are fixed, the octonion reduces to the scalar model.

## Associator as DDI severity metric

For a patient on three co-administered drugs with compartment-specific operators `L_A, L_B, L_C`:
```
S_DDI := ‖ (L_A · L_B) · L_C  −  L_A · (L_B · L_C) ‖²
```
`S_DDI` is a scalar invariant depending only on the algebraic structure of the three operators — not on temporal details. It is zero if the operators lie in an associative subalgebra (e.g. all scalar, or all within a quaternion subalgebra {e₀, e₁, e₂, e₃}); it is nonzero precisely when the operators span the full octonion, i.e. when drug-drug-drug interactions engage multiple Fano-plane triples simultaneously.

**Testable clinical prediction**: patients whose regimens have high `S_DDI` should show greater inter-patient variability in exposure outcomes than would be predicted from the sum of pairwise DDI models. The associator measures the "irreducible triple" — the part of DDI that pairwise additive models cannot capture.

## Relationship to existing contributions

Dissertation already has three novel contributions (per `project_masters_dissertation.md`):
1. GUM-through-ODE (JCGM 100:2008 linearized through Tsit5)
2. Compile-time confidence gates (type system prevents low-confidence dosing)
3. ISO uncertainty budget tables for PBPK

Octonion PBPK adds a 4th, *orthogonal* axis: **structural (algebraic) rather than metrological**. The three existing contributions are all about *uncertainty quantification*. Octonion PBPK is about *interaction geometry*.

Contribution 4 integrates with the others:
- GUM variance propagates through octonion multiplication (builds on contribution 1; unblocked by the ζ compiler fix, ref `docs/research/zeta_variance_fix_plan.md`).
- `Knowledge<Octonion<Concentration>>` type becomes the top-level PBPK state (builds on contribution 2).
- `S_DDI` enters the ISO budget as an additional uncertainty source with its own variance contribution (builds on contribution 3).

## Validation plan

The associator must be tested against known clinical DDI data, not just simulated.

**Candidate triples from the rapamycin literature**:
1. `{rapamycin, tacrolimus, cyclosporine}` — all CYP3A4 substrates, all immunosuppressants, frequently co-administered in transplant regimens. Order-of-administration effects documented (MacDonald et al., 2000; Groth et al., 1999).
2. `{rapamycin, verapamil, ketoconazole}` — P-gp + CYP3A4 inhibitors; strong DDI effects known.
3. `{rapamycin, diltiazem, erythromycin}` — moderate inhibitors; should show intermediate `S_DDI`.

**Validation procedure**:
1. For each triple, fit operators `L_A, L_B, L_C` to published AUC-change data for each *pairwise* DDI.
2. Compute predicted `S_DDI` from triple data.
3. Compare predicted vs observed *triple* AUC change (where reported). If `S_DDI > 0` correlates with triple-AUC residual beyond pairwise additive prediction, the model has clinical content.

If correlation is absent — `S_DDI` carries no clinical information — the contribution becomes a theoretical one (unification of PBPK with non-associative algebra) rather than a biomarker.

## Implementation scope

| File | Lines | Role |
|------|-------|------|
| `stdlib/darwin_pbpk/octonion/state.sio` | ~100 | `OctState` struct, `oct_state_new`, accessors |
| `stdlib/darwin_pbpk/octonion/operators.sio` | ~150 | Drug-specific `L_p` constructors (rapa, tac, cyclo); Fano multiplication inline |
| `stdlib/darwin_pbpk/octonion/ddi.sio` | ~200 | `assoc_norm_sq`, `s_ddi` metric, regimen enumeration |
| `tests/run-pass/rapamycin_oct_ddi.sio` | ~120 | Known triples, recovery of scalar PBPK as ℝ-projection, synthetic DDI |
| `tests/run-pass/oct_pbpk_scalar_recovery.sio` | ~80 | Gate: all-scalar octonion → classical result, `S_DDI = 0` |

**Total**: ~650 lines Sounio, no external dependencies. Fits in one dissertation sprint.

## Dependencies on other threads

- **ζ variance fix** (`docs/research/zeta_variance_fix_plan.md`): required if full `Knowledge<Octonion<f64>>` propagation is desired. Can start without it using raw `f64` and defer Knowledge-wrapped state to a later iteration.
- **Non-assoc connectomics Phase 2** (`experiments/non_assoc_connectomics/PROTOCOL_PHASE2.md`): independent. Two applications of the same algebra, no code coupling.
- **`stdlib/darwin_pbpk/ddi/mechanistic_ddi.sio`**: existing scalar DDI model. Octonion version *extends*, does not replace. Scalar code paths remain the recovery target (gate: `rapamycin_oct_ddi` with enzyme components zeroed must reproduce scalar DDI outputs to within 1e-10).

## Open questions (advisor discussion)

1. **Is pairwise-to-triple extrapolation supported in the rapamycin DDI literature?** Without triple-dosing data, `S_DDI` can only be compared to *model predictions*, not measurements. Literature search needed before committing to the 4th contribution.
2. **Is the octonion choice uniquely motivated or arbitrary?** Quaternions suffice for non-commutativity without non-associativity; sedenions introduce zero-divisors. The octonion is picked because it's the *largest normed division algebra* (Hurwitz) and because its non-associativity is the *minimal* deviation from classical — but the argument for "why 8 components and not 4 or 16" for DDI specifically needs stronger grounding.
3. **Does the advisor consider 4 contributions too many for one master's thesis?** Standard is 2-3. If 4 is over-reach, octonion PBPK becomes a follow-on paper after dissertation defense, with the dissertation claiming contributions 1-3.

## Decision gate

Go/no-go on contribution 4:
- **Go** if: advisor confirms clinical interest; triple-DDI literature exists; thesis committee is comfortable with ≥4 contributions.
- **No-go** if: no triple-DDI clinical data exists (the model becomes untestable); advisor prefers focused 3-contribution thesis; or ζ compiler fix is not unblocked in time.

**Timeline**: decision before 2026-05-15 so that implementation fits the 6-month dissertation window ending 2026-09-22.
