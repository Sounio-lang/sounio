<!-- docs:meta
topic_id: repo.docs.research.neurodyn-algebra-b-attribution-prereg-2026-07-06
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.neurodyn-algebra-b-attribution-prereg-2026-07-06
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# NeuroDyn Algebra-B Attribution Preregistration

Date: 2026-07-06

Claim boundary: synthetic non-clinical algebra-necessity assay only. This
document does not make a clinical, biomarker, biological-mechanism,
treatment-response, solved-associator, or broad O-SSM superiority claim.

## Locked Question

Does octonionic non-associative state composition explain the current
fixed-dim6 synthetic signal, or is the effect explained by 8-D capacity and
associative dynamics?

The next cycle must answer attribution before any objective reformulation.
The current reference smoke is:

- artifact: `artifacts/research/neurodyn/synthetic/algebra_b_true_smoke_20260706T221000Z`
- O-SSM balanced accuracy: `54.732143`
- H-SSM balanced accuracy: `50.267857`
- raw-flat leave-site audit: `25.000000`
- decision after four-route gate: `ALGEBRA_B_ROUTE3_SUBTHRESHOLD_REFORMULATION_ALLOWED`

## Four Mutually Exclusive Routes

1. **Algebraic necessity candidate.**
   `O-SSM >= 55%` and `A8-SSM < 55%` and associative projection `< 55%`.
   Only this route permits the 99 pair-label null expansion. If the 99 nulls
   also collapse, a synthetic methods piece may be drafted.

2. **Dimensionality/capacity, not octonionic necessity.**
   `O-SSM >= 55%` and `A8-SSM >= 55%`, or associative projection `>= 55%`,
   or H-SSM is within `3.0 pp` of O-SSM. Reframe as a negative result for the
   algebraic hypothesis.

3. **Subthreshold with reformulation budget remaining.**
   `O-SSM < 55%`. One reformulation of objective/training is allowed only under
   a new preregistration that fixes the seed, threshold, and exact knob changes
   before the run.

4. **Terminal negative for this fixed-dim6 design.**
   Two reformulations have been attempted and `O-SSM < 55%` remains true.
   The fixed-dim6 octonionic algebraic-necessity line is closed.

## A8 Associative 8-D Baseline

The A8 baseline is the direct sum of two quaternionic state spaces:

`A8 = H_left + H_right`

It is associative within each summand and has eight real coordinates, matching
the O-SSM state width. It must use the same input manifest, splits, seeds,
readout shortcuts-off settings, training epochs, learning-rate schedule, and
trace settings as O-SSM. The implementer must match the trainable parameter
budget as closely as the current Sounio model surface permits and report the
count in the run bundle. If exact equality is not yet possible, the run is
exploratory and cannot satisfy route 1.

## Associative Projection Control

The associative-projection control keeps the O-SSM run surface but removes the
non-associative explanatory path by projecting state composition onto an
associative subalgebra/control approximation before readout. The projection
must be specified before execution and must not be chosen after seeing results.
If this projection remains `>=55%`, the octonionic product is not necessary for
the assay even if O-SSM itself is above threshold.

## Null Expansion Rule

Do not run 99 nulls unless route 1 is reached on the attribution smoke. The 99
nulls are pair-label permutations that preserve site, pair balance, input
features, and manifest structure. Passing nulls requires O-SSM BA and AUROC to
exceed all 99 nulls, giving plus-one empirical `p <= 0.01`.

## Offload Provenance Debt

The current xAI offload channel has repeatedly returned
`NO MATHEMATICAL CONTENT TO REVIEW`. Treat this as no external mathematical
review. Until a diagnostic proves the payload includes the intended mathematical
content and receives substantive review, offload logs are provenance records
only, not validation evidence.
