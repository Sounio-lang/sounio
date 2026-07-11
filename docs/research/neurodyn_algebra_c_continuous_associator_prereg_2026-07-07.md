<!-- docs:meta
topic_id: repo.docs.research.neurodyn-algebra-c-continuous-associator-prereg-2026-07-07
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.neurodyn-algebra-c-continuous-associator-prereg-2026-07-07
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# NeuroDyn Algebra-C Continuous Associator Fidelity Preregistration

Date: 2026-07-07

Claim boundary: synthetic non-clinical algebra-necessity assay only. This
document does not make a clinical, biomarker, biological-mechanism,
treatment-response, solved-associator, or broad O-SSM superiority claim.

## Coordination Status

This is the Codex-owned execution preregistration for the next NeuroDyn lane.
Claude Code Opus should own theoretical criticism, literature/SOTA review,
reviewer-attack framing, and claim-boundary pressure, but should not edit the
Codex-owned implementation/gate files without an explicit ownership transfer.

## Motivation

Algebra-B asked whether a binary orientation label could establish octonionic
necessity. It reached the attribution precondition but failed the null envelope:
`null_08` exceeded the true O-SSM balanced accuracy under pair-label
permutation. That failure invalidates the Algebra-B promotion claim but does
not by itself refute the more precise hypothesis:

> A non-associative state-space model may preserve a continuous third-order
> associator/path-dependence observable better than associative controls.

Algebra-C therefore changes the primary endpoint from binary classification to
continuous fidelity against a ground-truth associator observable.

## Locked Question

Does the O-SSM hidden trajectory encode the continuous ground-truth associator
signal more faithfully than matched associative controls under held-out splits
and null perturbations?

The target is not "does O-SSM classify a label"; the target is:

> Does the learned hidden state preserve a continuous non-associative quantity
> that A8/H+H, H-SSM, and associative-projection O-SSM do not preserve?

## Primary Endpoint

Primary endpoint: held-out Spearman correlation between an O-SSM hidden/readout
probe and the continuous ground-truth associator scalar.

Amendment after Opus B1 critique: the default scalar is a **per-sequence
realized signed associator component**, computed from the actual generated
trajectory event vectors written into the manifest package. It is not assigned
from the noiseless basis-triple category, and it is not allowed to collapse to a
single fixed sign.

The first executable Algebra-C package must use:

- `--triple-source=continuous`
- `--continuous-jitter=0.01` for the first fixed-component package, because
  larger jitter can move the dominant component and make fixed-component
  rejection sampling fail;
- an explicitly named target component, initially component `6` only if both
  positive and negative realized targets survive audit;
- `--target-assoc-sign=0`, meaning both signs are generated and audited;
- `associator_targets.tsv` as the sole source of target scalars.

The target audit must pass before any model smoke. Minimum audit criteria:

- distinct target values / rows `>= 0.80`;
- tie fraction `<= 0.20`;
- both signs present globally;
- both signs present in every held-out pseudo-site;
- target range non-constant.

Alternative scalars such as associator norm, norm-squared, or full-vector
fidelity are future reformulations, not this default gate.

## Secondary Endpoints

- held-out R2 against the continuous associator scalar;
- sign-AUC for positive versus negative associator sign, only when both signs
  are present after audit;
- calibration slope of predicted associator versus true associator;
- top-k ranking enrichment for high-magnitude associator examples;
- collapse under associative projection.

Binary label balanced accuracy is explicitly secondary and cannot promote
Algebra-C.

## Required Model Surfaces

All surfaces must use the same manifest, splits, seeds, train/test protocol,
shortcut-off readout configuration, and trace settings.

1. O-SSM: octonionic recurrent composition.
2. A8/H+H: direct-sum associative 8-D control.
3. H-SSM: quaternionic/associative capacity control available in the current
   benchmark surface.
4. Associative-projection O-SSM: same O-SSM surface with recurrent composition
   projected to the pre-specified quaternion subalgebra
   `H_123 = span{1, e1, e2, e3}` by zeroing `e4..e7` before the associative
   recurrent-product/readout comparison.
5. Raw/control probes: raw flat features and any direct manifest-derived scalar
   used to audit shortcut leakage.
6. Generic non-hypercomplex capacity control: a real-valued GRU regression
   baseline using the same manifest, leave-site folds, seeds, and target table.
   The run bundle must report trainable parameter count.
7. Higher-capacity generic warning control: `gru_wide`, the same GRU baseline
   with doubled hidden width. If absent, the decision gate must return
   `ALGEBRA_C_WARN_UNDERCONTROLLED`, not a clean candidate.

If a surface is missing or not parameter-matched closely enough to be a fair
control, the result is exploratory and cannot promote.

## Required Shortcut Controls

The first Algebra-C run must disable direct readout shortcuts:

```text
READOUT_ASSOC_SCALE=0
READOUT_MEAN_SCALE=0
READOUT_DELTA_SCALE=0
READOUT_FLAT_SCALE=0
TRACE_HIDDEN_STATE=1
TRACE_READOUT_ALL_FOLDS=1
```

Any direct access to the target associator scalar from the manifest or a trace
derived outside the trained hidden trajectory is a disqualifying shortcut for
the primary endpoint.

## Nulls

Nulls must be run only after the true O-SSM beats all associative controls on
the primary endpoint.

Required null families:

1. pair-label permutation adapted to continuous target permutation within pairs;
2. target permutation preserving site and pair balance;
3. trajectory-preserving continuous-target permutation: keep the realized
   feature trajectories fixed and permute the realized target scalars within the
   preregistered site/pair-balance strata;
4. temporal/order shuffle preserving per-subject feature marginals;
5. temporal reverse, reported separately because reversibility is a mechanistic
   diagnostic rather than an ordinary exchangeability null.

All nulls used for promotion are full-pipeline retrain nulls. Frozen-score nulls
are invalid for promotion. The first bridge may use five nulls for debugging
only; full promotion requires 20 trajectory-preserving continuous-target
retrain nulls plus 99 standard retrain nulls for the primary endpoint.

## Promotion Rule

Algebra-C reaches "continuous associator fidelity candidate" only if all are
true:

- O-SSM held-out Spearman is positive and exceeds every associative control by
  at least `0.10` absolute Spearman.
- O-SSM held-out Spearman is at least `0.10` above the generic GRU baseline.
- O-SSM held-out R2 exceeds every associative and generic control by at least
  `0.05`.
- Associative-projection O-SSM collapses below the O-SSM Spearman margin.
- Raw/control probes do not match O-SSM within the Spearman margin.
- The `gru_wide` control is present; if not, the result is at most
  `ALGEBRA_C_WARN_UNDERCONTROLLED`.
- The 20 trajectory-preserving continuous-target retrain nulls do not reach or
  exceed true O-SSM Spearman or R2.
- The 99-null envelope does not reach or exceed true O-SSM Spearman or R2.

If the true O-SSM fails to beat associative controls, route to negative
attribution: the signal is not octonionic necessity under this assay.

If O-SSM beats controls but nulls reach or exceed the true endpoint, route to
null failure: the continuous target is not stable enough for promotion.

Even if every rule passes, the strongest permitted claim is bounded:

> An octonionic state model recovers a synthetic, generator-matched
> non-associative observable better than matched associative and generic
> controls under leave-site splits and retrain nulls.

This remains an inductive-bias result. It is not evidence of a real
non-associative object in brain data and does not unblock MDD/ADHD, biomarker,
mechanism, treatment-response, or broad O-SSM superiority claims.

## Reformulation Budget

Two reformulations are allowed for Algebra-C. A reformulation must be
preregistered before execution and may change exactly one of:

- target scalar definition;
- probe/readout family;
- train objective for continuous fidelity;
- manifest generator scale/noise regime.

Changing more than one item opens a new lane rather than consuming a
reformulation.

After two failed reformulations, the synthetic continuous-associator necessity
line is terminal negative for this design.

## Real-Data Bridge Is Explicitly Blocked

MDD/ADHD data must not be used for positive O-SSM claims until Algebra-C either
passes or is explicitly abandoned. If Algebra-C passes, the first real-data
bridge should target dynamic state observables, not diagnosis:

- state switching rate;
- dwell time / occupancy instability;
- transition entropy;
- temporal irreversibility;
- higher-order synergy/redundancy summaries.

Clinical diagnosis, severity, or treatment response remain downstream
associations, not first proof targets.

## Required Opus Review

Before any Algebra-C Slurm run, Opus should return a written critique answering:

- Is continuous associator fidelity the right target, or is it still circular?
- Are A8/H+H and associative projection sufficient controls?
- Are the nulls exchangeability-valid for the continuous target?
- What SOTA baseline would make this look weak to a reviewer?
- Which exact wording would be an overclaim even if Algebra-C passes?

Codex may implement and run only after this critique is archived or explicitly
waived by the human author.

## Current Blocker Status

Opus returned `BLOCK_ALGEBRA_C_CIRCULAR_OR_UNDERCONTROLLED` in
`docs/handoff/neurodyn_algebra_c_opus_critique_2026-07-07.md`. Codex may
implement the acceptance-gate edits and request re-review, but may not run an
Algebra-C smoke until the B1 blocker is closed or explicitly waived by the human
author.

The Algebra-B null audit in
`docs/handoff/neurodyn_algebra_b_null_retrain_audit_2026-07-07.md` closed only
acceptance-gate item 1: Algebra-B nulls were full-pipeline retrains. Algebra-C
therefore inherits a real instability warning, not a frozen-score bug.
