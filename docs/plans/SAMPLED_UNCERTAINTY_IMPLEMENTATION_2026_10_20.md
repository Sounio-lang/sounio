# Sampled uncertainty implementation checkpoint — 2026-10-20

## Current boundary

**Production source prepared; typecheck, numerical concordance, native execution
and figure acceptance pending. No tests have been added or run for this slice.**

Worktree:
`/workspace/sounio-sprint-stats-viz-main7-bands-20261020`.

The root selected a new worktree based on the freshly observed main085da.
Exact source/compiler/snapshot identities belong in the subsequent root execution
receipt. This note does not turn a prepared source change into executable evidence.
The historical114-entry renderer acceptance remains historical and unchanged.

## Root cause and chosen correction

The existing uncertainty builder checked size/capacity but did not validate
finite means/X or nonnegative finite variances. Its auto domain considered only
means. Canvas used a private20-step Newton sqrt and drew sampled narrow columns,
while SVG routed the same node to a plain center polyline and omitted uncertainty.

The chosen contract preserves the existing scientific meaning:
variance sigma², sampled mean±1sigma, **not a continuous ribbon or CI**.
It uses canonical sqrt, validates all active rows before builder mutation, and
includes the computed uncertainty bounds in automatic Y domains.

A shared scalar helper computes a three-pixel-wide strip and intersects it with
closed plot bounds. It retains interval fragments when the mean lies outside
the viewport. Shared center projection/clipping handles constants without epsilon
padding. Canvas clips before raster loops; SVG uses a2×2 parity pattern and
compressed per-sample geometry. The correction introduces no additional IR
lower/upper arrays; bounds are recomputed through scalar helpers.

Direct n validation precedes access in epiviz_band and its knowledge wrapper.
n0 remains a valid empty input; n1 is one sampled mark. The builder gains Div
and Panic effects in addition to Mut, and direct rendering callers must carry
the appropriate Div/Panic effects. These are source API changes to document,
not evidence of a new allocation or inferential procedure.

## Preservation and limitations

The production owner is separate from this documentation owner. The root owns
Git, integration, compiler selection and any authorized Forge/Slurm operation.
No unrelated compiler worktree is modified. This slice preserves NB2 diagnostic
work and preceding mathematical/genomic/physical/statistical implementations.

No continuous interpolation of uncertainty, coverage probability, hypothesis
test, nonlinear uncertainty propagation or performance superiority is claimed.
The source contract is [UNCERTAINTY_BANDS.md](../../stdlib/viz/UNCERTAINTY_BANDS.md).

## Prepared consumer and pending execution

The root has prepared `examples/viz_sampled_uncertainty.sio`, a figure-producing
example without test assertions. Planned current-compiler typecheck and any
subsequent compilation/execution are pending; no success is inferred here.
The production source review found a degenerate-center SVG line-cap issue for
the production owner to resolve before its final freeze.

## Pending evidence

- Final production source review fixes and exact frozen source hashes.
- Current-compiler typecheck, if coordinated by the root.
- Independently authorized numerical/native/figure verification.
- Caller effect propagation and preservation of prior consumers.
- Actual Canvas/SVG output and semantic metadata inspection.

No execution result is inferred from the old smoke cases or prior114-entry corpus.
The master sprint remains active across statistics, scientific rendering,
mathematics, genomics, physics and measured execution/memory/artifact work.
