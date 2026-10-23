<!-- docs:meta
topic_id: repo.docs.plans.sampled-uncertainty-implementation-2026-10-20
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.sampled-uncertainty-implementation-2026-10-20
-->

# Sampled uncertainty implementation checkpoint — 2026-10-20

## Current boundary

**Current-compiler typecheck and native SVG demonstration execution completed.
Numerical concordance and Canvas runtime validation remain pending. Structural
and visual inspection of the native SVG demonstration is complete. No tests have been added or run for this slice.**

Worktree:
`/workspace/sounio-sprint-stats-viz-main7-bands-20261020`.

The worktree began on the main085da line. Frozen checked source commit:
52ba99b87d49da04e97524fa0d8911b1d1d4d46b. Slurm15776 completed0:0 in26s with
18 TYPECHECK_OK targets and native SVG compilation/execution exit0.
Compiler built from current source in15772, attested SHA256:
0c896942018ff4be9a86d69f30e3a8aadac955f32451c445a07fbf502ae37fc3.
It was reused only after exact unchanged bin/scripts/self-hosted/stdlib tree
verification. These are coordinated native evidence, not a full corpus run.
The historical114-entry renderer acceptance remains historical and unchanged.

See the root-owned [execution receipt](SAMPLED_UNCERTAINTY_EXECUTION_2026_10_20.md)
for exact package/compiler custody and the bounded visual-inspection outcome.

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
closed plot bounds. X centers must lie inside the closed X domain; outside centers
are omitted. In Y, interval fragments remain visible when the mean lies outside
the window but its uncertainty interval intersects it. Shared center projection/clipping handles constants without epsilon
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

## Executed and inspected native SVG consumer

The root's `examples/viz_sampled_uncertainty.sio` is a figure-producing example
without test assertions. It passed current-compiler typecheck, compiled and
executed with exit0 in15776. Native HTML SHA256:
82b72b8c718a8ab2743c038d289f4c704f4bb541f6e49a3a5cbf2ad4015f614e.
The root downloaded the actual native HTML and extracted SVG bytes between the
opening/closing tags unchanged. xmllint --nonet --noout returned0. QuickLook
rendered the artifact and the actual preview was inspected: title/tick labels
were legible, the negative Y label remained visible, sampled strips and centerline
were visible, and pattern/clip/semantic metadata were present.
The production review's degenerate-center SVG line-cap issue is closed by
explicit round caps, confirmed by fresh source readback. The public secondary
raster helper also checks bounded integer endpoints/scissor before subtraction.
Source identities at review: viz_html.sio
128ece5e463ac6d63ed7b8951d6ccfe26a2fa1f88e142e2bb8b8aba2b3f9d2c0;
uncertainty.sio
30ebd3d0f4738b7cf5d041063e988441a718c65c964d35eb1cb882ecdcbb05db.
These are source review identities, not execution receipts.

Public HTML emission and Canvas scene rendering now propagate Panic; the
workbench build path propagates Mut/Div/Panic. The source API effect change
remains explicit. The example was given additional left margin for scientific
Y-axis labels. Native generation and bounded SVG inspection are complete. Numerical agreement,
Canvas runtime behavior, pixel parity and coverage are not inferred from them.

## Pending evidence

- Independent numerical concordance and explicit boundary-case verification.
- Canvas runtime validation and semantic pixel/geometry checks.
- Full prior-consumer corpus validation and repository CI/merge.

The18 typechecks cover the selected source targets, not every possible external
caller. No tests or full corpus were run for this checkpoint.

No execution result is inferred from the old smoke cases or prior114-entry corpus.
The master sprint remains active across statistics, scientific rendering,
mathematics, genomics, physics and measured execution/memory/artifact work.
