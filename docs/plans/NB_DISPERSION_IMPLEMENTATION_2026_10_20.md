<!-- docs:meta
topic_id: repo.docs.plans.nb-dispersion-implementation-2026-10-20
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.nb-dispersion-implementation-2026-10-20
-->

# Count-dispersion diagnostic continuation — 2026-10-20

## State

**Implemented, independently reviewed and typechecked with current-source Madaros.**
Slurm 15747 completed with exit 0:0 and both file checks passed.
See [the typecheck receipt](NB_DISPERSION_TYPECHECK_2026_10_20.md) for identities
and raw evidence. No numerical example, native figure, tests or corpus were run.
The earlier signed-bar checkpoint remains 114/114 on its historical snapshot.

Worktree: `/workspace/sounio-sprint-stats-viz-main6-counts-20261020`.
Branch: `sprint/stats-viz-main-b10c-20261020`.
Fresh remote main incorporated: `b10c046e7b003c4346d5f2d97f47f380081cf06a`.
Merge checkpoint: `4b72c4d70f4532ce01f76b0ed86a72031ce44ddb`.
All accepted sprint changes and archival receipts through `db11ea711d` are retained.
The shared compiler worktree `/workspace/sounio` was not changed.

## Production kernel

Port the production NB2 module and its documentation from historical candidate
`9f8b8617aaa50730444ff0208ae99ee37d5a8fdd` into the refreshed sprint. The historical
candidate's test/reference files were not imported as new acceptance evidence.

The supplied-means estimator preserves its signed alpha, nonnegative projection,
truncation flag, declared parameter count, residual df, Pearson sum and aggregate
weight. It does not fit an NB likelihood or assess NB assumptions.

Two arithmetic changes:

1. An individually underflowed `mu*mu` no longer fails the complete estimate;
   the aggregate rounded weight must remain positive and finite.
2. For zero counts, the Pearson contribution uses `mu` directly, avoiding
   squaring underflow before division. Positive counts retain the previous path.

Independent static review accepted the bounded change. It does not replace
executable evidence. If all individual squared means round to zero, status 5
remains possible even when the mathematical total would be representable.
Intermediate/final arithmetic limits and binary64 count conversion remain explicit.
Status 6 is documented as a reserved defensive guard, unreachable under the
current validation `0 <= parameters_declared < n`.

## Consumer

The example and adjacent documentation connect supplied counts and exposure
information with the existing offset-aware Poisson kernel and the NB2 moment
diagnostic. Plot labels and printed metadata must distinguish observed counts,
fitted means/rates, signed dispersion, its nonnegative projection and unassessed
assumptions. No confidence interval, NB maximum likelihood, RNA-seq differential
expression or empirical calibration claim follows from this consumer.

The example is source for a future native scientific figure. It passed current-source typecheck but has not produced
an accepted SVG or Canvas artifact in this operation.

## Resources and archival continuity

The preceding approved exact-59 offload is complete; see
[SOURCE59_ARCHIVE_EXECUTION_2026_10_20.md](SOURCE59_ARCHIVE_EXECUTION_2026_10_20.md).
It measured 10,733,223,936 bytes available on the shared 3 TiB quota immediately
after cleanup. This is a historical observation under concurrent writers, not
a reservation for new jobs. Quota was not raised and this continuation performs
no further source deletion.

The kernel uses scalar compensated accumulators and borrows fixed-capacity input
arrays; no new per-row staging or jackknife/resampling buffer is introduced.
Those are source-level resource properties, not measured time or peak-RSS gains.

## Remaining acceptance work

Native numerical baseline and example compilation/execution, independent C++23 concordance
including mixed tiny/ordinary means, subnormal zero-count Pearson contributions,
all-rounded-zero weights and declared-parameter boundaries, discriminating old
kernel controls, actual SVG inspection, and resource observations remain open.
Execution belongs in Forge/Slurm, with exact source/compiler/artifact identities
and a storage check before submission. Broader CI, PR/merge and calibration remain
separate checkpoints.

## Next rendering defect located by source audit

`stdlib/viz/viz_html.sio` routes `VIZ_NODE_UNCERTAINTY_BAND` through the plain
polyline emitter, which reads center coordinates and draws no variance band.
The Canvas branch instead calls `epiviz_band` with `d.vars`. This is an existing
cross-backend semantic mismatch, not repaired by the count diagnostic example.
The next isolated rendering slice should define shared checked bounds and
explicit variance-versus-interval semantics before changing either backend.
No NB confidence band should be inferred from the moment coefficient.

Resource review also located eager scene payloads and full-series value bindings
in the ordinary Canvas line/band path. These remain source-level optimization
candidates; no measured compiler copy-elision, speed or RSS result is claimed.
