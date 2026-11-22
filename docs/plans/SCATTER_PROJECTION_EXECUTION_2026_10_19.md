<!-- docs:meta
topic_id: repo.docs.plans.scatter-projection-execution-2026-10-19
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.scatter-projection-execution-2026-10-19
-->

# Checked scatter projection and viewport execution — 2026-10-19

## Acceptance boundary

**Accepted native checkpoint:** Forge/Slurm15650 passes111/111, zero
failures/skips/errors, with three passing summary layers. All108 previous
entries remain accepted. Corpus phase340s; overall19:11:12–19:16:53UTC341s.
Slurm reports COMPLETED0:0, elapsed5:42,2allocated CPUs, TotalCPU11:18.137,
batch MaxRSS7152996K. These are whole two-worker compile/run observations,
not a kernel/render runtime, aggregate memory total or performance comparison.

Active worktree: /workspace/sounio-sprint-stats-viz-main3-20261019, starting
HEAD5a5f852. Runtime source snapshot:
370cc50246991bc39421b4530b8ec55b10b3dbd67cc2102ab710e1c055cf009e.
Compiled manifests are authoritative for this snapshot; final documentation
edits occur after that archive and are not claimed as native inputs.

## Scientific contract and actual repair

Legacy Canvas scattered constants through a padded domain while SVG substituted
a span of1, yielding different positions for singleton/constant data. Neither
scatter renderer shared a checked finite projection or plot glyph scissor.

The repair adds one scalar projector without an IR dependency. It validates
finite samples, finite ordered domains and finite spans before conversion;
constant axes accept only exactly equal samples, with integer offset extent/2.
Y is origin+height-height/2. Nonconstant mapping retains the existing truncating
pixel rule. Bounded geometry and normalized coordinates are explicit limits;
an overflowing raw span is invalid rather than an arbitrary-range claim.

Scatter builders derive auto bounds from paired finite rows while preserving
raw n and arrays. Nonempty all-invalid input fails transactionally. Invalid
points are omitted before floating-to-integer conversion. Legacy Canvas crosses
and SVG radius3 circles retain their shapes; their center projection is shared.

The checked raw-linear viewport validates domains, layout and opaque RGB8 style
before mutation. It emits only centers within the closed raw domain. Outside
centers are omitted, never clamped to fabricate boundary observations. Radius3
glyphs are contained in closed pixel-center bounds[x,x+w]×[y,y+h]. Canvas uses
the integer disk dx²+dy²≤9 with a per-pixel scissor; SVG uses radius3 circles and
a matching user-space pixel-cell clipPath. Clipped Canvas scatter draws no
implicit axes. Legacy axes use the same safe projector for zero lines.

Visible/outside/invalid counters retain raw row accounting, alongside the old
scatter n comment. Log/symlog scatter and bar/ribbon clipping are not added.
The raw viewport setter rejects transformed flags; logzero is not replaced by
epsilon. Full contract: [CLIPPING.md](../../stdlib/viz/CLIPPING.md).

## Numerical and visual evidence prepared

The independent C++23 reference contains64 projection cases. Root reproduced its
report and generated Sounio fixture byte-identically. It checks constant,
zero/large/subnormal, invalid/nonfinite, outside, layout and truncation semantics.
The scientific consumer uses actual checked QR residuals, not fabricated fitted
statistics. It also checks singleton/constant and invalid-row behavior, plus an
exhaustive small Canvas footprint.

Production and reference computation follow ADR009. No Sounio compiler or
native execution was run in the Mac/workspace by the implementation agent.
Darwin sounio_verificado was called before new Sounio; its candidate failed
compilation and was corrected. No semantic acceptance is attributed to that call.

## Completed real-before control

Job15649 used the previous renderer with the discriminating singleton control.
Native compilation/execution completed; Canvas assertions and a pass marker
preceded the failed SVG expectation. Actual SVG center was(40,110), whereas
the shared-center expectation is(140,70). Result:0/1. This is a real source-before
failure, not a deliberately altered expected value.

Root changed queued resource requests from4CPUs to2 (including CPU-per-task and
minimum CPUs per node), retaining16G, without source changes. The job ran10s;
batch MaxRSS3191264K includes compilation/execution and is not a renderer-only
measurement or a before/after resource improvement.

## Completed focused and artifact evidence

New native entries pass in the full run: singleton entry100 (10s),64-case
reference entry101 (1s), and viewport entry102 (11s). The viewport also passes
the exact-boundary rectangle annotation check. The final full111-entry result above includes these entries.

The final independent C++23 checker passes the actual native figure:5 circles,
cropped panel visible2/outside2, and mixed panel invalid2. Mutated SVG center,
clip width and removed clipping group copies each exit2. The exact-boundary
artifact tests clip rectangle serialization. Root also checked scientific and
boundary SVG XML using xmllint, both exit0. This is not browser raster proof.

- SVG SHA256:f48aa8ca3bcee8e622b4aee18c5954367e7c7683c184e8f056f92f948304b272
- HTML SHA256:e5b285af87bd45d86fb50a126c9df8d7b44ea12b4b7ec3c37f65832e339fb40d
- Boundary SVG SHA256:6607351d2d866e3d5a6ce77c0c09132a179b6d072d9dce8a2434f4a120b8ae1e

Scissor fault control15651 compiles384functions, then exits1 in the selected
viewport fixture:0/1, zero skips,7s. Only qx/qy pixel-bound checks are removed.
The original fixture therefore discriminates glyph-footprint containment.

Root rebuilt the final C++23 generator/checker warning-free and reproduced the
generated reference fixture byte-identically (cmp0; ROOT_CPP_ORACLES_READY).

## Snapshot and checker custody

Final source snapshot is recorded above. The full111-entry summary and compiled
manifests identify the executed production/native fixture sources. Final SVG
checker source has7deb490 prefix; the runtime packet's251234-prefix checker
differs only in formatting/casts. Root compiled and ran the warning-free final
checker separately; production/native fixtures are unchanged.

The scissor native fixture execution took7s; its whole control job took10s
with MaxRSS3546896K. Before control15649 took10s with MaxRSS3191264K.
Neither control resource observation demonstrates a performance improvement.
The source archive precedes these final documentation edits.

## Scope and continuation

This is bounded rendering correctness, not evidence of universal projection,
identical SVG/Canvas antialiasing, performance superiority, nonlinear nice ticks,
or completion of the mass stdlib sprint. The scalar projector allocates no heap;
scatter construction adds a paired-finite scan before the existing series copy,
so no faster construction is claimed. Mathematics conditioning/convergence,
statistical calibration, count-domain integration, physical units/invariants,
and execution/memory/artifact improvements remain separate active tracks.
CI, merge and publication remain separate from native acceptance.

## Publication isolation

After acceptance, unrelated untracked NB-dispersion work appeared in the
validation worktree. It was preserved in place. Publication uses
`/workspace/sounio-sprint-stats-viz-main4-scatter-20261019`, branched from the
same5a5f852 parent. Exactly the13 completed scatter paths were copied and their
SHA256 identities checked before staging. Production and Sounio fixtures remain
identical to the accepted runtime inputs; final C++ checker identity is recorded
separately above. The unrelated NB paths are absent from the accepted runtime
manifest and from this publication. The original worktree is not claimed clean.
