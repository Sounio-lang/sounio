<!-- docs:meta
topic_id: repo.docs.plans.bars-execution-2026-10-20
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.bars-execution-2026-10-20
-->

# Scientific bar rendering — native execution, 2026-10-20

## Accepted checkpoint

Remote main incorporated: `d7077ef1ed21df355eee57ef3479981d554d8ee4`.
Production HTML recovery: `b9cfcc7d98138b3507a732fd16e31db40c65e3fd`.
Final fixture/checker margin correction: `f17bb2c67b`.
Publication branch: `sprint/bars-main-d7077-20261020`.

| Slurm job | Immutable run | Observed result |
|---|---|---|
|15712|main-refresh-20261020-v3|111/111 pass, COMPLETED 0:0|
|15714|bars-20261020-v3|114/114 pass, COMPLETED 0:0; superseded figure margin|
|15718|bars-20261020-v5|114/114 pass, COMPLETED 0:0; final margin-corrected consumer|
|15713|bar-before-20261020-v3|Compiled and ran; Canvas assertions and BAR_SIGNED_PASS completed; new SVG rectangle expectation failed|
|15715|bars-control-20261020-v2|Compiled; removing only horizontal intersection caused the viewport native assertions to fail|

All positive corpus entries passed; zero failures/skips/errors. Runtime receipts record `clean-v1`, UID/GID1000. The final corpus phase took235s; Slurm elapsed3:57. These are whole compile/run job observations, not isolated renderer benchmarks.

Final snapshot:
`4ecec0bfe0eaa1e87c6cdccc24d86ddb68987a753e8a4b03fe2c9ca2a04617f7`.

Baseline snapshot:
`5baaaf18a68a46b95f9c20818e7f9ef64e1f7d5052ec65799635b5489a1e6bf6`.

The focused package is not a Git checkout. Payload manifests, snapshots and source hashes establish its identity. Production bar/IR/Canvas/HTML hashes were compared between the packet and publication worktree. Both canonical and basic_math fixture aliases are content-identical. The committed compiler materialized a125051232-byte ELF with SHA256:
`125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f`.
No compiler rebuild/fixed-point or full compiler CI result is inferred from these gates.

## Implemented scientific semantics

Shared scalar bar geometry drives Canvas and SVG: actual sample x positions, true zero baseline for signed values, paired-finite auto-domain, zero included in automatic Y bounds, transactional rejection of nonempty all-invalid input, and explicit invalid/empty/visible statuses.

The opt-in raw-linear viewport clamps tip and baseline separately to Y bounds and intersects the whole horizontal rectangle. Outside centers retain overlapping bar footprints. Width comes from the first two paired-finite input rows before crop; conservative finite projection limits and width2..40 are explicit. This policy is documented in [BARS.md](../../stdlib/viz/BARS.md), including constant/empty cases and resource limits. It is not a histogram bin estimator or grouped/stacked-bar API.

The independent C++23 geometry reference covers83 cases:61 rectangles and22 widths. The regenerated Sounio fixture matched byte for byte and passed inside the native corpus. The positive-window consumer uses synthetic OLS residuals and explicitly labels the shift+8.5mV and Y crop[5,10]; no inferential interval is implied.

## Actual native output and independent checks

- [Final native SVG](artifacts/bars-20261020-v5/viewport.actual-native.svg)
- [Final native HTML](artifacts/bars-20261020-v5/viewport.actual-native.html)
- [Original native stdout](artifacts/bars-20261020-v5/viewport.native.log)
- [Checker output](artifacts/bars-20261020-v5/checker.native-log.stdout)
- [Final receipt](artifacts/bars-20261020-v5/runtime-receipt.json)

The independent C++ checker accepts the original native log and extracted SVG:8 rectangles,4 signed,4 shifted-positive cropped and2 retained outside-center footprints. It also checks15 actual tick texts with explicit `textLength`, including horizontal extent and vertical font/baseline bounds. Non-textLength labels receive no approximate font-width guarantee.

The original v3 SVG has tick x positions -16/-8 that can clip minus signs. The final fixture translates only the left plot, its axes/heading and two Canvas probes by64. Data, scales, extents, right panel and production code are unchanged. The updated C++ checker rejects v3 specifically with `tick text bounds`, before bar geometry comparison. A labelled NEGATIVE-ONLY copy of final v5, moving the first bar x163.5→164.5, is rejected with exit2 and `bar geometry/color 0`. Neither negative file is presented as native output.

Final SVG SHA256:
`91aa6ac23cc64996ae6606bc7a3644134b6020b5aa7f6681aa10a5100f5067ab`.

Final HTML SHA256:
`526692d81a8e8f557f5719056c0f78a1e1d799c563d26dd96b9c0817dee5acaa`.

The final SVG also passed `xmllint --nonet --noout` on the downloaded custody copy. This checks artifact XML structure; all Sounio compilation/execution occurred in Slurm. Canvas/SVG antialiasing equality and a complete publication layout engine are outside this bounded checkpoint.

## Failures preserved and corrected

Jobs15706/15707/15708/15709 used physically truncated HTML ending in `let ver` inside an axis function. They are excluded from acceptance and discriminating-control claims. The complete suffix was recovered directly from the accepted versioned blob after exact partial-function prefix comparison. Bar/scatter/forest and upstream changes were retained; full-file streams and checksums replaced tool-output capture.

The first margin packet `bars-20261020-v4` was made readonly, causing Forge validation cleanup to fail before Slurm submission. Its source archive remains preserved. Owner-write permissions were restored without changing file contents/manifests, and final runv5 submitted normally. Packet freeze relies on verified hashes; runtime source directories must support compiler materialization and validation cleanup.

## Storage and resource scope

[The authorized offload](ARTIFACT_OFFLOAD_EXECUTION_2026_10_20.md) removed107 historical source archives after two complete restore-verified copies and encrypted GitHub custody. It retained logs/results and wrote an A/B journal before and after each exact removal. Including the earlier one-file step,1262940884 logical bytes were offloaded; quota remained3TiB. Observed availability after the107 offload was96468992bytes. New validation snapshots consumed most of this margin; final availability observed was4194304bytes. This is insufficient headroom for a massive ongoing batch and requires further reviewed retention work.

Final batch MaxRSS7340964K includes compiler, harness and native workloads. It does not measure isolated bar rendering or establish comparative speed/memory superiority. Shared geometry uses scalar storage and renderers avoid a coordinate buffer; broader execution/memory profiling remains open.

The sprint continues across statistics, scientific rendering, mathematics, genomics and physics. This checkpoint accepts the bounded bar slice and updated111-case baseline. Conditioning/convergence, inferential calibration, broader count/variant consumers, physical invariants and resource/retention work remain active. GitHub CI and merge are separate pending states.
