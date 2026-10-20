<!-- docs:meta
topic_id: repo.docs.plans.scientific-auto-axes-execution-2026-10-17
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.scientific-auto-axes-execution-2026-10-17
-->

# Automatic scientific axes and shared line domains — 2026-10-17

## Accepted execution

Final run `stdlib-nice-20261017-final1`, Slurm **14879**:
**COMPLETED 0:0**, elapsed **00:01:10**.
JUnit **20 tests, 0 failures, 0 skips, 0 errors**.
Snapshot SHA-256:
`abee462d7704cc3a8ebca848ee9687b882934dc5138eed7b5bb9fd47827627e6`.

Pinned committed Madaros v0.80 ELF SHA-256:
`125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f`.
Compressed compiler SHA-256:
`feaaf7fe181f1ee3efd30e7fe23b1709937ea7e93b52e812276fb093b6ccf52a`.
This is not a compiler rebuilt from current sources. Current canonical workspace
has a separate local compiler artifact, which is not the compiler used here.

Allocation: 2 CPUs, 8192 MiB, GPU off, five-minute time limit.
Runtime UID/GID 1000/1000, clean-v1.
Evidence root:
`/var/lib/cephfs-slurm/training/sounio-foundry/runs/stdlib-nice-20261017-final1`.

Original isolated remote-main baseline:
`f5b2c5850be9f3d3995719ffcade44a6d851658a`.
Remote main fetched/rechecked:
`94a2280f92f2c056b64105bec2dbb802f897d601`.
Committed stdlib differences from that baseline are confined to
optimize/nelder_mead.sio and polynomial/lib.sio, outside this focused closure.
The active canonical checkout was not edited. No commit, push, PR, merge or
remote CI acceptance is claimed.

## Implemented

- `stdlib/viz/nice_axis.sio`: automatic covering domains with steps from
  1,2,2.5,5 times a decimal power; outward endpoint rounding, interval budget,
  finite-span/index guards and validation through the existing axis_ticks API.
- `stdlib/viz/plot.sio`: creates two automatic axes and applies the exact same
  limits to the source line viewport. Additional line series can share those
  limits on the same rectangle. Active graph capacity/geometry/domain validation
  precedes mutation; rejected operations preserve active counts, IDs and limits.
- No renderer fork, scene enlargement or per-point metadata was introduced.
  Existing Canvas/SVG axis/clip/legend code executes the resulting scene.
- `viz_auto_plot.sio`: actual case-deletion DFFITS drives the figure, with
  screen lines sharing the automatically planned domain. All points and both
  screening bounds must fit. Axis/series limit equality is asserted.
- `viz_nice_axis.sio`: independent numerical covering-domain comparisons and
  unavailable-domain/count/label cases.
- Packaging helper includes the new two fixtures and C++ reference, preserving
  the earlier 18 focused procedures and consumers.
- Public contract: `stdlib/viz/AUTO_AXES.md`; explicit AXES.md updated.

The step family is a custom covering policy. It does not reproduce D3 ticks.
Primary conceptual reference for the distinction between ticks inside a range
and a nice covering range: https://d3js.org/d3-array/ticks .

## Independent C++23 evidence

`tests/reference/nice_axis_reference.cpp` enumerates decimal candidates globally
using long-double powers, floor/ceil and integer lattice counts. It does not
repeat the Sounio decimal normalization loops.

```sh
g++ -std=c++23 -O2 -Wall -Wextra -pedantic nice_axis_reference.cpp -o nice-axis-reference
./nice-axis-reference viz_nice_axis.sio
```

Twelve generated cases include positive/negative/zero-crossing domains, 2.5 steps,
and scales around 1e-299 and 1e299. Numeric comparisons use characteristic-step
normalization at tolerance 2e-12, plus exact interval-count and covering checks.
Failure fixtures include invalid budgets, constant input, one-interval crossing,
overflowing span, colliding six-significant-digit labels and NaN/infinity.
The consumer additionally exercises capacity, label and target-geometry rejection.
This is bounded evidence, not certification of all finite f64 domains.

## Native scientific artifact

The exact native SVG is exported as `dffits-automatic-axes.svg`, 800x320.
Canvas executes the same IR; SVG and its Quick Look thumbnail were inspected.
The highest DFFITS and both screening limits are visible. Axis limits are
derived from the source series, not copied as manual numbers in the consumer.
The six-digit scientific label format is inherited from the explicit axes;
short general-number/offset formatting remains future work.
Quick Look PNG has white square padding and is not a Canvas PNG export.

Local custody:
`/Users/demetriosagourakis/Documents/Sounio-stdlib-sprint-2026-10-16/automatic-axes-2026-10-17/`.

## Limits and continuation

- Automatic linear axes only. Log/symlog and narrow-domain offset notation remain.
- Snapshot binding, not live dependency tracking. Rebuild/rebind after changing
  data or layout. Source bounds are authoritative; sharing can clip larger target
  series. Compute a union first when all series must influence the limits.
- Existing line-clip contract: one-pixel stroke and bounded plot geometry.
  The caller reserves margins and output-canvas dimensions.
- No heap allocation inside numeric planning or binding. The helper consumes
  two nodes and zero series; it never copies a 1024-point series to share bounds.
  No measured RSS/runtime/binary-size improvement is claimed in this batch.
- Geometry and label parity do not imply identical Canvas/SVG raster pixels.
- Native receipts cover 20 focused tests, not the entire stdlib corpus.

The five-track master sprint remains active. Next render work: compact SVG paths,
log/symlog and other-mark clipping. Statistics assumption/resampling procedures,
genomics count-matrix contracts, mathematics boundaries and physics covariance
remain implementation work, not declared complete.

## Controlled earlier-policy comparison

Run `stdlib-nice-20261017-before1`, Slurm **14880**, FAILED **1:0**, elapsed
**00:01:12**. JUnit: **18 passes, 2 failures, zero skips/errors**.
Snapshot SHA-256:
`9b84fe85eb0609f72c3990972c9517c77ff8a5148154026c08342bc989f9b9b4`.

The complete package manifests differ only in `stdlib/viz/nice_axis.sio`.
The counterfactual preserves the new API and wraps the previous explicit-axis
policy: raw endpoints and the requested fixed interval count. Both new fixtures
compile successfully and fail at runtime. The numeric fixture reaches NICE_CASE_1;
the automatic figure fixture also fails. All earlier 18 cases remain passing.

This is a controlled earlier-policy comparison with the new API retained, not an
unmodified historical checkout. It establishes sensitivity to non-nice raw
intervals and inadequate figure coverage; it is not a comparative performance
benchmark or proof of optimal tick selection.
