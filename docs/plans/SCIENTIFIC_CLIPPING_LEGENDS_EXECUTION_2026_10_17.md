<!-- docs:meta
topic_id: repo.docs.plans.scientific-clipping-legends-execution-2026-10-17
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.scientific-clipping-legends-execution-2026-10-17
-->

# Scientific line clipping and legends — execution 2026-10-17

## Sprint continuity and source

This implements the next bounded library checkpoint of the original major sprint:
statistics, scientific visualization, mathematics, genomics and physics.
It extends the existing Visual IR and Canvas/SVG renderers. It does not introduce
another plotting stack. The next active statistical target remains real regression
diagnostics (DFFITS and honest assumptions), followed by associated consumers.

Remote worktree: /workspace/sounio-sprint-stats-viz.
Original remote-main baseline: f5b2c5850be9f3d3995719ffcade44a6d851658a.
Remote main freshly fetched: 67e14779dd2b708767a064d15fc696dc5eebd734.
The diff from baseline to origin/main is empty for the targeted stdlib/viz,
stdlib/display/canvas_ext.sio, stats, math/pure, genomics and physics paths.
The live canonical checkout belongs to another agent and was not edited.
Changes remain uncommitted; no push, PR, CI or merge is claimed.

## Implemented

- viz::clip::{clip_unit_segment,clip_series_segment,clip_pixel}: shared bounded
  Liang-Barsky clipping, linear normalization, explicit statuses, exact binding
  of the active clipping face, and nearest-integer pixel quantization.
- viz::ir::viz_line_set_viewport: validated explicit domains, opt-in line clipping,
  transactional rejection, bounded geometry and a one-pixel-stroke contract.
- display::canvas_ext::{PixelScissor,canvas_draw_line_aa_scissored}: bounded
  integer rational antialiasing. Every blended pixel obeys the inclusive scissor.
- SVG uses the same accepted endpoint geometry and a clipPath over the closed
  pixel cells. Round line caps preserve a segment that degenerates to a point.
  The line-cap choice follows https://www.w3.org/TR/SVG2/painting.html#LineCaps .
- Both renderers omit segments adjacent to nonfinite samples, preserving gaps.
  SVG emits visible/outside/invalid segment counts.
- viz::legend::legend_layout: one swatch/text origin, vertical centering and ASCII
  cell-capacity calculation shared by Canvas and SVG.
- viz::ir::viz_add_series_legend: validated source-chart style snapshot, printable
  ASCII label, transactional rejection and no additional series slot.
- Legacy legend controls use the same renderer geometry. Text starts after the
  swatch, gets truncated by whole cells, and XML-special characters are escaped.

No scene/series structures grew. Legend text remains complete in IR even when
the visible text is shorter. No performance, RSS or artifact-size gain was measured.
SVG path compaction, alpha/wide-stroke rendering and live style binding remain work.

Full API/precision/missing-data/geometry/legend limits: stdlib/viz/CLIPPING.md.
The scalar and small-record projections used in new IR adapters were required to
pass the pinned native integration; no compiler root-cause fix is claimed.

## Independent numerical reference

tests/reference/scientific_clip_reference.cpp generates the Sounio expectations.
It uses a separate long-double Cohen-Sutherland implementation, rather than
sharing Liang-Barsky implementation steps. Twenty deterministic segment cases
include all sides, parallel/coincident boundaries, reversed direction, corner
contacts, wholly outside segments, points and normalized endpoints at +/-1e6.
Normalized coordinate comparisons use absolute tolerance 1e-9. Reference
coordinates exactly on 0/1 also require exact boundary equality.

C++23 integer arithmetic independently generates coverage-pixel expectations for
x-major, y-major and negative-slope red AA segments.
Additional Sounio checks cover NaN/infinity, normalized-domain rejection, raw
values around 1e300, invalid-domain rejection before mutation, strict zero-valued
margins outside the viewport, point rendering and nonfinite gaps.
Legend tests cover swatch/glyph pixels, source color, bounded text, full IR label
preservation, ASCII rejection, geometry rejection and SVG escaping.

Generate helpers:
c++ -std=c++23 -O2 tests/reference/scientific_clip_reference.cpp -o /tmp/clip-reference
/tmp/clip-reference > /tmp/clip-reference.sio

No Python, Rust or Julia numerical oracle was used.

## Consumer

The native program evaluates electron_compton_shift at 13 angles from 0 to pi.
It renders a sampled, piecewise-linear nominal Compton curve with explicit
viewport 0.5..2.5 rad and 1..4 pm, linear scientific axes, units and a style-linked
legend. The physical kernel was independently checked in the preceding
multi-domain batch; this checkpoint tests its visualization integration.
This is not experimental data, a continuous-curve error bound, or an uncertainty fit.

The same program separately exports a simple crossed viewport, a NaN gap and a
zero-length visible segment. The categorical forest consumer from the prior
checkpoint is retained in the focused suite.

## Controlled failing-before / passing-after

The before candidate is a separate runtime package. It retains the new input/domain
adapters and the exact same two acceptance fixtures, but leaves line clipping
dispatch disabled and restores the old Canvas/SVG legend renderer branches.
These are enabling prerequisites for running the new consumers against the old
render behavior; original main did not already expose the new public APIs.

| Run | Slurm | State | Focused JUnit |
|---|---|---|---|
| stdlib-clip-20261017-before2 | 14851 | FAILED 1:0, expected regression | 15 tests, 2 failures, 0 skips/errors |
| stdlib-clip-20261017-final2 | 14850 | COMPLETED 0:0, 54 seconds | 15 tests, 0 failures/skips/errors |

The failing consumers are viz_scientific_clip and viz_series_legend.
The previous thirteen focused cases remain passing. This does not certify the
whole stdlib, all chart kinds, full corpus, CI or a compiler fixed point.

Earlier accepted checkpoints in this batch: stdlib-clip-20261017-c (14845)
and stdlib-clip-20261017-final (14848), both 15/15.
The final2 checkpoint additionally covers the SVG zero-length cap and
closed-pixel-cell mask. The earlier controlled before run 14849 was 13/15.
Initial runs 14842/14844 caught integration guard failures; typed scalar and
small-record projections fixed those before native acceptance.

## Exact executed identity

Final snapshot SHA-256:
49544cdeb5d15ed15fc1669499c67e321d8133267688327ed37cb45cba837e7d

Committed pinned Madaros v0.80.0 SHA-256:
125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f

This is execution with the committed pinned compiler, not a compiler rebuild
from current sources. Slurm allocation: 2 CPUs, 8192 MiB, GPU off.
Runtime receipt: clean-v1 environment, actual UID/GID 1000.

The runtime-package MANIFEST.sha256 was regenerated before submission.
Core source copied for local custody is checked against that snapshot manifest.
Documentation was updated after execution and has separate custody hashes.

c68a7a92815806d9d4c50e0b4020419861300a503aeed4228730ae0a10938235  stdlib/viz/clip.sio
5466218a6922f921747c067b47c75129a242a69688a436fbdd78ce1ee462d553  stdlib/viz/legend.sio
5fdac8a6f45509cb2f8f66f904ae05494f4f27240b30b1034ccbcc8a5f3f9cd2  stdlib/viz/ir.sio
e828a045617c821f2bd8bdfb4b20aabb550a45b9e04d9ee2aada277cac838e25  stdlib/viz/viz_canvas.sio
8649625545b5c25bc33aa9c69342994bdf8d4fd3a97895c6eaa3f6ec12551a5e  stdlib/viz/viz_html.sio
54f93fbb01a13c3229d399457ce0781b6e7a8529e19fe45fc896e492d4d8035f  stdlib/display/canvas_ext.sio
b2dea92aef5120d88e5cdda77fa1d2040eb9a3703bba5cc83b6f212ee573d605  tests/reference/scientific_clip_reference.cpp
04c43d0b14b45701a4e5d5328521511d0c53b3da98d25a5867c42005522399fa  tests/run-pass/viz_scientific_clip.sio
d3318193cf5b47afaf36a5630fc2283334158475acd801ebdee81b49d58d91d3  tests/run-pass/viz_series_legend.sio

## Artifacts and continuation

Native run:
 /var/lib/cephfs-slurm/training/sounio-foundry/runs/stdlib-clip-20261017-final2

Local custody:
 /Users/demetriosagourakis/Documents/Sounio-stdlib-sprint-2026-10-16/scientific-clipping-2026-10-17

compton-scientific-viewport.svg is extracted from the actual final native stdout.
The separate clipped-point.svg preserves the degenerate-segment example.
SVG XML and a rendered physical-consumer preview are inspected during custody.
Quick Look PNGs are previews; SVG is the scientific figure artifact.

Next: real regression DFFITS/diagnostics and a scientific consumer, then automatic
ticks/log policies, automatic axis/domain binding and clipping for other marks.
Compact SVG paths remain an explicit artifact-size follow-up. Existing package
and compile-once bundle support remains available without starting a new
infrastructure/profiling sprint.
