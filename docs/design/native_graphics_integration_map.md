<!-- docs:meta
topic_id: repo.docs.design.native-graphics-integration-map
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.design.native-graphics-integration-map
-->

# Native Graphics Integration Map

Status: active integration note
Scope: Grok architecture + Kimi scaffold + Codex quality companion
Workspace: `/workspace/sounio`

## Current Shape

The native graphics work is now one stack with three coordinated layers:

1. Grok owns the architecture intent in `docs/design/native_graphics_library.md`.
2. Kimi owns the working scaffold under `stdlib/graphics/*`.
3. Codex owns the publication-quality scientific companion in `stdlib/graphics/quality.sio`.

The stack should stay layered:

- `graphics::drawing`: primitive color/geometry vocabulary.
- `graphics::surface`: fixed-size RGBA pixel surface and low-level drawing.
- `graphics::plot`, `graphics::scatter`, `graphics::heatmap`: compact compatibility plots.
- `graphics::raster`: antialiasing helpers for the quality layer.
- `graphics::quality`: SOTA scientific raster plots.
- `graphics::epistemic`: Sounio-native uncertainty/provenance visualization.
- `graphics::svg`: vector export bridge.
- `graphics::png`: pure-Sounio PNG output using stored DEFLATE.
- `graphics::tile`, `graphics::tiled_plot`: tiled scaling path.
- `graphics::view`: terminal viewing for rapid visual iteration.

## Gate Contract

Use two gates, in this order:

1. `scripts/ci/graphics_scaffold_gate.sh`
   - Owner gate for the base graphics scaffold.
   - Checks all base graphics modules.
   - Also checks `graphics::quality` when the module exists.

2. `scripts/ci/graphics_companion_gate.sh`
   - Integrated acceptance gate.
   - Runs the scaffold gate first.
   - Then exercises companion, raster, quality, SVG, tile, PNG, and tiled-export smokes.

Do not replace the scaffold gate with the companion gate. The scaffold gate is
the stable owner gate; the companion gate is the broader integration gate.

If the companion gate fails while the scaffold gate still passes, treat that as
an integration regression rather than a scaffold ownership failure. Fix the
companion regression before release or PR merge, but keep the scaffold gate
available as the smaller isolation gate for base-module work.

## Current Quality Surface

`graphics::quality` currently provides:

- antialiased line plot
- smooth heatmap
- contour raster
- viridis colorbar
- uncertainty band
- markers
- error bars
- histogram
- ECDF
- boxplot
- violin density plot

These functions intentionally render precomputed statistical structures. Heavy
statistics such as KDE, quantile estimation, binning policy, and model-aware
uncertainty decomposition should be added as separate data-preparation helpers
only after the render contracts are stable.

## Next Integration Order

1. Keep the base scaffold green.
2. Keep quality smokes green.
3. Add one quality primitive at a time.
4. For each primitive, add one pixel-level smoke.
5. Add it to `graphics_companion_gate.sh`.
6. Only promote into `graphics_scaffold_gate.sh` when the module boundary is stable.
7. After multiple primitives are stable, add a gallery smoke instead of expanding every smoke into a broad visual suite.

## Do Not Mix

Do not mix these in one small increment:

- render primitive behavior
- PNG encoder changes
- tiling semantics
- epistemic semantics
- module governance docs

The only acceptable cross-layer edit is a narrow integration update like this
document, module index text, or a gate line that checks an already-green module.

## Current Known Constraint

The live implementation uses fixed-size arrays and by-value `Surface` mutation
patterns because current compiler/codegen behavior makes large mutable struct
references risky. Preserve this style until the compiler lane proves a safer
mutation contract.
