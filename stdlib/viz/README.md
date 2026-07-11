# Sounio Viz

`stdlib/viz` is the first native visual frontend layer for Sounio.

## Stdlib harness

- `tests/stdlib/viz/test_viz_core.sio` — coord mapping + `viz_scene_new` (check-only)
- `tests/run-pass/viz_*` — Visual IR layout, replay, physchem, workbench smokes

The v1 architecture is:

- `viz::ir`: fixed-capacity Visual IR owned by Sounio.
- `viz::viz_canvas`: immediate native renderer to `display::Canvas`.
- `viz::viz_html`: static HTML/SVG export renderer over the same IR.
- `viz::viz_app`: headless-testable app runner for Event -> Visual IR -> Canvas frames.
- `viz::viz_replay`: deterministic event replay over `VizApp`, `VizScene`, Canvas rendering, and frame hashes.
- `viz::inspector`: native Canvas inspector panel for Visual IR identity and interaction state.
- `viz::viz_window`: optional native `display::Window` bridge for manual event-loop demos.
- `viz::audit`: shared state hashes used by replay, inspector, HTML/export audits, and headless tests.
- `viz::physchem`: Sounio data structures for molecules, bonds, scalar/vector fields, trajectories, spectra, lattice/phonon fields, particle event views, and uncertainty overlays.
- `viz::{coord,chart,epiviz,sci}`: direct drawing helpers that the IR renderer can lower into.

Scientific meaning stays in Sounio. Units, epistemic variance, molecules, lattices, fields, simulation clocks, scene nodes, and interaction state are represented as Sounio data. Browser or terminal export surfaces display pixels or markup; they do not own the scientific model.

## V1 Constraints

- Fixed node/data arrays; no heap-heavy scene graph.
- Nodes receive stable scene-local `id` values, optional `parent_id`, and an integer `tag` for application-owned identity.
- Fixed rectangles plus a simple row/column layout pass over contiguous node ranges; no flexbox.
- CPU Canvas is the primary renderer.
- HTML/SVG is static export only. It serializes Visual IR geometry and simple glyphs; no JavaScript owns model state.
- Scalar fields keep their Sounio `ScalarField2D` payload and mirror values into heatmap storage for renderer-friendly export.
- Vector fields keep their Sounio `VectorField2D` payload and mirror components into renderer-friendly vector slots with explicit row/column metadata.
- Trajectories keep their Sounio `Trajectory3D` payload and mirror path coordinates into renderer-friendly trajectory slots.
- Spectra keep their Sounio `SpectrumViz` payload and mirror wavelength/intensity/variance into renderer-friendly spectrum slots.
- Chart builders cover line, scatter, bar, heatmap, uncertainty band, forest plot, and waterfall nodes over the shared Visual IR.
- Scene time, selected node, hovered node, active tabs, and toggle values are Sounio state that renderers display.
- Focused node state is Sounio data. `viz_scene_focus_next`, `viz_scene_focus_prev`, and `viz_scene_activate_focused` provide deterministic keyboard-style navigation for controls.
- `viz_scene_dump` emits deterministic node/data counts plus per-node identity, parent, tag, kind, data slot, and rectangle fields for debugging and CI proof logs.
- `viz_html_emit_scene` wraps each Visual IR node in a static SVG group with `data-viz-id`, `data-viz-parent-id`, `data-viz-tag`, `data-viz-kind`, and `data-viz-slot` attributes so exports remain auditable without JavaScript semantics.
- `viz_html_emit_scene` also emits a passive `viz-audit` comment with node count, selected/hovered/focused nodes, active tab, and integer scene time.
- `viz_app_frame_hash` gives headless tests a compact deterministic probe over app frame state, Visual IR interaction state, and a few Canvas pixels.
- `viz_replay_events` replays a fixed event array into the Sounio reducer, renders every frame, and returns the final deterministic frame hash.
- `viz_replay_trace_events` records one frame summary per event: input event, changed flag, Canvas hash, audit hash, selected/hovered/focused, active tab, and integer scene time.
- `viz_inspector_draw` renders node IDs, tags, kinds, selected/hovered/focused state, and node counts into a native Canvas panel.
- `viz_inspector_hit_node` and `viz_inspector_select` make inspector rows interactive Sounio state: selecting a row updates selected/hovered/focused node fields without delegating state to UI runtime code.
- `viz_inspector_draw_node_highlights` overlays native Canvas highlights on selected/hovered/focused Visual IR nodes.
- `viz::editor` edits the selected/focused/hovered node: nudge/resize fixed rectangles, set app tags, set clamped control values, and record edit traces from keyboard events.
- `viz::workbench` composes charts, molecule data, scalar/vector fields, trajectory, spectrum, mesh, controls, inspector/edit/replay/export hooks into a reusable native visual workbench scene blueprint.
- Labels and text controls use fixed `[i8; 64]` buffers in the Visual IR. Canvas renders them with `display::font`; HTML/SVG escapes XML-sensitive ASCII before export.
- Native frontend builders cover button, slider, toggle, tabs, legend, tooltip, text viewport, and plot viewport nodes over the same control reducer machinery.
- `display::event::Event` can be reduced directly into `VizScene` for headless tests and future native window loops.
- Native windows are an optional layer over the same app runner; CI proofs stay headless.
- `render::renderer3d` remains a CPU Canvas renderer. Its v1.1 path adds Blinn-Phong specular shading and antialiased triangle edge lines while keeping 3D scene meaning in Sounio data.
- GPU rendering is deferred until general compiler GPU backend wiring is ready.
