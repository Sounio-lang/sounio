<!-- docs:meta
topic_id: repo.docs.design.native-visual-frontend-roadmap
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.design.native-visual-frontend-roadmap
-->

# Native Visual Frontend Roadmap

Sounio visualization is moving from chart helpers to a native visual frontend.

## V1: CPU Canvas

- Visual IR in `stdlib/viz/ir.sio`.
- Native renderer in `stdlib/viz/viz_canvas.sio`.
- Scientific layers in `stdlib/viz/physchem.sio` and existing `viz::sci`.
- Headless CI proof through `tests/run-pass/viz_headless.sio`.
- No display server required.

## V1.1: Identity, Focus, And 3D Polish

- `VizNode` carries stable scene-local `id`, optional `parent_id`, and app-owned `tag` fields.
- `VizScene` carries `focused_node` plus deterministic focus next/previous/activate reducers for native controls.
- `viz_scene_dump` provides a simple text audit surface for node identity, hierarchy, tags, data slots, and fixed rectangles.
- `render::renderer3d` keeps CPU Canvas as the target while adding Blinn-Phong specular highlights and antialiased edge outlines.
- Proofs: `tests/run-pass/viz_identity_focus_dump.sio` and `tests/run-pass/viz_renderer3d_edges.sio`.

## V1.2: Live Lab Interaction And Export Identity

- `examples/viz_lab/main.sio` exercises pointer and keyboard interaction over a richer Visual IR scene: chart, uncertainty band, heatmap, molecule, scalar field, mesh, slider, tabs, and toggle.
- `tests/run-pass/viz_lab_interaction.sio` provides a headless lab proof with mouse events, keyboard focus traversal, focused activation, Canvas pixel checks, `viz_scene_dump`, and `viz_app_frame_hash`.
- `viz_html_emit_scene` exports every node inside a passive SVG group carrying `data-viz-id`, `data-viz-parent-id`, `data-viz-tag`, `data-viz-kind`, and `data-viz-slot`.
- `tests/run-pass/viz_html_identity.sio` proves those identity attributes in static HTML/SVG export.
- `viz_app_render` remains covered for small scenes by `tests/run-pass/viz_app_frame.sio`; richer lab scenes currently use explicit `viz_app_handle_event` plus `viz_canvas_render_scene` to match the demo path.

## V1.3: Replay, Inspector, And Cross-Renderer Audit

- `stdlib/viz/viz_replay.sio` replays fixed `display::event::Event` arrays into `VizApp`/`VizScene`, renders every Canvas frame, and returns the final `viz_app_frame_hash`.
- `tests/run-pass/viz_replay_deterministic.sio` proves two independent scenes and canvases converge to the same frame hash and interaction state from the same event tape.
- `stdlib/viz/inspector.sio` draws a native Canvas inspector panel for node count, selected/hovered/focused state, IDs, tags, kinds, and data slots.
- `tests/run-pass/viz_inspector_panel.sio` gives the inspector a headless pixel proof.
- `viz_html_emit_scene` now emits a passive `viz-audit` comment with node count, selected/hovered/focused nodes, active tab, and integer scene time so HTML/SVG exports carry the same audit trail without JavaScript semantics.

## V1.4: Interactive Inspector And Replay Timeline

- `stdlib/viz/audit.sio` defines shared Visual IR audit hashes for replay traces, inspector checks, and export/runtime parity tests.
- `viz_replay_trace_events` records a bounded timeline of replay frames: event kind/key/position, changed flag, Canvas hash, audit hash, selected/hovered/focused, active tab, and scene time.
- `tests/run-pass/viz_replay_trace.sio` proves timeline recording over a slider/toggle interaction tape.
- `viz_inspector_hit_node` and `viz_inspector_select` let native Canvas inspector rows select/focus/hover the corresponding Visual IR node.
- `viz_inspector_draw_node_highlights` overlays selected/hovered/focused node highlights on the rendered scene.
- `tests/run-pass/viz_inspector_interactive.sio` proves inspector row selection, shared audit hashes, and highlight pixels without a display server.

## V1.5: Minimal Scene Editor

- `stdlib/viz/editor.sio` edits the selected/focused/hovered Visual IR node without moving state into UI runtime code.
- The v1.5 editor supports fixed-rectangle nudging, fixed-rectangle resizing, app tag edits, clamped control value edits, and deterministic edit traces from keyboard events.
- `tests/run-pass/viz_editor_scene.sio` proves inspector selection -> editor mutation -> edit trace -> Canvas highlight -> static HTML/SVG export over the same edited Visual IR.

## V1.6: Native Visual Workbench

- `stdlib/viz/workbench.sio` composes chart, epistemic band, heatmap, molecule data, scalar/vector field, trajectory, spectrum, 3D mesh, controls, labels, legend, tooltip, viewport, replay, inspector, editor, Canvas, and HTML/SVG into one reusable scene blueprint.
- `examples/viz_workbench/main.sio` is the first native Workbench demo surface.
- `tests/run-pass/viz_workbench_roundtrip.sio` proves Workbench render -> replay -> inspector selection -> editor mutation -> Canvas highlight -> static HTML/SVG identity export without delegating scientific semantics to JavaScript.

## V1.7: Static HTML/SVG

- `stdlib/viz/viz_html.sio` serializes the same Visual IR as static markup.
- JavaScript is not used for scientific semantics.
- Exported markup can be viewed externally, but the model remains Sounio data.

## V2: GPU Renderer

Deferred until general `bin/souc --backend gpu` wiring exists for ordinary programs.

GPU work should add a renderer over Visual IR, not move molecule, field, unit, uncertainty, or simulation semantics into shader-side ad hoc state.
