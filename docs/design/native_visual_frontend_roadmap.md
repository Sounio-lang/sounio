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

- `stdlib/viz/workbench.sio` composes chart, epistemic band, heatmap, rendered molecule nodes, scalar/vector field, trajectory, spectrum, 3D mesh, controls, labels, legend, tooltip, viewport, replay, inspector, editor, Canvas, and HTML/SVG into one reusable scene blueprint.
- `examples/viz_workbench/main.sio` is the first native Workbench demo surface.
- `stdlib/viz/molecule_editor.sio` adds atom-level hit-testing, selection, position/radius/color mutation, and deterministic atom hashes over Sounio molecule payloads.
- `tests/run-pass/viz_workbench_roundtrip.sio` proves Workbench render -> replay -> inspector selection -> editor mutation -> Canvas highlight -> molecule atom authoring -> static HTML/SVG identity export without delegating scientific semantics to JavaScript.

## V1.7: Checked Molecule Authoring

- `stdlib/viz/authoring.sio` lifts molecule edits into typed transactions over Visual IR: select atom, select atom at Canvas position, nudge selected atom, nudge with angstrom/nanometer units, set radius/color, add atom, add bond, set bond order, and delete atom.
- `VizMoleculeConstraint` keeps authoring constraints in Sounio data: locked atoms, bond-length bounds, radius bounds, and fixed molecule capacity.
- Checked authoring returns explicit reason codes for unsupported actions, invalid atoms, unknown unit, bond geometry, locked atom, radius range, and capacity. Rejected actions preserve before/after audit and atom hashes.
- Authoring traces and certificates record before/after scene audit plus atom hashes for each frame. Verification is a Sounio function over the certificate data; no browser runtime owns molecule semantics.
- `viz_audit_hash` includes molecule atom and bond payloads, so atom moves, radius/color edits, bond changes, and capacity mutations are visible to replay/export audit surfaces.
- `tests/run-pass/viz_workbench_roundtrip.sio` proves typed action replay, checked constraints, certificate verification, Canvas pixels, and static HTML/SVG export from the same Workbench scene.

## V1.8: Native Molecule Studio

- `stdlib/viz/molecule_studio.sio` turns checked molecule authoring into a native Workbench tool layer. Tool state, unit scale, pending bond endpoint, last action, last reason, accept/reject counts, and timeline count are fields on `VizScene`.
- `stdlib/viz/workbench.sio` now binds a molecule node to Studio V0 and adds Visual IR toolbar controls for select, move, add atom, add bond, delete, lock, status, and timeline.
- Studio pointer and nudge commands reduce into `viz::authoring` checked actions, so constraints for locked atoms, geometry, units, radius, and capacity remain Sounio data.
- `viz_molecule_studio_handle_event` connects native `display::event::Event` input to the Studio toolbar and molecule canvas. Button clicks choose tools, molecule clicks author atoms/bonds/deletes/locks, and arrow keys nudge selected atoms through checked unit-aware authoring.
- `viz_workbench_handle_event` and `viz_workbench_app_handle_event` compose Studio events with generic Visual IR controls, giving native Workbench demos a single app/event bridge while constraints remain explicit Sounio data.
- `viz_audit_hash`, `viz_app_frame_hash`, and `viz_html_emit_scene` expose Studio state for replay/export parity without moving semantics into JavaScript.
- `tests/run-pass/viz_workbench_roundtrip.sio` proves the direct Studio path end to end, `tests/run-pass/viz_molecule_studio_events.sio` proves the native event path, and `tests/run-pass/viz_workbench_app_events.sio` proves the composed app bridge from event -> dirty app -> render/hash.

## V1.8b: Workbench Replay Proof With Studio Constraints

- `tests/run-pass/viz_workbench_replay_studio.sio` replays a fixed `display::event::Event` tape through the composed Workbench app bridge instead of the generic Visual IR reducer alone.
- The proof uses `viz_replay_present` plus the public `viz_replay_record` hook after each Workbench-specific reducer step, so composed frontends can share replay timelines without adding a separate replay module.
- The trace records Canvas hash, audit hash, scene time, event identity, and final Workbench molecule state while the tape crosses generic controls and Molecule Studio authoring constraints.

## V1.8c: Physchem Dynamics Nodes

- `VizScene` now stores `LatticeField2D` and `ParticleEventView` payloads in fixed-capacity Sounio-owned slots.
- `viz_add_lattice_field` and `viz_add_particle_event` create first-class Visual IR nodes for lattice/phonon motion and particle-event traces instead of leaving those physchem structs as standalone data.
- `viz_canvas_render_scene` renders displaced lattice nodes with phase/variance cues and particle-event traces with energy, kind, and variance cues.
- `viz_html_emit_scene` serializes both node types as passive static SVG groups plus audit comments; JavaScript still owns no scientific semantics.
- `tests/run-pass/viz_physchem_dynamics.sio` proves the new node kinds, slots, and Canvas pixels without requiring `DISPLAY`.

## V1.8d: Workbench Physchem Dynamics

- `viz::workbench` now includes lattice/phonon and particle-event nodes in the composed native scene.
- Workbench slider events synchronize existing physchem slots with `scene.time`, so dynamic state changes in-place under stable Visual IR node identities.
- `viz_audit_hash` includes lattice/phonon and particle-event renderer data, making physchem dynamics visible to replay traces and export/audit parity.
- `tests/run-pass/viz_workbench_physchem_dynamics.sio` proves the composed app bridge updates lattice phase/displacement, particle energy/position, audit hash, frame hash, and Canvas pixels.

## V1.8e: Workbench Physchem Export Parity

- `viz_workbench_sync_dynamics` keeps Workbench lattice/phonon and particle-event payloads synchronized with the native time slider as Sounio data.
- `viz_html_emit_scene` includes quantized physchem samples in passive comments for lattice/phonon and particle-event nodes: scene time, first phase/displacement/variance sample, and first event position/energy/variance sample.
- `tests/run-pass/viz_workbench_html_physchem.sio` drives the Workbench scrubber, verifies the updated payload values before export, and emits static HTML/SVG whose grepable audit comments prove the same dynamic scene reached the HTML renderer.

## V1.8f: Time-Driven 3D Mesh

- `VizMeshData` keeps the existing `Tri3D` array plus primitive vertex/normal/color mirrors, so Canvas, audit, and HTML/SVG renderers can read mesh geometry across module boundaries without relying on fragile large array-of-struct copies.
- `viz_workbench_sync_dynamics` updates the Workbench mesh from `scene.time`, giving the native scrubber a CPU Canvas 3D target whose vertices, normal, depth, and color are Sounio-owned data.
- `viz_canvas_render_scene` rebuilds `Tri3D` draw calls from the primitive mirror, and `viz_html_emit_scene` emits SVG polygons plus passive mesh audit comments with quantized vertex/normal/color samples.
- `tests/run-pass/viz_workbench_mesh_dynamics.sio` proves slider -> mesh payload -> audit hash -> Canvas frame hash -> static HTML/SVG export for the Workbench 3D scene.

## V1.9: Static HTML/SVG

- `stdlib/viz/viz_html.sio` serializes the same Visual IR as static markup.
- JavaScript is not used for scientific semantics.
- Exported markup can be viewed externally, but the model remains Sounio data.

## V2: GPU Renderer

Deferred until general `bin/souc --backend gpu` wiring exists for ordinary programs.

GPU work should add a renderer over Visual IR, not move molecule, field, unit, uncertainty, or simulation semantics into shader-side ad hoc state.
