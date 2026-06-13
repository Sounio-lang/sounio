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
- Molecule rendering and audit now use a fixed-capacity primitive atom/bond mirror on `VizScene`, preserving the `MoleculeViz` semantic payload while avoiding fragile renderer reads of large nested arrays across modules.
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

## V1.8g: Native Timeline Playback

- `viz::workbench` adds play/pause, step-back, step-forward, and frame-readout controls as Visual IR nodes with stable tags.
- `viz_workbench_tick` advances `scene.time` only while the play control is active, then synchronizes lattice/phonon, particle-event, trajectory cursor, and 3D mesh state through the same Workbench dynamics path.
- `viz_audit_hash` includes control node values and selection state, so playback state is visible to replay/certificate/export proofs.
- `tests/run-pass/viz_workbench_timeline_playback.sio` proves native control clicks plus deterministic ticks drive the full composed Workbench scene and static HTML/SVG export without moving simulation semantics into JavaScript.

## V1.8h: Workbench Replay Sessions

- `VizWorkbenchReplaySession` records bounded Workbench tapes containing native event steps and deterministic tick steps.
- `viz_workbench_replay_record_session` stores per-frame Canvas hashes, audit hashes, and quantized scene time after rendering through the native Canvas path.
- `viz_workbench_replay_verify_session` applies the recorded tape to a freshly rebuilt Workbench scene and verifies every frame against the recorded hashes and time samples.
- `viz_workbench_replay_dump_session` emits a compact text export of the session timeline for CI logs and external audit, while state and verification remain Sounio-owned data.
- `tests/run-pass/viz_workbench_replay_session.sio` proves record -> dump -> rebuild -> verify across playback controls, molecule authoring, frame hashes, audit hashes, readout state, and 3D mesh state.

## V1.8i: Replay Session HTML Archive

- `viz_workbench_replay_session_hash` derives a deterministic signature from the full Workbench replay tape, frame hashes, audit hashes, and time samples.
- `viz_workbench_replay_emit_html_archive` emits a passive `workbench-replay-session` comment for static HTML/SVG exports, including step count, frame count, session signature, final frame hash, final audit hash, and final time.
- `tests/run-pass/viz_workbench_replay_html_archive.sio` proves record -> verify -> archive comment -> static scene export parity: the archive final time, `viz-audit` scene time, timeline readout, and 3D mesh audit comments all agree on the same Sounio-owned replay state.

## V1.8i.1: Experiment Packet Domain Stamps

- `VizWorkbenchExperimentExportPacket` now carries compact domain stamps for scene nodes, chart/heatmap payloads, physchem field payloads, molecule payloads, mesh payloads, and the combined export payload.
- `viz_workbench_experiment_export_packet_emit` serializes those stamps in the passive packet comment, so a static export can be reviewed by domain before a fresh Workbench import/replay verifies zero Canvas-frame and scene-audit drift.
- `tests/run-pass/viz_workbench_experiment_diff_player.sio` proves the domain stamps are populated, packet-hashed, emitted, imported, and replayed without moving experiment semantics into JavaScript.

## V1.8i.2: Experiment Packet Ledger

- `VizWorkbenchExperimentPacketLedger` sequences shareable experiment packets into fixed-capacity Sounio-owned evidence with packet, payload-stamp, frame, and scene-audit hashes per entry.
- `viz_workbench_experiment_packet_ledger_add` records deltas between consecutive packets, while `viz_workbench_experiment_packet_ledger_verify` checks count bounds, nonzero evidence, last-entry mirrors, per-domain deltas, and the deterministic ledger hash.
- `viz_workbench_experiment_packet_ledger_emit` adds passive ledger metadata to static HTML/SVG exports, and `tests/run-pass/viz_workbench_experiment_diff_player.sio` proves a two-packet playback sequence with nonzero packet-hash delta.

## V1.8j: Canonical Headless Visual Gate

- `scripts/ci/native_visual_frontend_gate.sh` is the one-command proof surface for the native visual frontend lane.
- `scripts/ci/native_visual_frontend_gate.manifest.tsv` lists the check/run/compile/script entries and required run markers, while `scripts/ci/check_native_visual_frontend_gate_manifest.py` validates the manifest and executable script entries before execution.
- `scripts/ci/check_native_visual_frontend_plan_coverage.py` binds the original native visual frontend plan to executable evidence: Visual IR node kinds, fixed layouts, renderers, controls, physchem payloads, demos, gate labels, and GPU deferral docs.
- The gate checks `viz::sci` and Workbench modules, runs Canvas, HTML/SVG, passive no-JavaScript export, physchem, interaction, replay, Workbench, timeline, mesh, replay-session, and replay-archive proofs, then compile-gates the native window lab plus other visual demos and headless-runs the demos that do not require `DISPLAY`.
- `tests/run-pass/viz_module_surface_gate.sio` is a module import smoke used by the gate so the public `stdlib/viz` surface is checked through a real executable entrypoint rather than relying on module files that intentionally have no `main`.

## V1.8k: Scene Snapshot Authoring State

- `stdlib/viz/snapshot.sio` captures the edit-facing Visual IR state into a bounded Sounio struct: nodes, layout/style/labels/control values, identity, selection/focus, molecule-studio state, counts, and audit hashes.
- `viz_scene_restore_snapshot` restores the captured native frontend state onto an existing scene while scientific payload slots remain Sounio-owned `VizScene` data.
- `tests/run-pass/viz_scene_snapshot.sio` proves Workbench snapshot -> editor mutation -> audit change -> restore -> matching audit/node hashes -> Canvas and static HTML/SVG render from the restored state.

## V1.8l: Undo/Redo Scene History

- `VizSceneHistory` stores a compact bounded timeline for the current editor target node: label, rectangle, tag, value, selection/focus state, audit hash, node hash, cursor, and deterministic history hash.
- `viz_scene_history_undo` and `viz_scene_history_redo` restore authored Visual IR node state from Sounio history data. Recording a new edit after undo truncates the redo branch.
- `tests/run-pass/viz_scene_history.sio` proves Workbench snapshot history -> two editor mutations -> undo -> undo -> redo -> branch replacement -> Canvas and static HTML/SVG render from the authored state.

## V1.8m: Scene Package Export/Import

- `VizScenePackage` wraps a full `VizSceneSnapshot`, compact history metadata, and deterministic package hash into a Sounio-owned package.
- `viz_scene_package_emit` exports a passive textual audit comment for package metadata. The comment is display-only; package verification and restore remain Sounio functions.
- `tests/run-pass/viz_scene_package.sio` proves authored Workbench scene -> package capture -> textual package audit -> restore into rebuilt scene -> audit/node hash parity -> Canvas and static HTML/SVG export.

## V1.9: Static HTML/SVG

- `stdlib/viz/viz_html.sio` serializes the same Visual IR as static markup.
- JavaScript is not used for scientific semantics.
- Exported markup can be viewed externally, but the model remains Sounio data.

## V2: GPU Renderer

Deferred until general `bin/souc --backend gpu` wiring exists for ordinary programs.

GPU work should add a renderer over Visual IR, not move molecule, field, unit, uncertainty, or simulation semantics into shader-side ad hoc state.
