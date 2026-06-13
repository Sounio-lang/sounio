# Sounio Viz

`stdlib/viz` is the first native visual frontend layer for Sounio.

The v1 architecture is:

- `viz::ir`: fixed-capacity Visual IR owned by Sounio.
- `viz::viz_canvas`: immediate native renderer to `display::Canvas`.
- `viz::viz_html`: static HTML/SVG export renderer over the same IR.
- `viz::viz_app`: headless-testable app runner for Event -> Visual IR -> Canvas frames.
- `viz::viz_replay`: deterministic event replay over `VizApp`, `VizScene`, Canvas rendering, and frame hashes.
- `viz::inspector`: native Canvas inspector panel for Visual IR identity and interaction state.
- `viz::viz_window`: optional native `display::Window` bridge for manual event-loop demos.
- `viz::audit`: shared state hashes used by replay, inspector, HTML/export audits, and headless tests.
- `viz::molecule_editor`: atom-level molecule hit-testing, selection, and mutation helpers over `VizScene`.
- `viz::authoring`: typed molecule authoring transactions, checked constraints, replay traces, and proof-hash certificates over `VizScene`.
- `viz::molecule_studio`: native molecule-authoring tool state over the Workbench, backed by checked `viz::authoring` transactions and Visual IR controls.
- `viz::snapshot`: bounded Visual IR scene snapshots, compact undo/redo history, and auditable scene packages for native authoring, restore, and future persistence surfaces.
- `viz::physchem`: Sounio data structures for molecules, bonds, scalar/vector fields, trajectories, spectra, lattice/phonon fields, particle event views, and uncertainty overlays.
- `viz::{coord,chart,epiviz,sci}`: direct drawing helpers that the IR renderer can lower into.

Scientific meaning stays in Sounio. Units, epistemic variance, molecules, lattices, fields, simulation clocks, scene nodes, and interaction state are represented as Sounio data. Browser or terminal export surfaces display pixels or markup; they do not own the scientific model.

The canonical headless acceptance gate for this lane is:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/native_visual_frontend_gate.sh
```

It checks the module surface, Canvas renderer, static HTML/SVG export, physico-chemistry nodes, Workbench interaction, replay sessions, timeline playback, and demo compile/run surfaces without requiring `DISPLAY`.

The gate reads `scripts/ci/native_visual_frontend_gate.manifest.tsv`, validates it with `scripts/ci/check_native_visual_frontend_gate_manifest.py`, and checks original-plan coverage with `scripts/ci/check_native_visual_frontend_plan_coverage.py`, so the accepted proof surface is explicit, reviewable, and tied to the native frontend plan rather than only to the easiest passing demos.

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
- Lattice/phonon fields keep their Sounio `LatticeField2D` payload and render as displaced lattice nodes with phase and variance visible in Canvas and static SVG.
- Particle event views keep their Sounio `ParticleEventView` payload and render as bounded event traces with energy, event kind, and variance visible in Canvas and static SVG.
- Chart builders cover line, scatter, bar, heatmap, uncertainty band, forest plot, and waterfall nodes over the shared Visual IR.
- Scene time, selected node, hovered node, active tabs, and toggle values are Sounio state that renderers display.
- Focused node state is Sounio data. `viz_scene_focus_next`, `viz_scene_focus_prev`, and `viz_scene_activate_focused` provide deterministic keyboard-style navigation for controls.
- `viz_scene_dump` emits deterministic node/data counts plus per-node identity, parent, tag, kind, data slot, and rectangle fields for debugging and CI proof logs.
- `viz_html_emit_scene` wraps each Visual IR node in a static SVG group with `data-viz-id`, `data-viz-parent-id`, `data-viz-tag`, `data-viz-kind`, and `data-viz-slot` attributes so exports remain auditable without JavaScript semantics.
- `viz_html_emit_scene` also emits a passive `viz-audit` comment with node count, selected/hovered/focused nodes, selected molecule atom state, active tab, and integer scene time. Lattice/phonon and particle-event comments include quantized time and payload samples so static exports can be checked against the same Sounio-owned dynamic state without JavaScript semantics.
- `viz_audit_hash` includes molecule atom/bond payloads plus lattice/phonon and particle-event renderer data, so molecule edits and physchem dynamics are visible to replay, certificate, and export audit surfaces.
- `viz_app_frame_hash` gives headless tests a compact deterministic probe over app frame state, Visual IR interaction state, and a few Canvas pixels.
- `viz_replay_events` replays a fixed event array into the Sounio reducer, renders every frame, and returns the final deterministic frame hash.
- `viz_replay_trace_events` records one frame summary per event: input event, changed flag, Canvas hash, audit hash, selected/hovered/focused, active tab, and integer scene time.
- `viz_replay_record` is public so composed reducers can reuse the same replay timeline after applying domain-specific events, such as Workbench Molecule Studio constraints.
- `viz_inspector_draw` renders node IDs, tags, kinds, selected/hovered/focused state, and node counts into a native Canvas panel.
- `viz_inspector_hit_node` and `viz_inspector_select` make inspector rows interactive Sounio state: selecting a row updates selected/hovered/focused node fields without delegating state to UI runtime code.
- `viz_inspector_draw_node_highlights` overlays native Canvas highlights on selected/hovered/focused Visual IR nodes.
- `viz::editor` edits the selected/focused/hovered node: nudge/resize fixed rectangles, set app tags, set clamped control values, and record edit traces from keyboard events.
- `viz::snapshot` captures edit-facing Visual IR state as Sounio data: node identity/hierarchy/layout/style/labels/control values, selection/focus, molecule-studio state, counts, and expected audit hashes. `viz_scene_restore_snapshot` restores that state without giving renderer or browser code ownership of the scene model.
- `VizSceneHistory` keeps a compact bounded history of the current editor target node with deterministic hash, undo, redo, and redo truncation after a new edit, so Workbench authoring can grow toward persistent scene packages without moving state into browser runtime code.
- `VizScenePackage` combines a full scene snapshot with compact history metadata and a deterministic package hash. `viz_scene_package_emit` prints a passive audit comment, while `viz_scene_package_restore` restores the authored Visual IR state into a rebuilt Sounio scene.
- `VizSceneProject` is the Visual Project Format v0 envelope: a scene package plus replay signature, final frame/audit/time hashes, export hash, deterministic project hash, restore/verify helpers, and a passive `viz-scene-project` export comment.
- `viz::workbench` exposes native package controls as Visual IR nodes: undo, redo, save, restore, and package audit. These route through `VizSceneHistory` and `VizScenePackage`, so persisted authoring state remains Sounio data rather than renderer or browser state.
- `VizWorkbenchProjectStore` connects native Workbench controls to Visual Project Format v0: save/load/export buttons update project state, dirty status, project/replay/export hashes, and restore the scene without moving semantics into the renderer.
- `VizWorkbenchProjectWorkspace` keeps two complete Visual Project Format slots for A/B scientific scene work. Native `saveA`, `saveB`, `loadA`, `loadB`, `diff`, `ab`, `dt`, `dh`, `df`, `dp`, and `dn` viewport controls are Visual IR nodes handled by `viz_workbench_workspace_app_handle_event`; compare produces `VizWorkbenchProjectDiff` with time, audit, frame, project, and node-count deltas while the project envelopes remain Sounio data.
- `viz::molecule_editor` edits molecule payloads through the Visual IR: count atoms, project atom positions to Canvas coordinates, hit-test atoms, select molecule nodes, nudge atoms, set atom radius, and recolor atoms.
- `viz::authoring` lifts molecule edits into typed transactions: select atom, select by hit-test, nudge with angstrom/nanometer units, set radius/color, add/delete atoms, add bonds, set bond order, and replay bounded action tapes.
- Checked molecule authoring returns explicit reason codes for unsupported actions, invalid atoms, unknown units, bond geometry, locked atoms, radius bounds, and capacity. Rejected checked actions leave audit and atom hashes unchanged.
- Authoring certificates are Sounio data derived from before/after audit and atom hashes. `viz_authoring_verify_checked_trace` verifies every frame without delegating scientific meaning to JavaScript or Python.
- `viz::molecule_studio` adds Workbench-facing native tools for select, move, add atom, add bond, delete, and lock. Pointer gestures and nudge commands reduce into checked authoring actions, then update Sounio-owned last-action, last-reason, accepted/rejected, pending-bond, unit, and timeline state.
- `viz_molecule_studio_handle_event` reduces native mouse/key events into that same Studio state: toolbar button hits switch tools, molecule hits apply checked actions, and arrow keys nudge the selected atom while the move tool is active.
- `viz::workbench` provides the composed app bridge: `viz_workbench_handle_event` tries Molecule Studio first, then falls back to generic Visual IR controls; `viz_workbench_app_handle_event` marks the native app dirty for either path.
- The Workbench molecule toolbar is Visual IR too: tool buttons, status tooltip, and timeline viewport are nodes with stable tags and parent identity. Canvas and static HTML/SVG render the same controls without JavaScript semantics.
- `viz_html_emit_scene`, `viz_audit_hash`, and `viz_app_frame_hash` include molecule studio state so exported scenes, replay traces, and headless frame hashes observe the active tool, unit, reason, action, timeline, and checked action counts.
- `tests/run-pass/viz_workbench_replay_studio.sio` proves replay over the composed Workbench app bridge, including Molecule Studio constraints, Canvas hashes, scene time, molecule tool, timeline, accepted/rejected counters, and atom count per frame.
- `viz_workbench_replay_record_session`, `viz_workbench_replay_verify_session`, `viz_workbench_replay_dump_session`, and `viz_workbench_replay_emit_html_archive` lift Workbench replay into a Sounio-owned session tape with event steps, tick steps, per-frame Canvas hashes, audit hashes, integer scene time, a deterministic session signature, and a passive HTML/SVG archive comment. `VizWorkbenchSessionTimeline` and `viz_workbench_replay_to_frame` turn that tape into a navigable frame timeline with selected frame/time/hash/delta state and can rebuild a selected frame in a fresh scene without JavaScript or Python semantics. `VizWorkbenchSessionArchive` then binds session hash, visual project hash, selected timeline frame, and optional A/B diff hash into one auditable lab-notebook-style envelope. `VizWorkbenchNotebook` indexes multiple archives as a fixed-capacity visual lab notebook with selected/baseline/compare slots, run-to-run deltas, a notebook hash, verification, and passive HTML/SVG export. `VizWorkbenchNotebookBrowser` selects an archive from that notebook, replays its session into a fresh Workbench scene, and records replay/browser hashes for native navigation. `VizWorkbenchNotebookCompareBrowser` replays baseline and comparison archives side by side, then binds both browser hashes to run-to-run frame/audit deltas for native comparison. `VizWorkbenchExperimentDiffPlayer` scrubs baseline and comparison sessions at arbitrary frames and records timeline, frame, audit, time, and hash-distance state as Sounio-owned data. `VizWorkbenchExperimentDiffOverlay` applies that diff-player state back onto native Visual IR controls, changing Canvas pixels and passive HTML/SVG metadata without JavaScript semantics. `VizWorkbenchExperimentDiffArtifact` packages notebook/player/overlay hashes with scene audit and rendered frame hashes as a portable experiment comparison artifact. `VizWorkbenchExperimentDiffLibrary` indexes multiple artifacts as a fixed-capacity native experiment library with selected/baseline/compare slots and artifact/render/audit deltas. `VizWorkbenchExperimentDiffLibraryBrowser` projects that library selection back into native Visual IR controls so Canvas and HTML/SVG exports show which artifact, baseline, compare, and deltas are active. `VizWorkbenchExperimentCompareMode` handles native key/mouse events to advance artifact selection, reset baseline, update comparison, reapply the browser, and keep event counters plus frame hashes in Sounio data. `VizWorkbenchExperimentPackage` binds the notebook, artifact library, browser, compare mode, rendered frame hash, and scene audit hash into one reproducible Workbench experiment envelope. `VizWorkbenchExperimentPackageRestore` restores that envelope into a fresh Workbench scene by matching artifacts, rebuilding library/browser/mode state, rendering Canvas, and verifying restore/frame/audit hashes. `VizWorkbenchExperimentPackageStore` keeps multiple experiment envelopes in fixed slots with active/baseline/compare selection, save/select counters, and package/render/audit deltas. `viz_workbench_handle_experiment_package_store_event` makes those slots native Visual IR controls for save, active-slot cycling, baseline/compare assignment, active restore, and package/frame/audit delta markers. `VizWorkbenchExperimentTimelineCockpit` binds the active package store, notebook archives, selected/baseline/compare frames, timeline hashes, frame hashes, audit hashes, and package/render/audit deltas into one native cockpit state. `VizWorkbenchExperimentCockpitPlayback` lets native time buttons step active, baseline, and compare experiment timelines together while preserving verified frame/time/hash state. `VizWorkbenchExperimentExportPacket` binds package, restore proof, store, cockpit, playback, export frame, scene audit hashes, and fixed-capacity Visual IR scene snapshot buffers into one passive shareable lab packet. `VizWorkbenchExperimentImportReplayReport` imports that packet into a fresh Workbench scene, rebuilds package restore, cockpit, and playback state, restores packet snapshot buffers, renders Canvas, and records frame/audit drift against the exported packet.
- `viz::workbench` composes charts, rendered molecule nodes, scalar/vector fields, trajectory, spectrum, lattice/phonon dynamics, particle-event traces, time-driven mesh, timeline playback controls, inspector/edit/replay/export hooks into a reusable native visual workbench scene blueprint.
- Workbench time controls synchronize lattice/phonon, particle-event, and 3D mesh slots in-place, so interaction changes Sounio-owned physchem/geometry state rather than recreating a renderer-only scene. Play/pause, step-back, step-forward, and frame readout are Visual IR controls and replay/audit-visible Sounio data.
- Mesh slots keep their `Tri3D` payload and a primitive-array mirror for Canvas, audit, and HTML/SVG renderers. The mirror avoids fragile cross-module array-of-struct reads while preserving the Sounio-owned Visual IR as the semantic source.
- Labels and text controls use fixed `[i8; 64]` buffers in the Visual IR. Canvas renders them with `display::font`; HTML/SVG escapes XML-sensitive ASCII before export.
- Native frontend builders cover button, slider, toggle, tabs, legend, tooltip, text viewport, and plot viewport nodes over the same control reducer machinery.
- `display::event::Event` can be reduced directly into `VizScene` for headless tests and future native window loops.
- Native windows are an optional layer over the same app runner; CI proofs stay headless.
- `render::renderer3d` remains a CPU Canvas renderer. Its v1.1 path adds Blinn-Phong specular shading and antialiased triangle edge lines while keeping 3D scene meaning in Sounio data.
- GPU rendering is deferred until general compiler GPU backend wiring is ready.
