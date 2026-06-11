# Viz Examples

## Headless Proof

Run:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh run tests/run-pass/viz_headless.sio
```

Expected output:

```text
VIZ_HEADLESS_PASS
```

The test renders axes, an antialiased line, a triangle, a heatmap, an epistemic band, a molecule node, and a Visual IR scene into a heap canvas.

## Canvas Extension Proof

Run:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh run tests/run-pass/viz_canvas_ext_surface.sio
```

Expected output:

```text
VIZ_CANVAS_EXT_SURFACE_PASS
```

The test proves axis-aligned antialiased lines are rendered as exact one-pixel strokes, diagonal lines produce partial coverage, and `graphics::surface::Surface` blits into `display::Canvas` with alpha blending.

## Static HTML Proof

Run:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh run tests/run-pass/viz_html_static.sio
```

Expected output includes:

```text
VIZ_HTML_STATIC_PASS
```

The test emits static SVG/HTML for a line chart, heatmap cells, molecule glyph, scalar-field heatmap cells, vector-field arrows, trajectory path with a scene-time cursor, spectrum trace, and mesh glyph.

## Chart Builders Proof

Run:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh run tests/run-pass/viz_chart_builders.sio
```

Expected output includes:

```text
VIZ_CHART_BUILDERS_PASS
```

The test proves Visual IR builders and Canvas/HTML lowering for scatter, bar, forest, and waterfall nodes. Line, heatmap, and uncertainty-band builders are covered by the headless and static HTML proofs.

## Layout Proof

Run:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh run tests/run-pass/viz_layout.sio
```

Expected output:

```text
VIZ_LAYOUT_PASS
```

The test applies row and column layout to Visual IR control nodes, verifies hit testing uses the resolved rectangles, and renders the laid-out controls into a headless Canvas.

## Physchem Proof

Run:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh run tests/run-pass/viz_physchem.sio
```

Expected output:

```text
VIZ_PHYSCHEM_PASS
```

The test validates H2O, a coarse rapamycin molecule, scalar-field variance/data, vector-field variance/data, trajectory variance/path data, spectrum intensity/variance data, Visual IR molecule and field slots, scene time, and headless Canvas pixels for the rendered physchem nodes.

## Interaction Time Proof

Run:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh run tests/run-pass/viz_interaction_time.sio
```

Expected output:

```text
VIZ_INTERACTION_TIME_PASS
```

The test treats pointer input as Sounio data: a slider hit selects a Visual IR node, pointer movement clamps the slider value, updates `scene.time`, and redraws a trajectory cursor from that state.

## Control Modes Proof

Run:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh run tests/run-pass/viz_controls_modes.sio
```

Expected output:

```text
VIZ_CONTROLS_MODES_PASS
```

The test proves native control reducers for tabs, toggles, hover state, Canvas rendering, and static HTML/SVG serialization over the same Visual IR.

## Text Controls Proof

Run:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh run tests/run-pass/viz_text_controls.sio
```

Expected output includes:

```text
VIZ_TEXT_CONTROLS_PASS
```

The test proves fixed-buffer Visual IR labels, legend text, tooltip text, Canvas text pixels, and static HTML/SVG text export including escaping for `&`, `<`, `>`, and `"`.

## Frontend Builders Proof

Run:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh run tests/run-pass/viz_frontend_builders.sio
```

Expected output includes:

```text
VIZ_FRONTEND_BUILDERS_PASS
```

The test proves named Visual IR builders for button, slider, toggle, tabs, text viewport, plot viewport, legend, and tooltip controls. It checks control slots, values, fixed labels, Canvas pixels, and static HTML/SVG output.

## Event Reducer Proof

Run:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh run tests/run-pass/viz_event_reducer.sio
```

Expected output:

```text
VIZ_EVENT_REDUCER_PASS
```

The test feeds `display::event::Event` values into `viz_scene_handle_event`, proving that mouse press/motion/release and keyboard arrows/space update the Sounio Visual IR state directly.

## Identity, Focus, And Dump Proof

Run:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh run tests/run-pass/viz_identity_focus_dump.sio
```

Expected output includes:

```text
VIZ_IDENTITY_FOCUS_DUMP_PASS
```

The test proves scene-local node IDs, `parent_id`, app-owned tags, focus next/previous traversal, focused activation, keyboard fallback to the focused control, and deterministic `viz_scene_dump` output.

## Lab Interaction Proof

Run:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh run tests/run-pass/viz_lab_interaction.sio
```

Expected output:

```text
VIZ_LAB_INTERACTION_PASS
```

The test drives the native lab scene as Sounio data: pointer events move the time slider and switch tabs, keyboard Tab moves focus, Space activates a focused toggle, Canvas pixels are checked, and `viz_app_frame_hash` proves the frame snapshot changed after interaction.

## HTML Identity Proof

Run:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh run tests/run-pass/viz_html_identity.sio
```

Expected output includes:

```text
data-viz-id="1"
data-viz-parent-id="1"
data-viz-tag="77"
VIZ_HTML_IDENTITY_PASS
```

The test proves static HTML/SVG export preserves Visual IR identity, parent identity, tag, kind, and data-slot metadata as passive attributes. JavaScript still owns no scientific semantics.

## Replay Determinism Proof

Run:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh run tests/run-pass/viz_replay_deterministic.sio
```

Expected output includes:

```text
VIZ_REPLAY_DETERMINISTIC_PASS
```

The test replays the same fixed event tape twice through `VizApp`, `VizScene`, Canvas rendering, and `viz_app_frame_hash`. It proves the final frame hash and interaction state match, while the dump shows node identity/tags and final hover/focus state.

## Inspector Panel Proof

Run:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh run tests/run-pass/viz_inspector_panel.sio
```

Expected output:

```text
VIZ_INSPECTOR_PANEL_PASS
```

The test draws the native Canvas inspector panel over a focused Visual IR scene and verifies panel/text pixels without requiring a display server.

## Replay Trace Proof

Run:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh run tests/run-pass/viz_replay_trace.sio
```

Expected output:

```text
VIZ_REPLAY_TRACE_PASS
```

The test records a timeline of event kind/key/position, changed flag, Canvas hash, audit hash, selected/hovered/focused state, active tab, and scene time for every replayed frame.

## Inspector Interactive Proof

Run:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh run tests/run-pass/viz_inspector_interactive.sio
```

Expected output:

```text
VIZ_INSPECTOR_INTERACTIVE_PASS
```

The test clicks an inspector row, selects/focuses/hovers the corresponding Visual IR node, verifies shared audit hashes, and checks the native Canvas highlight/text pixels.

## Scene Editor Proof

Run:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh run tests/run-pass/viz_editor_scene.sio
```

Expected output includes:

```text
data-viz-tag="7202"
VIZ_EDITOR_SCENE_PASS
```

The test selects a Visual IR node through the inspector, edits its tag/value/rectangle, replays keyboard edit events into a deterministic edit trace, checks Canvas highlight pixels, and emits HTML/SVG with the edited identity and geometry.

## Visual Workbench Roundtrip

Run:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh run tests/run-pass/viz_workbench_roundtrip.sio
```

Expected output includes:

```text
data-viz-tag="9101"
VIZ_WORKBENCH_ROUNDTRIP_PASS
```

The test builds the reusable Workbench scene, renders Canvas, replays a deterministic interaction script, selects a node through the inspector, edits the selected Visual IR node, redraws highlights, and exports HTML/SVG from the same edited scene.

## Renderer3D Edge Proof

Run:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh run tests/run-pass/viz_renderer3d_edges.sio
```

Expected output:

```text
VIZ_RENDERER3D_EDGES_PASS
```

The test renders one software 3D triangle into a headless Canvas and checks both filled face pixels and antialiased edge-line pixels.

## App Frame Proof

Run:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh run tests/run-pass/viz_app_frame.sio
```

Expected output:

```text
VIZ_APP_FRAME_PASS
```

The test proves the `viz_app` runner: events mark an app dirty, dirty scenes render to Canvas exactly once, expose events force redraw, and close events stop the app without requiring a display server.

## Native Lab

Compile:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh compile examples/viz_lab/main.sio -o /tmp/viz_lab.elf
```

The lab currently proves the shared scene model with `viz_app` event handling and Canvas rendering, then prints `VIZ_LAB_READY`. Opening a real window remains optional manual work for the next phase.

## Native Window Lab

Compile:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh compile examples/viz_lab_window/main.sio -o /tmp/viz_lab_window.elf
```

The window lab uses `viz_window_step` to pump `display::Window` events into `VizApp`/`VizScene` and present Canvas frames. It is compile-gated for CI and intended for manual runs when `DISPLAY` is available.

## Physchem Demo

Compile or run:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh run examples/viz_physchem_demo/main.sio
```

The demo builds one Visual IR scene with H2O, coarse rapamycin, a scalar field, a vector field, a trajectory, an absorption spectrum, a 3D mesh, and a time-evolution chart with an uncertainty band. It renders the same scene to native Canvas pixels and static HTML/SVG, then prints `VIZ_PHYSCHEM_DEMO_READY`.

## Hello

Compile:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh compile examples/viz_hello/main.sio -o /tmp/viz_hello.elf
```

The smallest example builds one Visual IR line chart and renders it into a headless Canvas.

## Renderer Split

Build scenes with `viz::ir`, then choose a renderer:

- `viz_canvas_render_scene(scene, canvas)` for native Canvas.
- `viz_html_emit_scene(scene, width, height)` for static HTML/SVG output.

The same node/data slots are used by both renderers.
