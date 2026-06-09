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

## Native Lab

Compile:

```bash
SOUNIO_STDLIB_PATH=./stdlib ./scripts/ci/souc-native-wrapper.sh compile examples/viz_lab/main.sio -o /tmp/viz_lab.elf
```

The lab currently proves the shared scene model and prints `VIZ_LAB_READY`. Window/event-loop wiring remains optional manual work for the next phase.

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
