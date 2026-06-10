# Native Visual Frontend Roadmap

Sounio visualization is moving from chart helpers to a native visual frontend.

## V1: CPU Canvas

- Visual IR in `stdlib/viz/ir.sio`.
- Native renderer in `stdlib/viz/viz_canvas.sio`.
- Scientific layers in `stdlib/viz/physchem.sio` and existing `viz::sci`.
- Headless CI proof through `tests/run-pass/viz_headless.sio`.
- No display server required.

## V1.5: Static HTML/SVG

- `stdlib/viz/viz_html.sio` serializes the same Visual IR as static markup.
- JavaScript is not used for scientific semantics.
- Exported markup can be viewed externally, but the model remains Sounio data.

## V2: GPU Renderer

Deferred until general `bin/souc --backend gpu` wiring exists for ordinary programs.

GPU work should add a renderer over Visual IR, not move molecule, field, unit, uncertainty, or simulation semantics into shader-side ad hoc state.
