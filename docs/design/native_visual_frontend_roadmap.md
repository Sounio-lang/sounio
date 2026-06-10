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

## V1.5: Static HTML/SVG

- `stdlib/viz/viz_html.sio` serializes the same Visual IR as static markup.
- JavaScript is not used for scientific semantics.
- Exported markup can be viewed externally, but the model remains Sounio data.

## V2: GPU Renderer

Deferred until general `bin/souc --backend gpu` wiring exists for ordinary programs.

GPU work should add a renderer over Visual IR, not move molecule, field, unit, uncertainty, or simulation semantics into shader-side ad hoc state.
