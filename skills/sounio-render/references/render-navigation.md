# Render Platform Navigation

## stdlib/render/ Modules

| Module | Purpose | Key Functions |
|--------|---------|--------------|
| `types.sio` | Core structs | `Vec3`, `Color`, `Vertex2D`, `Ray`, `AABB` |
| `framebuffer.sio` | Pixel buffer + PPM output | `fb_new`, `fb_set_pixel`, `fb_set_pixel_depth`, `fb_clear`, `fb_emit_ppm`, `clamp_byte`, `float_to_byte` |
| `rasterizer.sio` | CPU rasterizer | `edge_fn`, `rasterize_triangle`, `rasterize_line`, `rasterize_rect` |
| `scene.sio` | 3D projection | `ScreenPt`, `project_point3d`, `Camera`, `scene_default_camera` |
| `pipeline.sio` | Full render pass | `render_scene`, `RenderStats` |
| `epistemic.sio` | Uncertainty visualization | `uncertainty_to_alpha`, `value_to_color`, `epistemic_heatmap_cell` |

## examples/render/ Examples

| File | Status | Dimensions | Description |
|------|--------|-----------|-------------|
| `triangle_basic.sio` | Real render | 128×128 | Barycentric triangle + background gradient |
| `triangle_ppm.sio` | Real render | 128×128 | Direct PPM triangle output |
| `cube_wireframe.sio` | Real render | 192×192 | Bresenham wireframe cube, perspective projection |
| `uncertainty_ppm.sio` | Real render | 128×128 | 16×16 epistemic heatmap grid |
| `uncertainty_field.sio` | Real render | 128×128 | Full epistemic field with value_to_color |
| `causal_dag.sio` | Real render | 256×128 | Front-door DAG: X→M→Y with U latent |
| `quaternion_rotation.sio` | Real render | 192×192 | Quaternion-rotated tetrahedron wireframe |

## GPU Emitters (self-hosted/gpu/)

| File | Language | Output |
|------|---------|--------|
| `ptx_render.sio` | PTX (NVIDIA) | `.ptx` text + scene header comment |
| `spirv_render.sio` | SPIR-V | Hex dump + scene triangle count |
| `metal_render.sio` | Metal MSL | `.metal` text + scene comment |
| `wgsl_render.sio` | WebGPU WGSL | `.wgsl` text + scene comment |

## Website Asset Pipeline

```
examples/render/*.sio
    ↓ souc run
PPM P3 stdout
    ↓ parsePpm()
pixel data
    ↓ ppmToSvg()
SVG with viewBox="0 0 W H"
    ↓ written to
website/public/assets/generated/render/*.svg
website/public/assets/generated/render/manifest.json
```

Script: `website/scripts/render-assets.mjs`
Config: `renderSpecs` array (add new entries here)
Check mode: `node website/scripts/render-assets.mjs check` (non-zero = stale)

## Key Constraints for render examples

1. **No imports**: `use` is not supported; inline everything from stdlib
2. **Array size**: `[i64; 65536]` not `[i64; 16384]` (128×128 = 16384, but 256×256 = 65536 is the shared contract)
3. **No negative literals**: `0.0 - 0.471` not `-0.471`
4. **Effect annotations**: `with IO, Mut, Div, Panic` required on any fn calling raster helpers
5. **Helper ordering**: helpers must be defined before their callers
6. **Flat locals**: `var fb_r: [i64; 65536]` not `fb_new()` (struct too large for stack in main)

## Common Copy Patterns

**inline `emit_ppm`** (from framebuffer.sio:75-97):
```sio
fn emit_ppm(width: i64, height: i64, fb_r: &[i64; 65536], fb_g: &[i64; 65536], fb_b: &[i64; 65536]) with IO, Mut, Panic {
    print("P3\n")
    print_int(width)
    print(" ")
    print_int(height)
    print("\n255\n")
    var i: i64 = 0
    let total = width * height
    while i < total {
        print_int((*fb_r)[i])
        print(" ")
        print_int((*fb_g)[i])
        print(" ")
        print_int((*fb_b)[i])
        print("\n")
        i = i + 1
    }
}
```

**inline `edge_fn`** (from rasterizer.sio:17-19):
```sio
fn edge_fn(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> f64 {
    (cx - ax) * (by - ay) - (cy - ay) * (bx - ax)
}
```

**inline `clamp_byte`** (from framebuffer.sio:63-67):
```sio
fn clamp_byte(v: i64) -> i64 {
    if v < 0 { return 0 }
    if v > 255 { return 255 }
    v
}
```
