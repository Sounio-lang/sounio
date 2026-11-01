# Render Platform Navigation

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

1. **No imports**: inline everything the example needs
2. **Array size**: `[i64; 65536]` not `[i64; 16384]` (128×128 = 16384, but 256×256 = 65536 is the frame contract)
3. **Effect annotations**: `with IO, Mut, Div, Panic` required on any fn calling raster helpers
4. **Helper ordering**: helpers must be defined before their callers
5. **Flat locals**: `var fb_r: [i64; 65536]` in `main()`
