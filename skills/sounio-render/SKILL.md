---
name: sounio-render
description: "Work on Sounio render examples, website showcase generation, and GPU backend emitters; use when editing examples/render or the website render preview."
---

# Sounio Render Platform

## Overview

Covers render examples, GPU preview emitters, and website showcase assets. Examples under `examples/render/` are self-contained (no `use` imports) and output PPM P3 format to stdout.

## Workflow

### 1) Complete or edit a render example

- Examples are **self-contained**: inline all needed functions (no `use` imports).
- Use `[i64; 65536]` framebuffer arrays; `[i64; 16384]` is undersized for 256-wide frames.
- Effect annotations required: `with IO, Mut, Div, Panic` on any function calling raster helpers.
- `var` for mutable locals, `let` for immutable. No `pub`, no Rust macros.
- Keep `var fb_r: [i64; 65536]` as flat locals in `main()`.

### 2) Spot-check PPM output

```bash
SOUC=./bin/souc
$SOUC run examples/render/uncertainty_field.sio > /tmp/uf.ppm && head -3 /tmp/uf.ppm
# Expected: P3 / 128 128 / 255
```

PPM dimensions verified per example:

| Example | Width | Height |
|---------|-------|--------|
| `triangle_basic.sio` | 128 | 128 |
| `triangle_ppm.sio` | 128 | 128 |
| `cube_wireframe.sio` | 192 | 192 |
| `uncertainty_ppm.sio` | 128 | 128 |
| `uncertainty_field.sio` | 128 | 128 |
| `causal_dag.sio` | 256 | 128 |
| `quaternion_rotation.sio` | 192 | 192 |

### 3) GPU emitter changes

- Each GPU render file (`ptx_render.sio`, `metal_render.sio`, `spirv_render.sio`, `wgsl_render.sio` in `self-hosted/gpu/`) is **additive-only**.
- Existing `main()` must continue working; new scene-aware functions are added alongside it.
- Scene contract: add `struct RenderScene` + `fn render_scene_default()`.

### 4) Regenerate website assets after any example change

```bash
npm --prefix website run generate:render-assets
# Commits new SVGs under website/public/assets/generated/render/
```

- `website/scripts/render-assets.mjs` is data-driven via `renderSpecs` array.
- To add a new example to the website: append an entry to `renderSpecs` — no other script changes needed.
- SVG dimension check: `viewBox="0 0 W H"` must match PPM dimensions.

## References

- Examples: `examples/render/`
- GPU emitters: `self-hosted/gpu/ptx_render.sio`, `metal_render.sio`, `spirv_render.sio`, `wgsl_render.sio`
- Website: `website/scripts/render-assets.mjs`, `website/src/content/showcases/graphics.mdx`
- Navigation: `skills/sounio-render/references/render-navigation.md`
