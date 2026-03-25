---
name: sounio-render
description: "Work on the Sounio render platform: stdlib/render/ API, examples/render/ examples, website showcase generation, and GPU backend emitters; use when editing any render-related .sio file or website preview."
---

# Sounio Render Platform

## Overview

Covers the CPU-rasterizer render pipeline, flagship epistemic scenes, GPU preview emitters, and website showcase assets.

Six modules in `stdlib/render/` define the shared API: `types.sio`, `framebuffer.sio`, `rasterizer.sio`, `scene.sio`, `pipeline.sio`, `epistemic.sio`. All render examples are self-contained (no `use` imports) and output PPM P3 format to stdout.

## Workflow

### 1) Check stdlib/render/ first

- All six modules must pass `souc check` before any example edits.
- Adding a new primitive (e.g. `rasterize_rect`)? Add it to `rasterizer.sio`, verify with a `souc run` smoke test.
- Helper functions must precede callers — no forward references in Sounio.

### 2) Complete or edit a render example

- Examples are **self-contained**: inline all needed functions (no `use` imports).
- Use `[i64; 65536]` framebuffer arrays; `[i64; 16384]` is undersized for the shared API.
- Effect annotations required: `with IO, Mut, Div, Panic` on any function calling raster helpers.
- Negative float literals: write `0.0 - 0.471`, not `-0.471` (Sounio constraint).
- `var` for mutable locals, `let` for immutable. No `pub`, no Rust macros.
- Keep `var fb_r: [i64; 65536]` as flat locals in `main()` — never instantiate `Framebuffer` struct on the stack (it is ~2 MB).

### 3) Run the PPM gate

```bash
SOUC=./bin/souc
bash scripts/sprint53_render_platform_gate.sh
# Spot-check a render:
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

### 4) GPU emitter changes

- Each GPU render file (`ptx_render.sio`, `metal_render.sio`, `spirv_render.sio`, `wgsl_render.sio` in `self-hosted/gpu/`) is **additive-only**.
- Existing `main()` must continue working; new scene-aware functions are added alongside it.
- Scene contract: add `struct RenderScene` + `fn render_scene_default()` (Sprint 54 pattern).
- Gate: `bash scripts/sprint54_gpu_contract_gate.sh`

### 5) Regenerate website assets after any example change

```bash
npm --prefix website run generate:render-assets
# Commits new SVGs under website/public/assets/generated/render/
```

- `website/scripts/render-assets.mjs` is data-driven via `renderSpecs` array.
- To add a new example to the website: append an entry to `renderSpecs` — no other script changes needed.
- SVG dimension check: `viewBox="0 0 W H"` must match PPM dimensions.
- Gate: `bash scripts/sprint55_website_render_gate.sh`

## References

- Render API: `stdlib/render/` — types, framebuffer, rasterizer, scene, pipeline, epistemic
- Examples: `examples/render/`
- GPU emitters: `self-hosted/gpu/ptx_render.sio`, `metal_render.sio`, `spirv_render.sio`, `wgsl_render.sio`
- Website: `website/scripts/render-assets.mjs`, `website/src/content/showcases/graphics.mdx`
- Gates: `scripts/sprint53_render_platform_gate.sh`, `scripts/sprint54_gpu_contract_gate.sh`, `scripts/sprint55_website_render_gate.sh`
- Navigation: `skills/sounio-render/references/render-navigation.md`
