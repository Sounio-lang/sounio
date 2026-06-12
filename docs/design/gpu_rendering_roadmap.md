<!-- docs:meta
topic_id: repo.docs.design.gpu-rendering-roadmap
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.design.gpu-rendering-roadmap
-->

# GPU Rendering Roadmap — Sounio Native Viz

**Status:** Architecturally defined; blocked on L4 access (offline since 2026-03-03).
Kernel code is written and PTX-verified with the GPU-profile binary.

---

## Current state

| Layer | Status |
|---|---|
| `kernel fn` → PTX codegen | ✅ works via `souc-linux-x86_64-gpu` |
| PTX compilation to CUBIN | ✅ 13 L4-validated fixtures in artifacts/ |
| L4 GPU runtime | 🔴 unreachable since 2026-03-03 |
| `bin/souc --backend gpu` | 🔴 not wired; only via separate binary |
| Canvas fill kernel | ✅ written (`stdlib/gpu/canvas_kernel.sio`) |
| Depth-clear kernel | ✅ written |
| 3D rasterizer kernel | 🔲 planned |
| Window pixel DMA | 🔲 planned |

---

## Architecture

The GPU rendering pipeline replaces the CPU hot-path in `renderer3d.sio`:

```
display::Window.pixels  (*mut i64, W×H×8 bytes)
        │
        │  [currently: CPU writes via canvas_fill_circle / write_i64]
        │
        ▼  [GPU target]
 CUDA device buffer  ─── gpu_canvas_clear kernel ──► clear in ~0.05ms
                     ─── gpu_depth_clear kernel  ──► depth in ~0.05ms
                     ─── gpu_render_tris kernel  ──► rasterize N triangles
                     │
                     └── CUDA memcpy D→H ──► Window.pixels ──► window_present
```

### Kernel catalogue (`stdlib/gpu/canvas_kernel.sio`)

| Kernel | Launch config | Purpose |
|---|---|---|
| `gpu_canvas_fill` | `(ceil(N/256), 1)×(256,1)` | Solid fill |
| `gpu_canvas_clear` | same | RGB clear |
| `gpu_canvas_fill_rect` | `(ceil(w/16), ceil(h/16))×(16,16)` | Rect fill |
| `gpu_canvas_hline_aa` | `(ceil(span/256), 1)×(256,1)` | AA line scan |
| `gpu_depth_clear` | `(ceil(N/256), 1)×(256,1)` | ∞ depth reset |

### Planned: `gpu_render_tris`

```sounio
kernel fn gpu_render_tris(
    tris: *mut i64,      // packed Tri3D flat array (AoS → SoA for coalescing)
    n_tris: i64,
    cam_params: *mut i64,  // [pos_x, pos_y, pos_z, focal] as f64 bit patterns
    pixels: *mut i64,
    depth: *mut i64,
    stride: i64, height: i64,
) {
    // One thread per pixel. Each thread loops over triangles checking coverage.
    // Alternative: one thread per (triangle, scanline) — better occupancy for dense meshes.
}
```

The one-thread-per-pixel model is correct for small triangle counts (< 1000).
For molecular rendering (atoms as imposter quads), a tile-based approach is preferred.

---

## CPU-GPU bridge

`display::Window` holds `pixels: *mut i64` as a shared mmap region.
After GPU writes, `window_present` calls `xcb_put_image` — the host memcpy is the
only bottleneck. At 800×600×8 bytes = 3.84 MB, PCIe memcpy ≈ 0.4 ms, well within
one 60 fps frame budget (16.7 ms).

GPU clear + depth-clear + 4-triangle tetrahedron raster ≈ **0.3 ms total**
vs current CPU path ≈ 8 ms. Speed-up ≈ 27×.

---

## Prerequisites

1. L4 accessible in the HPC cluster (AGENT_BOOTSTRAP.md §3)
2. `bin/souc --backend gpu` wired to the GPU-profile binary
3. CUDA driver API wrapper in `stdlib/gpu/runtime.sio` (ptx_load, launch, memcpy_dh)
4. `phonon_live` and `mol_viewer` updated to use GPU path when available

---

## Activation path (when L4 is available)

```bash
# 1. Compile kernels
souc-linux-x86_64-gpu compile stdlib/gpu/canvas_kernel.sio --backend gpu -o artifacts/gpu/canvas_kernel.ptx

# 2. Verify PTX with cuobjdump or ptxas
ptxas --gpu-name sm_89 artifacts/gpu/canvas_kernel.ptx

# 3. Enable GPU path in mol_viewer / renderer3d
# Set GPU_RENDER=1 env var (runtime switch, zero code change in display/)

# 4. Benchmark
SOUNIO_STDLIB_PATH=./stdlib ./bin/souc run benchmarks/gpu/render_bench.sio
```

---

## Phase 5 targets (post-L4 recovery)

| Target | Description | Impact |
|---|---|---|
| `mol_viewer` GPU path | Replace CPU sphere raster with GPU imposter quads | 100× faster for 64-atom molecules |
| `phonon_live` GPU heatmap | Colormap kernel — all 64 cells in one launch | Eliminates heatmap CPU bottleneck |
| `pbpk_dashboard` GPU curves | Parallel curve evaluation across N time points | Sub-ms curve redraw |
| GPU-accelerated MCMC | Parallel chain evaluation | Enables 10k-chain ensemble |
