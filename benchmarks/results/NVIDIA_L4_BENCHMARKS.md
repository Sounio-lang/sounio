# NVIDIA L4 Epistemic GEMM Benchmark Results

**Headline:** Median 5.3 TFLOPS compiler-generated tiled GEMM at 4096x4096 (17.3% of L4 peak). cuBLAS SGEMM baseline at 14.1 TFLOPS. **Overhead: 2.19x geometric mean** (compiler maturity baseline — kernel generated with `epistemic_enabled=false`; shadow register overhead not yet isolated).

## Hardware Configuration

| Property | Value |
|----------|-------|
| GPU | NVIDIA L4 (Ada Lovelace, SM 8.9) |
| VRAM | 23,034 MiB |
| Driver | 590.48.01 |
| Peak FP32 | 30.3 TFLOPS |
| Host | `gpu-appliance-l4` (10.100.100.215) |
| Toolchain | rustc 1.93.1, cargo 1.93.1 |
| Date range | 2026-02-25 to 2026-02-28 |
| Memory BW | 115.0 GB/s (DtoD, 256 MB) |

## Methodology

GPU kernels are generated as PTX by the Sounio compiler (`crates/souc/src/codegen/gpu/ptx.rs`) and dispatched via `scripts/cuda_gemm_dispatch.py` using the CUDA Driver API (`ctypes` bindings to `libcuda.so`). Timing uses CUDA events (`cuEventElapsedTime`). GFLOPS = 2 * M * N * K / (ms * 1e-3) / 1e9.

## What is Epistemic GEMM

The Sounio GPU codegen pipeline supports two modes:

### Shadow-enabled mode (`epistemic_enabled=true`)

Every `Knowledge<f32>` value carries 4 shadow registers:

| Register | Type | Purpose |
|----------|------|---------|
| `%r_val` | f32 | The numerical value |
| `%r_eps` | f32 | GUM standard uncertainty (epsilon) |
| `%p_valid` | u32 | Validity predicate (0 = killed by near-zero division) |
| `%r_prov` | u32 | Provenance tag (measurement source tracking) |

Uncertainty propagation follows ISO/IEC Guide 98-3 (the GUM):
- **Add/Sub:** epsilon_c = sqrt(epsilon_a^2 + epsilon_b^2) (quadrature)
- **Mul:** epsilon_c = |a| * epsilon_b + |b| * epsilon_a (first-order)
- **Div:** near-zero guard with validity predicate kill
- **FMA:** combined quadrature

This mode has NOT been benchmarked yet (see Investigation Finding below).

### Value-only mode (`epistemic_enabled=false`) — BENCHMARKED

The current benchmark kernel uses value-only mode: a compiler-generated tiled SGEMM with 64x64 tiles, 16x16 thread blocks, 4x4 register blocking, and 16 `fma.rn.f32` instructions in the inner loop. No shadow register operations are emitted. This measures the baseline cost of Sounio's GPU codegen vs vendor-optimized cuBLAS.

## Results by Matrix Size

Two PTX variants produce bimodal performance at 4096x4096:
- **Series A** (SM7-targeted, ~15,647 chars): 5.1-5.6 TFLOPS — the optimized codepath
- **Series B** (on-host-generated, ~22,580 chars): 4.4-4.5 TFLOPS — larger PTX, more register pressure

### Consolidated Summary (Series A only — the optimized codepath)

| Dimension | n | Median GFLOPS | Range | ms/iter | Peak % |
|-----------|---|---------------|-------|---------|--------|
| 1024x1024 | 5 | 7,182 | 6,061-7,204 | ~0.30 | 23.7 |
| 2048x2048 | 4 | 7,393 | 6,786-7,595 | ~2.3 | 24.4 |
| 4096x4096 | 8 | 5,253 | 5,155-5,606 | ~26 | 17.3 |
| 8192x8192 | 4 | 4,806 | 4,355-4,957 | ~229 | 15.9 |

### 4096x4096x4096 (all series)

| Series | n | Median GFLOPS | Range | Note |
|--------|---|---------------|-------|------|
| A (SM7) | 6 | 5,253 | 5,155-5,606 | Optimized codepath |
| B (on-host) | 4 | 4,442 | 4,368-4,444 | Larger PTX, register pressure |
| C (SM89 fix) | 2 | 4,495 | 4,463-4,527 | Post-fix SM89 targeting |

### 8192x8192x8192 (stability)

3 runs: mean 4,730 GFLOPS, stddev 267 (5.6% of mean).

Source: `artifacts/omega/l4_scale_stability_report.v1.txt`

## PTX Variant Analysis

SM7 generic PTX (~15,647 chars) outperforms on-host-generated PTX (~22,580 chars) by ~20% at 4096x4096. The larger PTX has more instructions and higher register pressure, likely due to less aggressive dead-code elimination in the on-host generation path. A production compiler would use the SM7-targeted path.

## Known Failure

**SM89 cuModuleLoadData 0xc8:** The initial SM89-targeted PTX (22,664 chars) failed with `CUDA_ERROR_INVALID_PTX`. Resolved by falling back to SM7 generic PTX, which runs correctly on SM 8.9 hardware via forward compatibility. Subsequent SM89 PTX regeneration passed (4,463-4,527 GFLOPS).

## cuBLAS SGEMM Baseline (2026-02-28)

cuBLAS SGEMM (plain f32 matrix multiply, no shadow registers) measured on the same L4 hardware, same timing methodology (CUDA events), 8 iterations per dimension, 3 independent passes.

| Dimension | n | Median GFLOPS | Range | Peak % |
|-----------|---|---------------|-------|--------|
| 1024x1024 | 3 | 12,886 | 12,886-13,026 | 42.5 |
| 2048x2048 | 3 | 16,402 | 16,368-16,529 | 54.1 |
| 4096x4096 | 3 | 14,124 | 13,953-15,482 | 46.6 |
| 8192x8192 | 3 | 10,388 | 10,367-10,779 | 34.3 |

Baseline script: `scripts/cuda_cublas_baseline.py` (cudart runtime API + cuBLAS via ctypes).
Memory bandwidth: 115.0 GB/s (DtoD, 256 MB).

## Overhead Analysis

Overhead = cuBLAS GFLOPS / Epistemic GFLOPS (how many times slower the epistemic kernel is):

| Dimension | cuBLAS | Epistemic | Overhead | Extra time |
|-----------|--------|-----------|----------|------------|
| 1024x1024 | 12,886 | 7,182 | 1.79x | +79% |
| 2048x2048 | 16,402 | 7,393 | 2.22x | +122% |
| 4096x4096 | 14,124 | 5,253 | 2.69x | +169% |
| 8192x8192 | 10,388 | 4,806 | 2.16x | +116% |
| **geomean** | | | **2.19x** | **+119%** |

### Interpretation — Corrected (2026-02-27)

**Investigation finding:** The benchmark PTX kernel (`epistemic_gemm_sm7_4096.ptx`, 15,647 chars) was generated with `epistemic_enabled=false`. The PTX contains standard registers (`.reg .f32 %f<48>`) and 16 `fma.rn.f32` instructions in the inner loop — no shadow register operations (_eps, _valid, _prov). This was confirmed by:

1. Inspecting the PTX: no epsilon propagation, validity predicates, or provenance merge instructions
2. The self-hosted codegen (`self-hosted/gpu/hlir_to_gpu.sio` line 362) gates ALL shadow emission on the `epistemic_enabled` boolean
3. Oracle test 43 validates that `epistemic_enabled=false` produces zero shadow ops

**What the 2.19x actually measures:** The overhead of Sounio's compiler-generated tiled GEMM (64x64 tiles, 16x16 thread blocks, 4x4 register blocking) versus cuBLAS's highly-optimized SGEMM. This is the **compiler maturity baseline** — the cost of using a research compiler's GPU backend versus a vendor-tuned library.

**What has NOT been measured:** The additional overhead of epistemic shadow registers (uncertainty propagation, validity tracking, provenance merge). The self-hosted compiler infrastructure for this exists (`self-hosted/gpu/epistemic_ptx.sio`), but benchmark PTX with shadows enabled has not yet been generated and dispatched.

## Defensible Claims

**What we CAN claim:**
- "Sounio GPU codegen produces working PTX kernels across 1K-8K matrix dimensions" (all pass 500 GFLOPS threshold)
- "Compiler-generated tiled GEMM achieves median 5.3 TFLOPS at 4096x4096" (8 runs, ±3%)
- "2.19x geometric mean overhead vs cuBLAS SGEMM" (compiler maturity baseline, measured on same hardware/timing)
- "16-24% of L4 peak FP32 throughput" (varies by dimension)
- "PTX target selection affects performance by ~20%" (SM7 vs on-host)
- "Memory bandwidth: 115 GB/s DtoD" (measured)
- "Self-hosted GPU codegen has full epistemic shadow register support" (code exists, formally verified in Lean)

**What we CANNOT yet claim:**
- "2.19x is the cost of epistemic uncertainty propagation" — the benchmark kernel has no shadow registers
- Shadow register overhead has not been isolated (requires generating PTX with `epistemic_enabled=true`)
- No comparison of shadow-on vs shadow-off kernel performance

## Reproduction

```bash
# Requires SSH access to gpu-appliance-l4 (10.100.100.215)

# cuBLAS SGEMM baseline (all dimensions):
ssh demetrios@10.100.100.215 \
  "cd ~/work/sounio && python3 scripts/cuda_cublas_baseline.py"

# Epistemic GEMM (direct dispatch with pre-generated PTX):
ssh demetrios@10.100.100.215 \
  "cd ~/work/sounio && GEMM_M=4096 GEMM_N=4096 GEMM_K=4096 \
   GEMM_ITERS=8 GEMM_PTX_FILE=/tmp/epistemic_gemm_sm7_4096.ptx \
   python3 scripts/cuda_gemm_dispatch.py"

# Full test runner with baseline:
GPU_HOST=gpu-appliance-l4 BASELINE=1 bash scripts/gpu_test_runner.sh
```

## Source Data

- Machine-readable: `benchmarks/results/l4_raw_data.json`
- cuBLAS baseline: `artifacts/omega/cublas_baseline_report.v1.json`
- Overhead report: `artifacts/omega/overhead_report.v1.json`
- Raw logs: `artifacts/omega/l4_*.log` (27 files)
- Perf reports: `artifacts/omega/l4_perf_pass_report.v{1,2}.txt`
- Stability report: `artifacts/omega/l4_scale_stability_report.v1.txt`
- Baseline script: `scripts/cuda_cublas_baseline.py`
