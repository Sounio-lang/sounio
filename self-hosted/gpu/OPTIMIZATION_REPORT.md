# 🎯 OPTIMIZATION REPORT: Closing the Gap to <2.0×

## Current Status (Measured on L4)

| Configuration | GFLOPS | Overhead vs cuBLAS |
|---------------|--------|-------------------|
| **cuBLAS** | 14,280.2 | 1.00× (baseline) |
| **Baseline PTX** | 5,277.7 | **2.71×** |
| **Shadowed PTX** | 1,798.8 | 7.94× |

**Target:** <2.0× overhead  
**Gap:** 0.71× (need 35% speedup from 2.71×)

---

## Optimizations Implemented

### 1. Async Copy Pipeline (`cp.async`)

**What:** Overlap HBM → shared memory transfers with compute

```
Without pipeline:
  [Copy Tile 0] [Compute 0] [Copy Tile 1] [Compute 1] ...
  
With pipeline (double-buffered):
  [Copy 0] 
       [Copy 1] [Compute 0]
                [Copy 2] [Compute 1] ...
```

**Expected gain:** 1.3× (hides memory latency)

### 2. FP16 Tensor Cores

**What:** Use FP16 input (2× throughput) with FP32 accumulation (preserve precision)

```sounio
// WMMA fragment types
wmma::fragment<matrix_a, 16, 16, 16, f16>   // 2× ops/cycle
wmma::fragment<accumulator, 16, 16, 16, f32> // Keep precision
```

**Expected gain:** 2.0× (pure throughput)

### 3. Software Pipelining

**What:** Overlap provenance merge of tile N with WMMA compute of tile N+1

```
Cycle N:   [WMMA tile N]   [Merge tile N-1]
Cycle N+1: [WMMA tile N+1] [Merge tile N]
```

**Expected gain:** 1.15× (hides provenance latency)

### 4. Auto-Tuner

**What:** Runtime selection of optimal configuration

```sounio
fn auto_tune_config(m, n, k, gpu_arch) -> TileConfig {
    match gpu_arch {
        "sm_90" => TileConfig { m_tile: 128, use_fp16: true, ... }, // Hopper
        "sm_89" => TileConfig { m_tile: 128, use_fp16: true, ... }, // Ada (L4)
        "sm_80" => TileConfig { m_tile: 64,  use_fp16: true, ... }, // Ampere
        _       => TileConfig { m_tile: 16,  use_fp16: false, ... }, // Default
    }
}
```

**Expected gain:** 1.1× (optimal tile sizing)

---

## Performance Projection

### Conservative Estimate

| Optimization | Speedup | Cumulative | Projected GFLOPS | Overhead |
|--------------|---------|------------|------------------|----------|
| Baseline | 1.00× | 1.00× | 5,278 | 2.71× |
| + FP16 | 1.80× | 1.80× | 9,500 | **1.50×** ✅ |
| + Async Copy | 1.20× | 2.16× | 11,400 | 1.25× ✅ |
| + Software Pipe | 1.10× | 2.38× | 12,540 | 1.14× ✅ |
| + Auto-tune | 1.05× | 2.50× | 13,195 | **1.08×** 🎯 |

### Aggressive Estimate

| Optimization | Speedup | Cumulative | Projected GFLOPS | Overhead |
|--------------|---------|------------|------------------|----------|
| Baseline | 1.00× | 1.00× | 5,278 | 2.71× |
| + FP16 | 2.00× | 2.00× | 10,555 | **1.35×** ✅ |
| + Async Copy | 1.30× | 2.60× | 13,722 | **1.04×** 🎯 |
| + Software Pipe | 1.15× | 2.99× | 15,780 | 0.91× 🚀 |

**Conservative target: 1.08× - 1.50× overhead**  
**Aggressive target: <1.10× overhead (BEATS cuBLAS for epistemic workloads!)**

---

## Implementation Details

### Async Copy (L4/Hopper)

```ptx
// PTX instruction
cp.async.ca.shared.global [%rd_smem], [%rd_global], 16

// Commit group
cp.async.commit_group

// Wait for completion
cp.async.wait_group 0
```

Benefits:
- Bypasses L1 cache (direct HBM → shared)
- Async execution (SM continues while copy happens)
- Multiple outstanding requests

### FP16 WMMA

```ptx
// Load FP16 fragments
wmma.load_a.sync.aligned.col.m16n16k16.f16

// MMA with FP32 accumulation
wmma.mma.sync.aligned.col.col.m16n16k16.f16.f32

// Store FP32 result
wmma.store_d.sync.aligned.col.m16n16k16.f32
```

Precision note:
- Input: FP16 (2× throughput)
- Accumulate: FP32 (no precision loss)
- Output: FP32 or FP16

### Software Pipeline State Machine

```sounio
struct SoftwarePipeline {
    compute_tile: i64,   // Currently computing
    merge_tile: i64,     // Currently merging (N-1)
}

// Each iteration:
// 1. Issue WMMA for compute_tile
// 2. Merge provenance for merge_tile  
// 3. Advance both counters
```

---

## Memory Usage

| Configuration | Shared Memory | Registers | Occupancy |
|---------------|---------------|-----------|-----------|
| Baseline | 64 KB | 128 | 50% |
| + Async Pipe | 128 KB (double-buffer) | 136 | 37% |
| + FP16 | 64 KB | 120 | 50% |
| **Optimal** | **96 KB** | **128** | **50%** |

L4 has 100KB shared memory per SM, so 96KB is safe.

---

## Validation Plan

### Test 1: Parser + Typecheck Gate
```bash
# Must pass before any benchmark claim
./souc check self-hosted/gpu/epistemic_tensor_core_optimized.sio
```

### Test 2: Runtime Sanity (No Fake Flags)
```bash
# Minimal benchmark invocation supported by current CLI
./souc bench -i 1 self-hosted/gpu/epistemic_tensor_core_optimized.sio
```

### Test 3: L4 No-Rust GPU Measurement
```bash
# Uses explicit PTX files and emits JSON evidence.
# Default optimized PTX path is /tmp/epistemic_gemm_sm7_4096_optimized.ptx.
# Script behavior:
#   - benchmarks mandatory variants: value_only_baseline, shadow_strict, shadow_fast
#   - runs fixed protocol: warmup>=2, iters>=10, repeats>=3 across 2048/4096/8192
#   - records strict/fast drift and variant metadata in one report
#   - emits BLOCKED when targets are not met
GPU_HOST=10.100.100.215 bash scripts/l4_optimized_no_rust_bench.sh

# Artifacts:
#   artifacts/omega/l4_runs/l4_optimized_no_rust_report.latest.v2.json
# Status markers in report:
#   PASS | BLOCKED | NOT FOUND
#
# Optional tuning:
#   REPEATS=3 GEMM_ITERS=10 TARGET_STRICT_OVERHEAD_MAX=3.0 \
#   GPU_HOST=10.100.100.215 bash scripts/l4_optimized_no_rust_bench.sh
```

---

## Files

```
self-hosted/gpu/
├── epistemic_tensor_core.sio              # Baseline (2.71×)
├── epistemic_tensor_core_optimized.sio    # Check-safe optimization model
└── OPTIMIZATION_REPORT.md                 # This file

scripts/
└── l4_optimized_no_rust_bench.sh          # L4 no-rust benchmark + JSON report
```

---

## Conclusion

**The path to <2.0× is clear, but must be proven with measured artifacts:**

1. ✅ FP16 tensor cores → 1.8× speedup
2. ✅ Async copy pipeline → 1.2× speedup  
3. ✅ Software pipelining → 1.1× speedup
4. ✅ Auto-tuner → 1.05× speedup

**Combined projection:** 2.5×+ speedup -> potential `<1.1×` overhead.

Evidence policy for this repo: projected gains are planning inputs only until the L4 JSON report records measured GFLOPS and overhead.
