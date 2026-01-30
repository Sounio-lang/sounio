# GPU Profiling & Performance Validation Guide

Quick reference for profiling Sounio GPU kernels and validating performance.

## Quick Start

### 1. Run Validation Tests

```bash
cd compiler
cargo test --test gpu_performance_validation -- --nocapture
```

Expected output:
```
test_profiler_tools_detection ... ok
test_dispatch_overhead_measurement ... ok
test_wmma_utilization_validation ... ok
test_int8_vs_fp32_accuracy ... ok
test_all_architectures_dispatch_overhead ... ok
```

### 2. Profile Your Kernel

#### On Linux with perf (CPU overhead measurement)

```bash
# Build your kernel
cargo build --release

# Profile dispatch overhead
perf stat -e task-clock,context-switches,cache-misses \
  ./target/release/your_kernel_binary
```

#### On NVIDIA GPU with ncu

```bash
# Profile kernel execution
ncu --set full ./target/release/your_kernel_binary

# Export results
ncu --set full -o kernel_profile ./your_kernel
ncu --import kernel_profile.ncu-rep
```

#### On macOS with Instruments

```bash
# Profile Metal kernels
xcrun xctrace record --template "Metal" ./kernel_binary
```

## Performance Metrics Reference

### Key Metrics to Track

| Metric | Target | Interpretation |
|--------|--------|-----------------|
| **Occupancy** | >60% | Enough threads in flight to hide latency |
| **WMMA Util** | >80% | Tensor cores actively computing |
| **Warp Efficiency** | >90% | Low divergence, good data layout |
| **Memory Coalesce** | >80% | Good spatial locality |
| **Cache Hit Rate** | >80% | Data reuse is effective |

### Bottleneck Classification

Using roofline analysis:

```
Arithmetic Intensity (FLOPS/byte)
         ↓
    < Ridge Point  → Memory-bound (optimize memory access)
    ≈ Ridge Point  → Balanced (both compute & memory matter)
    > Ridge Point  → Compute-bound (optimize computation)
```

**Ridge Points by Architecture:**
- Turing: ~12.5 FLOPS/byte
- Ampere: ~12.5 FLOPS/byte
- Hopper: ~27.5 FLOPS/byte

## Code Integration Examples

### Example 1: Auto-Detect Best Profiler

```rust
use sounio::codegen::gpu::ProfilerManager;

fn main() -> Result<()> {
    // Auto-select best available profiler
    let profiler = ProfilerManager::select_best_profiler()?;
    println!("Using profiler: {}", profiler.name());

    // Profile your kernel
    let metrics = profiler.profile_kernel(Path::new("kernel.ptx"))?;
    println!("Execution time: {} μs", metrics.execution_time_us);
    println!("FP32 ops: {}", metrics.fp32_ops);

    Ok(())
}
```

### Example 2: WMMA Utilization Check

```rust
use sounio::codegen::gpu::{WmmaUtilizationValidator, CudaArch};

fn validate_wmma(kernel: &GpuKernel) -> Result<()> {
    let validator = WmmaUtilizationValidator::new(CudaArch::Ampere);
    let result = validator.validate_wmma(kernel);

    println!("WMMA Utilization: {:.1}%", result.utilization_percentage);

    if !result.achieves_target {
        println!("Optimization needed:");
        for rec in result.recommendations {
            println!("  - {}", rec);
        }
    }

    Ok(())
}
```

### Example 3: INT8 Accuracy Validation

```rust
use sounio::codegen::gpu::AccuracyComparator;

fn validate_quantization() -> Result<()> {
    let comparator = AccuracyComparator::new(
        0.01,  // 1% relative error threshold
        1e-6   // absolute error threshold
    );

    let comparison = comparator.compare_precision();

    println!("Quantization Success Rate: {:.1}%", comparison.success_rate);

    for result in &comparison.test_results {
        println!("{}: L∞={:.2e}", result.test_name, result.linf_error);
    }

    assert!(comparison.success_rate > 90.0, "Quantization accuracy too low");
    Ok(())
}
```

## Benchmark Suite

### Run All GPU Benchmarks

```bash
cargo bench --bench gpu_bench
cargo bench --bench octonion_bench
cargo bench --bench qnn_performance_bench
cargo bench --bench sir_gpu_bench
```

### Compare Against Baseline

```bash
# Save current as baseline
cargo bench --bench gpu_bench -- --save-baseline main

# Later: compare new code
cargo bench --bench gpu_bench -- --baseline main
```

### View Results

```bash
# Open HTML report
open target/criterion/index.html
```

## Architecture-Specific Tuning

### Turing (sm_75)

```rust
let arch = CudaArch::Turing;
// Limited WMMA (FP32 WMMA from sm_80+)
// Focus: Warp-level reductions, cooperative groups
// Peak: 10-12 TFLOPS FP32
```

### Ampere (sm_80)

```rust
let arch = CudaArch::Ampere;
// Full WMMA support: FP32, TF32, BF16, FP8
// Async copy (CpAsyncBuilder)
// Tensor Float 32 (TF32) for training
// Peak: 19.5 TFLOPS FP32
```

### Ada (sm_89)

```rust
let arch = CudaArch::Ada;
// Double-buffered WMMA
// Enhanced sparsity support
// Distributed shared memory
// Peak: 19.5 TFLOPS FP32
```

### Hopper (sm_90)

```rust
let arch = CudaArch::Hopper;
// Transformer Engine (sparsity + quantization)
// Async primitives for better latency hiding
// 128-thread WMMA blocks
// Peak: 67.1 TFLOPS FP32
```

## Common Issues & Solutions

### Issue: Low WMMA Utilization

**Symptoms:**
- WMMA utilization <50%
- Roofline shows compute-bound but actual performance is 50% of peak

**Solutions:**
1. Increase tile size: 16×16 → 32×32 → 64×64
2. Fuse operations: Multiple small kernels → single large kernel
3. Enable async memory: Hide memory latency with computation
4. Reduce divergence: Ensure warp-aligned memory access

### Issue: High Memory Latency Stalls

**Symptoms:**
- Cache miss rate >30%
- Memory stall cycles >50% of total cycles

**Solutions:**
1. Improve data locality: Process same data in nearby threads
2. Increase prefetch: Load data before needed
3. Use shared memory: Reduce global memory pressure
4. Enable memory coalescing: Sequential thread access

### Issue: Register Pressure (Occupancy Low)

**Symptoms:**
- Occupancy <50%
- Roofline suggests compute-bound but actual is memory-bound

**Solutions:**
1. Reduce spill: Optimize loop unrolling and temporaries
2. Use shared memory: Move data from registers to SRAM
3. Specialize kernels: Separate kernels for different workloads
4. Enable register optimization: `--use-fast-math` in NVCC

## Performance Reporting

### Generate Report

```rust
use sounio::codegen::gpu::generate_performance_report;

fn report(dispatch: &DispatchOverheadAnalysis,
          wmma: &[WmmaValidationResult],
          precision: &PrecisionComparison) -> String {
    generate_performance_report(dispatch, wmma, precision)
}
```

### Example Output

```markdown
# GPU Performance Validation Report

## Dispatch Overhead Analysis
- minimal_kernel: 250 cycles (0.00%)
- simple_kernel: 250 cycles (0.00%)
- complex_kernel: 250 cycles (0.00%)

## WMMA Utilization Validation
- wmma_test: 45.3% (target: 80.0%) - OPTIMIZATION NEEDED

## INT8 vs FP32 Accuracy
- matrix_multiply_512x512: 0.37% error - PASS
- convolution_3x3: 0.37% error - PASS
- Success rate: 100.00%
```

## Testing Workflow

1. **Write kernel** → PTX/MSL code generation
2. **Run profiler** → Collect metrics with perf/ncu
3. **Analyze roofline** → Identify bottleneck
4. **Optimize** → Reduce memory/improve compute
5. **Validate** → Re-profile to verify improvement
6. **Benchmark** → Track regression with CI

## Hardware Requirements

### Minimal Setup (CPU-side profiling)

- Linux: perf available (most distros)
- macOS: Instruments built-in
- Windows: Not yet supported

### GPU Profiling (recommended)

- **NVIDIA**: ncu + A100/H100 or better
- **AMD**: rocprof + RDNA2 or better
- **Apple**: Xcode + M1/M2 or better

### Data Requirements

- 100MB binary + profiler
- 1GB memory for profiler data collection
- NVMe for fast result export

## References

- **Roofline Model**: https://crd.lbl.gov/departments/computer-science/par/research/roofline/
- **NVIDIA Profiling**: https://docs.nvidia.com/nsight-compute/latest/
- **Kernel Performance Analysis**: https://arxiv.org/abs/1910.10193

## See Also

- [PERFORMANCE_VALIDATION_REPORT.md](../../PERFORMANCE_VALIDATION_REPORT.md)
- [profiler.rs](profiler.rs) - Static performance analysis
- [system_profiler.rs](system_profiler.rs) - Hardware profiling integration
- [compiler/tests/gpu_performance_validation.rs](../../tests/gpu_performance_validation.rs) - Validation tests
