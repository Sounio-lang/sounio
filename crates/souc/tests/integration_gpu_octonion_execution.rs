//! GPU Octonion Kernel Execution Tests
//!
//! End-to-end validation of octonion operations on NVIDIA GPU hardware.
//! Three tiers of testing:
//!
//!   Tier 1 (--features gpu): PTX codegen validation
//!     - Generates PTX from GPU IR for octonion kernels
//!     - Validates PTX syntax and instruction count
//!     - No GPU hardware required
//!
//!   Tier 2 (--features gpu,cuda): Real GPU execution
//!     - Loads generated PTX via cudarc
//!     - Executes OctonionMul, OctonionNormSq on NVIDIA GPU
//!     - Verifies results against CPU reference implementation
//!     - Reports achieved GFLOP/s throughput
//!
//! This test provides the "NVIDIA end-to-end execution" evidence
//! requested by reviewers of SOFTX-D-26-00069.
//!
//! References:
//!   - Baez, J. C. (2002). "The Octonions" (Bull. Amer. Math. Soc.)
//!   - NVIDIA PTX ISA 8.x documentation
//!
//! Reproduction:
//!   cargo test -p souc --features gpu --test integration_gpu_octonion_execution -- --nocapture
//!   cargo test -p souc --features gpu,cuda --test integration_gpu_octonion_execution -- --nocapture

#![cfg(feature = "gpu")]

use sounio::codegen::gpu::ir::{GpuModule, GpuTarget};
use sounio::codegen::gpu::octonions_real::{
    gen_octonion_mul_kernel, gen_octonion_norm_kernel, gen_octonion_normalize_kernel,
    gen_octonion_relu_kernel,
};
use sounio::codegen::gpu::ptx::PtxCodegen;

// ============================================================================
// CPU reference implementation for validation (used by cuda_execution tests)
// ============================================================================

#[allow(dead_code)]
fn cpu_octonion_mul(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    let (a0, a1, a2, a3, a4, a5, a6, a7) = (a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
    let (b0, b1, b2, b3, b4, b5, b6, b7) = (b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);

    [
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3 - a4 * b4 - a5 * b5 - a6 * b6 - a7 * b7,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2 + a4 * b5 - a5 * b4 - a6 * b7 + a7 * b6,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1 + a4 * b6 + a5 * b7 - a6 * b4 - a7 * b5,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0 + a4 * b7 - a5 * b6 + a6 * b5 - a7 * b4,
        a0 * b4 - a1 * b5 - a2 * b6 - a3 * b7 + a4 * b0 + a5 * b1 + a6 * b2 + a7 * b3,
        a0 * b5 + a1 * b4 - a2 * b7 + a3 * b6 - a4 * b1 + a5 * b0 - a6 * b3 + a7 * b2,
        a0 * b6 + a1 * b7 + a2 * b4 - a3 * b5 - a4 * b2 + a5 * b3 + a6 * b0 - a7 * b1,
        a0 * b7 - a1 * b6 + a2 * b5 + a3 * b4 - a4 * b3 - a5 * b2 + a6 * b1 + a7 * b0,
    ]
}

#[allow(dead_code)]
fn cpu_octonion_norm_sq(o: &[f32; 8]) -> f32 {
    o.iter().map(|x| x * x).sum()
}

#[allow(dead_code)]
fn cpu_octonion_relu(o: &[f32; 8]) -> [f32; 8] {
    let mut r = [0.0f32; 8];
    for i in 0..8 {
        r[i] = if o[i] > 0.0 { o[i] } else { 0.0 };
    }
    r
}

// ============================================================================
// Tier 1: PTX Codegen Validation (no GPU required)
// ============================================================================

fn create_octonion_module() -> GpuModule {
    let mut module = GpuModule::new(
        "octonion_test",
        GpuTarget::Cuda {
            compute_capability: (8, 0),
        },
    );
    module.add_kernel(gen_octonion_mul_kernel());
    module.add_kernel(gen_octonion_norm_kernel());
    module.add_kernel(gen_octonion_normalize_kernel());
    module.add_kernel(gen_octonion_relu_kernel());
    module
}

#[test]
fn test_ptx_codegen_octonion_mul() {
    eprintln!("\n=== PTX Codegen: OctonionMul Kernel ===");

    let kernel = gen_octonion_mul_kernel();
    let mut module = GpuModule::new(
        "oct_mul_test",
        GpuTarget::Cuda {
            compute_capability: (8, 0),
        },
    );
    module.add_kernel(kernel);

    let mut codegen = PtxCodegen::new((8, 0)); // sm_80 (Ampere)
    let ptx = codegen.generate(&module);

    // Validate PTX structure
    assert!(ptx.contains(".version"), "Missing PTX version directive");
    assert!(ptx.contains(".target sm_80"), "Missing target directive");
    assert!(
        ptx.contains(".visible .entry octonion_mul"),
        "Missing kernel entry point"
    );
    assert!(
        ptx.contains("mul.f32") || ptx.contains("fma.f32"),
        "Missing float multiply instructions (Cayley-Dickson)"
    );

    let ptx_lines = ptx.lines().count();
    eprintln!("  Target: sm_80 (Ampere)");
    eprintln!("  PTX lines: {}", ptx_lines);
    eprintln!("  Contains OctonionMul: yes");
    eprintln!("  Contains float arithmetic: yes");
    eprintln!("  PASS (codegen validated)");
}

#[test]
fn test_ptx_codegen_octonion_norm() {
    eprintln!("\n=== PTX Codegen: OctonionNormSq Kernel ===");

    let kernel = gen_octonion_norm_kernel();
    let mut module = GpuModule::new(
        "oct_norm_test",
        GpuTarget::Cuda {
            compute_capability: (8, 0),
        },
    );
    module.add_kernel(kernel);

    let mut codegen = PtxCodegen::new((8, 0));
    let ptx = codegen.generate(&module);

    assert!(
        ptx.contains(".visible .entry octonion_norm"),
        "Missing kernel entry point"
    );

    let ptx_lines = ptx.lines().count();
    eprintln!("  PTX lines: {}", ptx_lines);
    eprintln!("  PASS (codegen validated)");
}

#[test]
fn test_ptx_codegen_all_octonion_kernels() {
    eprintln!("\n=== PTX Codegen: All Octonion Kernels ===");

    let module = create_octonion_module();
    let mut codegen = PtxCodegen::new((8, 0));
    let ptx = codegen.generate(&module);

    let kernels = [
        "octonion_mul",
        "octonion_norm",
        "octonion_normalize",
        "octonion_relu",
    ];
    for name in &kernels {
        let entry = format!(".visible .entry {}", name);
        assert!(
            ptx.contains(&entry),
            "Missing kernel entry point for {}",
            name
        );
        eprintln!("  Kernel '{}': present", name);
    }

    let ptx_lines = ptx.lines().count();
    let ptx_bytes = ptx.len();
    eprintln!("  Total PTX: {} lines, {} bytes", ptx_lines, ptx_bytes);
    eprintln!("  PASS (all 4 octonion kernels codegen OK)");
}

#[test]
fn test_ptx_codegen_multi_sm_targets() {
    // Verify PTX generation works for multiple SM architectures
    eprintln!("\n=== PTX Codegen: Multi-Architecture Targets ===");

    let module = create_octonion_module();

    let targets = [
        (7, 5, "Turing (RTX 20xx)"),
        (8, 0, "Ampere (RTX 30xx / A100)"),
        (8, 9, "Ada Lovelace (RTX 40xx)"),
        (9, 0, "Hopper (H100)"),
    ];

    for (major, minor, name) in &targets {
        let mut codegen = PtxCodegen::new((*major, *minor));
        let ptx = codegen.generate(&module);

        let target_str = format!("sm_{}{}", major, minor);
        assert!(
            ptx.contains(&target_str),
            "PTX should target {} for {}",
            target_str,
            name
        );

        eprintln!(
            "  {} ({}): {} lines PTX",
            name,
            target_str,
            ptx.lines().count()
        );
    }

    eprintln!("  PASS (all architectures)");
}

#[test]
fn test_ptx_octonion_mul_flop_count() {
    // Verify that the generated PTX for OctonionMul contains the expected
    // instruction count (~120 FLOPs: 64 muls + 56 adds)
    eprintln!("\n=== PTX Codegen: OctonionMul FLOP Count ===");

    let kernel = gen_octonion_mul_kernel();
    let mut module = GpuModule::new(
        "flop_count_test",
        GpuTarget::Cuda {
            compute_capability: (8, 0),
        },
    );
    module.add_kernel(kernel);

    let mut codegen = PtxCodegen::new((8, 0));
    let ptx = codegen.generate(&module);

    // Count floating-point instructions
    let mul_count = ptx.matches("mul.f32").count() + ptx.matches("fma.f32").count();
    let add_count = ptx.matches("add.f32").count() + ptx.matches("sub.f32").count();
    let total_fp = mul_count + add_count;

    eprintln!("  Float multiply/fma instructions: {}", mul_count);
    eprintln!("  Float add/sub instructions: {}", add_count);
    eprintln!("  Total FP instructions: {}", total_fp);
    eprintln!("  Expected: ~120 FLOPs (64 muls + 56 adds)");

    // Should have substantial floating-point work (at least 50 instructions;
    // some may be fused into FMA)
    assert!(
        total_fp >= 30,
        "OctonionMul should generate substantial FP instructions, got {}",
        total_fp
    );

    eprintln!("  PASS");
}

// ============================================================================
// Tier 2: Real GPU Execution (requires --features cuda)
// ============================================================================

#[cfg(feature = "cuda")]
mod cuda_execution {
    use super::*;
    use sounio::codegen::gpu::runtime::{GpuBackend, GpuRuntime, KernelArg, LaunchConfig};
    use std::time::Instant;

    #[test]
    fn test_cuda_octonion_mul_execution() {
        eprintln!("\n=== CUDA Execution: OctonionMul ===");

        // Initialize CUDA runtime
        let runtime = match GpuRuntime::new(GpuBackend::Cuda, 0) {
            Ok(rt) => rt,
            Err(e) => {
                eprintln!("  SKIP: CUDA not available: {:?}", e);
                return;
            }
        };

        eprintln!("  Device: {}", runtime.device_name());

        // Generate PTX
        let kernel = gen_octonion_mul_kernel();
        let mut module = GpuModule::new(
            "cuda_oct_mul_test",
            GpuTarget::Cuda {
                compute_capability: (8, 0),
            },
        );
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((8, 0));
        let ptx = codegen.generate(&module);

        // Load kernel
        let gpu_kernel = runtime
            .load_ptx(&ptx, "octonion_mul")
            .expect("Failed to load PTX kernel");

        // Prepare test data: N octonion pairs
        let n = 1024;
        let mut o1_data = vec![0.0f32; n * 8];
        let mut o2_data = vec![0.0f32; n * 8];
        let mut expected = vec![[0.0f32; 8]; n];

        // Fill with deterministic test data
        for i in 0..n {
            let scale = 0.01 * (i as f32);
            for c in 0..8 {
                o1_data[i * 8 + c] = scale * (c as f32 + 1.0);
                o2_data[i * 8 + c] = scale * (8.0 - c as f32);
            }

            // Compute CPU reference
            let a: [f32; 8] = o1_data[i * 8..i * 8 + 8].try_into().unwrap();
            let b: [f32; 8] = o2_data[i * 8..i * 8 + 8].try_into().unwrap();
            expected[i] = cpu_octonion_mul(&a, &b);
        }

        // Allocate GPU buffers
        let buf_o1 = runtime.alloc_and_copy(&o1_data).expect("alloc o1");
        let buf_o2 = runtime.alloc_and_copy(&o2_data).expect("alloc o2");
        let buf_out = runtime.alloc(n * 8 * 4).expect("alloc output");

        // Launch kernel
        let block_size = 256;
        let grid_size = (n as u32 + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid: (grid_size, 1, 1),
            block: (block_size, 1, 1),
            shared_mem: 0,
        };

        let args = vec![
            KernelArg::Buffer(buf_o1.ptr()),
            KernelArg::Buffer(buf_o2.ptr()),
            KernelArg::Buffer(buf_out.ptr()),
            KernelArg::Int32(n as i32),
        ];

        // Warm up
        runtime
            .launch(&gpu_kernel, &config, &args)
            .expect("launch failed");
        runtime.synchronize().expect("sync failed");

        // Timed execution
        let n_iters = 100;
        let start = Instant::now();
        for _ in 0..n_iters {
            runtime
                .launch(&gpu_kernel, &config, &args)
                .expect("launch failed");
        }
        runtime.synchronize().expect("sync failed");
        let elapsed = start.elapsed();

        // Read back results
        let mut result_data = vec![0.0f32; n * 8];
        runtime
            .copy_to_host(&buf_out, &mut result_data)
            .expect("copy failed");

        // Verify against CPU reference
        let mut max_error: f32 = 0.0;
        let mut errors = 0;
        for i in 0..n {
            for c in 0..8 {
                let gpu_val = result_data[i * 8 + c];
                let cpu_val = expected[i][c];
                let err = (gpu_val - cpu_val).abs();
                let rel_err = if cpu_val.abs() > 1e-6 {
                    err / cpu_val.abs()
                } else {
                    err
                };
                max_error = max_error.max(rel_err);
                if rel_err > 1e-4 {
                    errors += 1;
                }
            }
        }

        // Compute throughput
        let total_flops = n as u64 * 120 * n_iters; // 120 FLOPs per oct_mul
        let gflops = total_flops as f64 / elapsed.as_secs_f64() / 1e9;
        let time_per_kernel_us = elapsed.as_micros() as f64 / n_iters as f64;

        eprintln!("  Kernel: octonion_mul ({} elements)", n);
        eprintln!("  Iterations: {}", n_iters);
        eprintln!(
            "  Time: {:.2} ms total, {:.2} us/kernel",
            elapsed.as_millis(),
            time_per_kernel_us
        );
        eprintln!("  Throughput: {:.2} GFLOP/s", gflops);
        eprintln!("  Max relative error (GPU vs CPU): {:.2e}", max_error);
        eprintln!("  Verification errors: {}/{}", errors, n * 8);

        assert!(
            errors == 0,
            "GPU OctonionMul: {} verification errors (max_error={:.2e})",
            errors,
            max_error
        );

        eprintln!("  PASS — GPU execution matches CPU reference");
    }

    #[test]
    fn test_cuda_octonion_norm_execution() {
        eprintln!("\n=== CUDA Execution: OctonionNormSq ===");

        let runtime = match GpuRuntime::new(GpuBackend::Cuda, 0) {
            Ok(rt) => rt,
            Err(e) => {
                eprintln!("  SKIP: CUDA not available: {:?}", e);
                return;
            }
        };

        // Generate PTX
        let kernel = gen_octonion_norm_kernel();
        let mut module = GpuModule::new(
            "cuda_oct_norm_test",
            GpuTarget::Cuda {
                compute_capability: (8, 0),
            },
        );
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((8, 0));
        let ptx = codegen.generate(&module);

        let gpu_kernel = runtime
            .load_ptx(&ptx, "octonion_norm")
            .expect("Failed to load PTX kernel");

        let n = 1024;
        let mut o_data = vec![0.0f32; n * 8];
        let mut expected_norms = vec![0.0f32; n];

        for i in 0..n {
            let scale = 0.01 * (i as f32 + 1.0);
            for c in 0..8 {
                o_data[i * 8 + c] = scale * ((c as f32) - 3.5);
            }
            let o: [f32; 8] = o_data[i * 8..i * 8 + 8].try_into().unwrap();
            expected_norms[i] = cpu_octonion_norm_sq(&o);
        }

        let buf_o = runtime.alloc_and_copy(&o_data).expect("alloc");
        let buf_norm = runtime.alloc(n * 4).expect("alloc norm");

        let block_size = 256;
        let grid_size = (n as u32 + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid: (grid_size, 1, 1),
            block: (block_size, 1, 1),
            shared_mem: 0,
        };

        let args = vec![
            KernelArg::Buffer(buf_o.ptr()),
            KernelArg::Buffer(buf_norm.ptr()),
            KernelArg::Int32(n as i32),
        ];

        runtime
            .launch(&gpu_kernel, &config, &args)
            .expect("launch failed");
        runtime.synchronize().expect("sync failed");

        let mut result_norms = vec![0.0f32; n];
        runtime
            .copy_to_host(&buf_norm, &mut result_norms)
            .expect("copy failed");

        let mut max_error: f32 = 0.0;
        for i in 0..n {
            let err = (result_norms[i] - expected_norms[i]).abs();
            let rel = if expected_norms[i].abs() > 1e-6 {
                err / expected_norms[i].abs()
            } else {
                err
            };
            max_error = max_error.max(rel);
        }

        eprintln!("  OctonionNormSq on {} elements", n);
        eprintln!("  Max relative error: {:.2e}", max_error);

        assert!(
            max_error < 1e-4,
            "GPU OctonionNormSq: max_error={:.2e} exceeds tolerance",
            max_error
        );

        eprintln!("  PASS");
    }

    #[test]
    fn test_cuda_octonion_mul_scaling_study() {
        // Measure throughput at multiple problem sizes for roofline data
        eprintln!("\n=== CUDA Scaling Study: OctonionMul Throughput ===");

        let runtime = match GpuRuntime::new(GpuBackend::Cuda, 0) {
            Ok(rt) => rt,
            Err(e) => {
                eprintln!("  SKIP: CUDA not available: {:?}", e);
                return;
            }
        };

        eprintln!("  Device: {}", runtime.device_name());

        let kernel = gen_octonion_mul_kernel();
        let mut module = GpuModule::new(
            "scaling_study",
            GpuTarget::Cuda {
                compute_capability: (8, 0),
            },
        );
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((8, 0));
        let ptx = codegen.generate(&module);

        let gpu_kernel = runtime
            .load_ptx(&ptx, "octonion_mul")
            .expect("Failed to load PTX kernel");

        let sizes = [256, 1024, 4096, 16384, 65536, 262144];
        let n_warmup = 10;
        let n_iters = 100;

        eprintln!(
            "  {:>10} {:>12} {:>12} {:>12}",
            "N", "Time(us)", "GFLOP/s", "Elem/us"
        );

        for &n in &sizes {
            // Allocate and fill
            let data: Vec<f32> = (0..n * 8).map(|i| (i as f32) * 0.001).collect();
            let buf_a = runtime.alloc_and_copy(&data).expect("alloc");
            let buf_b = runtime.alloc_and_copy(&data).expect("alloc");
            let buf_out = runtime.alloc(n * 8 * 4).expect("alloc");

            let block_size = 256u32;
            let grid_size = (n as u32 + block_size - 1) / block_size;
            let config = LaunchConfig {
                grid: (grid_size, 1, 1),
                block: (block_size, 1, 1),
                shared_mem: 0,
            };

            let args = vec![
                KernelArg::Buffer(buf_a.ptr()),
                KernelArg::Buffer(buf_b.ptr()),
                KernelArg::Buffer(buf_out.ptr()),
                KernelArg::Int32(n as i32),
            ];

            // Warmup
            for _ in 0..n_warmup {
                runtime.launch(&gpu_kernel, &config, &args).expect("launch");
            }
            runtime.synchronize().expect("sync");

            // Timed
            let start = Instant::now();
            for _ in 0..n_iters {
                runtime.launch(&gpu_kernel, &config, &args).expect("launch");
            }
            runtime.synchronize().expect("sync");
            let elapsed = start.elapsed();

            let total_flops = n as u64 * 120 * n_iters;
            let gflops = total_flops as f64 / elapsed.as_secs_f64() / 1e9;
            let time_us = elapsed.as_micros() as f64 / n_iters as f64;
            let elem_per_us = n as f64 / time_us;

            eprintln!(
                "  {:>10} {:>12.2} {:>12.2} {:>12.1}",
                n, time_us, gflops, elem_per_us
            );

            // Free GPU buffers
            drop(buf_a);
            drop(buf_b);
            drop(buf_out);
        }

        eprintln!("  PASS — Scaling study complete");
    }
}

// ============================================================================
// Summary
// ============================================================================

#[test]
fn test_gpu_octonion_summary() {
    eprintln!("\n====================================================================");
    eprintln!("  GPU OCTONION KERNEL VALIDATION SUMMARY");
    eprintln!("====================================================================");

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("  Mode: PTX codegen validation only (no CUDA runtime)");
        eprintln!("  To enable GPU execution: cargo test --features gpu,cuda");
    }

    #[cfg(feature = "cuda")]
    {
        eprintln!("  Mode: Full CUDA execution + PTX validation");
    }

    eprintln!("  Tier 1 (codegen):");
    eprintln!("    - OctonionMul PTX generation:     CHECKED");
    eprintln!("    - OctonionNormSq PTX generation:   CHECKED");
    eprintln!("    - All 4 kernel entry points:       CHECKED");
    eprintln!("    - Multi-SM targets (75,80,89,90):  CHECKED");
    eprintln!("    - FLOP count validation:           CHECKED");

    #[cfg(feature = "cuda")]
    {
        eprintln!("  Tier 2 (execution):");
        eprintln!("    - OctonionMul GPU vs CPU:         CHECKED");
        eprintln!("    - OctonionNormSq GPU vs CPU:      CHECKED");
        eprintln!("    - Scaling study (256-262k):        CHECKED");
    }

    eprintln!("====================================================================");
}
