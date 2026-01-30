//! GPU Optimization Benchmarks - Phase 8 Performance Validation
//!
//! Benchmarks for:
//! 1. Occupancy Calculation - Architecture-specific occupancy
//! 2. PTX String Generation - Basic PTX emission
//! 3. Architecture Queries - CudaArch method performance

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use sounio::codegen::gpu::autotune::OccupancyCalculator;
use sounio::codegen::gpu::ir::CudaArch;
use sounio::codegen::gpu::ptx::PtxCodegen;

/// Benchmark occupancy calculation across architectures
fn bench_occupancy_calc(c: &mut Criterion) {
    let mut group = c.benchmark_group("occupancy");
    group.sample_size(200);

    let archs = [
        ("turing", CudaArch::Turing),
        ("ampere", CudaArch::Ampere),
        ("hopper", CudaArch::Hopper),
        ("blackwell", CudaArch::Blackwell),
    ];

    for (name, arch) in archs {
        group.bench_with_input(BenchmarkId::new("arch", name), &arch, |b, &arch| {
            let calc = OccupancyCalculator::from_cuda_arch(arch);
            b.iter(|| {
                // Calculate occupancy for various configurations
                let mut total = 0.0f64;
                for block_size in [64, 128, 256, 512, 1024] {
                    for registers in [16, 32, 48, 64] {
                        for shared_mem in [0, 4096, 16384, 32768] {
                            let info = calc.calculate_occupancy(block_size, registers, shared_mem);
                            total += info.occupancy;
                        }
                    }
                }
                black_box(total)
            });
        });
    }
    group.finish();
}

/// Benchmark PTX codegen creation and header emission
fn bench_ptx_codegen(c: &mut Criterion) {
    let mut group = c.benchmark_group("ptx_codegen");
    group.sample_size(100);

    let sm_versions = [
        ("sm_75", (7, 5)),
        ("sm_80", (8, 0)),
        ("sm_90", (9, 0)),
        ("sm_100", (10, 0)),
    ];

    for (name, sm_version) in sm_versions {
        group.bench_function(format!("{}_create", name), |b| {
            b.iter(|| {
                let codegen = PtxCodegen::new(sm_version);
                black_box(codegen)
            });
        });
    }
    group.finish();
}

/// Benchmark architecture-specific queries
fn bench_arch_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("arch_queries");
    group.sample_size(500);

    let archs = [
        ("turing", CudaArch::Turing),
        ("ampere", CudaArch::Ampere),
        ("ada", CudaArch::Ada),
        ("hopper", CudaArch::Hopper),
        ("blackwell", CudaArch::Blackwell),
    ];

    for (name, arch) in archs {
        group.bench_function(format!("{}_compute_cap", name), |b| {
            b.iter(|| black_box(arch.compute_capability()));
        });
    }
    group.finish();
}

/// Benchmark occupancy calculator creation
fn bench_occupancy_create(c: &mut Criterion) {
    let mut group = c.benchmark_group("occupancy_create");
    group.sample_size(200);

    let archs = [
        ("turing", CudaArch::Turing),
        ("ampere", CudaArch::Ampere),
        ("hopper", CudaArch::Hopper),
        ("blackwell", CudaArch::Blackwell),
    ];

    for (name, arch) in archs {
        group.bench_function(name, |b| {
            b.iter(|| {
                let calc = OccupancyCalculator::from_cuda_arch(arch);
                black_box(calc)
            });
        });
    }
    group.finish();
}

// Define benchmark groups
criterion_group!(
    occupancy_benches,
    bench_occupancy_calc,
    bench_occupancy_create,
);

criterion_group!(ptx_benches, bench_ptx_codegen,);

criterion_group!(arch_benches, bench_arch_queries,);

// Main benchmark runner
criterion_main!(occupancy_benches, ptx_benches, arch_benches);
