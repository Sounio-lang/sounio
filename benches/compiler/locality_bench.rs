//! Locality Module Benchmarks
//!
//! Benchmarks for the locality analysis system including:
//! - Access pattern analysis
//! - Cache optimization recommendations
//! - HIR traversal performance

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use sounio::locality::{
    access::{AccessAnalyzer, AccessKind, AccessPattern, CoAccess, FieldAccess},
    packing::CacheLinePacker,
    types::Locality,
};

/// Benchmark field access recording
fn benchmark_access_recording(c: &mut Criterion) {
    let mut group = c.benchmark_group("access_recording");

    for num_accesses in [10, 100, 1000] {
        group.bench_with_input(
            BenchmarkId::new("field_access", num_accesses),
            &num_accesses,
            |b, &n| {
                b.iter(|| {
                    let mut analyzer = AccessAnalyzer::new();
                    analyzer.enter_function("test_fn");

                    for i in 0..n {
                        let field = format!("field_{}", i % 10);
                        analyzer.record_access("TestType", &field, AccessKind::Read);
                    }

                    analyzer.exit_function();
                    // Return whether the pattern exists rather than a reference
                    black_box(analyzer.get_pattern("test_fn").is_some())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark co-access detection
fn benchmark_co_access_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("co_access_detection");

    for window_size in [5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::new("co_access_window", window_size),
            &window_size,
            |b, &_ws| {
                b.iter(|| {
                    let mut analyzer = AccessAnalyzer::new();
                    analyzer.enter_function("co_access_fn");

                    // Simulate typical access pattern with co-accesses
                    for _ in 0..50 {
                        analyzer.record_access("Patient", "name", AccessKind::Read);
                        analyzer.record_access("Patient", "age", AccessKind::Read);
                        analyzer.record_access("Patient", "id", AccessKind::Read);
                    }

                    analyzer.exit_function();
                    // Return the number of recommendations instead of a reference
                    black_box(analyzer.recommendations().len())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark stride pattern detection
fn benchmark_stride_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("stride_detection");

    for stride in [1, 4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::new("stride_pattern", stride),
            &stride,
            |b, &s| {
                b.iter(|| {
                    let mut analyzer = AccessAnalyzer::new();
                    analyzer.enter_function("stride_fn");
                    analyzer.enter_loop();

                    for _ in 0..100 {
                        analyzer.record_stride("Array", "data", s);
                    }

                    analyzer.exit_loop();
                    analyzer.exit_function();
                    // Return the number of patterns
                    black_box(analyzer.all_patterns().count())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark access pattern analysis
fn benchmark_pattern_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_analysis");

    group.bench_function("get_hot_fields", |b| {
        let mut pattern = AccessPattern::new("benchmark_pattern");

        // Add many accesses with varying heat levels
        for i in 0..100 {
            let mut access = FieldAccess::new("Type", format!("field_{}", i % 20));
            if i % 5 == 0 {
                access = access.in_loop(2);
            } else if i % 3 == 0 {
                access = access.in_loop(1);
            }
            pattern.add_access(access);
        }

        b.iter(|| black_box(pattern.get_hot_fields().len()));
    });

    group.bench_function("get_co_access_groups", |b| {
        let mut pattern = AccessPattern::new("benchmark_pattern");

        // Add many co-access relationships
        for i in 0..50 {
            pattern.add_co_access(CoAccess::new(
                format!("field_{}", i),
                format!("field_{}", (i + 1) % 50),
                0.8,
            ));
        }

        b.iter(|| black_box(pattern.get_co_access_groups().len()));
    });

    group.finish();
}

/// Benchmark recommendation generation
fn benchmark_recommendations(c: &mut Criterion) {
    let mut group = c.benchmark_group("recommendations");

    for num_patterns in [5, 20, 50] {
        group.bench_with_input(
            BenchmarkId::new("generate_recommendations", num_patterns),
            &num_patterns,
            |b, &n| {
                b.iter_batched(
                    || {
                        let mut analyzer = AccessAnalyzer::new();

                        for p in 0..n {
                            let func_name = format!("func_{}", p);
                            analyzer.enter_function(&func_name);
                            analyzer.enter_loop();

                            for i in 0..20 {
                                analyzer.record_access(
                                    &format!("Type_{}", p),
                                    &format!("field_{}", i % 5),
                                    AccessKind::Read,
                                );
                            }

                            analyzer.record_stride(&format!("Type_{}", p), "array", 8);

                            analyzer.exit_loop();
                            analyzer.exit_function();
                        }

                        analyzer
                    },
                    |analyzer| black_box(analyzer.recommendations().len()),
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

/// Benchmark memory layout operations
fn benchmark_memory_layout(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_layout");

    group.bench_function("cache_line_packer_creation", |b| {
        b.iter(|| {
            let packer = CacheLinePacker::new(64);
            black_box(packer)
        });
    });

    group.bench_function("locality_comparisons", |b| {
        let localities = [
            Locality::L1,
            Locality::L2,
            Locality::L3,
            Locality::Local,
            Locality::Remote,
        ];

        b.iter(|| {
            let mut equal_count = 0;
            for l1 in &localities {
                for l2 in &localities {
                    if l1 == l2 {
                        equal_count += 1;
                    }
                }
            }
            black_box(equal_count)
        });
    });

    group.finish();
}

/// Benchmark loop depth impact on heat calculation
fn benchmark_heat_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("heat_calculation");

    for loop_depth in [0, 1, 2, 3, 4] {
        group.bench_with_input(
            BenchmarkId::new("heat_at_depth", loop_depth),
            &loop_depth,
            |b, &depth| {
                b.iter(|| {
                    let access = FieldAccess::new("Type", "field").in_loop(depth);
                    black_box(access.heat())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark co_access_for method
fn benchmark_co_access_for(c: &mut Criterion) {
    let mut group = c.benchmark_group("co_access_for");

    for num_types in [5, 20, 50] {
        group.bench_with_input(
            BenchmarkId::new("co_access_lookup", num_types),
            &num_types,
            |b, &n| {
                b.iter_batched(
                    || {
                        let mut analyzer = AccessAnalyzer::new();

                        for t in 0..n {
                            let func_name = format!("func_{}", t);
                            analyzer.enter_function(&func_name);

                            for i in 0..10 {
                                analyzer.record_access(
                                    &format!("Type_{}", t),
                                    &format!("field_{}", i),
                                    AccessKind::Read,
                                );
                            }

                            analyzer.exit_function();
                        }

                        analyzer
                    },
                    |analyzer| {
                        let mut total = 0;
                        for t in 0..n {
                            total += analyzer.co_access_for(&format!("Type_{}", t)).len();
                        }
                        black_box(total)
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_access_recording,
    benchmark_co_access_detection,
    benchmark_stride_detection,
    benchmark_pattern_analysis,
    benchmark_recommendations,
    benchmark_memory_layout,
    benchmark_heat_calculation,
    benchmark_co_access_for,
);

criterion_main!(benches);
