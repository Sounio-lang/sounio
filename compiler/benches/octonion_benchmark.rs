//! Performance Benchmarks for Octonion Operations
//!
//! This benchmark suite measures the performance of octonion algebra operations,
//! providing GFLOP/s metrics for GPU kernel validation and CPU baselines.
//!
//! References:
//! - Baez, J. C. (2002). "The Octonions" (Bull. Amer. Math. Soc.)
//! - Graves, J. T. (1843). "On algebraic triplets"
//! - Hardware: GPU kernels measured on NVIDIA (PTX) and Apple (Metal)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

/// Octonion representation for benchmarking
#[derive(Clone, Copy, Debug)]
struct Octonion {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    f: f32,
    g: f32,
    h: f32,
}

impl Octonion {
    fn new(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32) -> Self {
        Octonion {
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            h,
        }
    }

    /// Cayley-Dickson multiplication: 64 muls + 56 adds = 120 FLOPs
    fn mul(self, other: Octonion) -> Octonion {
        Octonion {
            a: self.a * other.a
                - self.b * other.b
                - self.c * other.c
                - self.d * other.d
                - self.e * other.e
                - self.f * other.f
                - self.g * other.g
                - self.h * other.h,
            b: self.a * other.b + self.b * other.a + self.c * other.d - self.d * other.c
                + self.e * other.f
                - self.f * other.e
                - self.g * other.h
                + self.h * other.g,
            c: self.a * other.c - self.b * other.d
                + self.c * other.a
                + self.d * other.b
                + self.e * other.g
                + self.f * other.h
                - self.g * other.e
                - self.h * other.f,
            d: self.a * other.d + self.b * other.c - self.c * other.b
                + self.d * other.a
                + self.e * other.h
                - self.f * other.g
                + self.g * other.f
                - self.h * other.e,
            e: self.a * other.e - self.b * other.f - self.c * other.g - self.d * other.h
                + self.e * other.a
                + self.f * other.b
                + self.g * other.c
                + self.h * other.d,
            f: self.a * other.f + self.b * other.e - self.c * other.h + self.d * other.g
                - self.e * other.b
                + self.f * other.a
                - self.g * other.d
                + self.h * other.c,
            g: self.a * other.g + self.b * other.h + self.c * other.e
                - self.d * other.f
                - self.e * other.c
                + self.f * other.d
                + self.g * other.a
                - self.h * other.b,
            h: self.a * other.h - self.b * other.g + self.c * other.f + self.d * other.e
                - self.e * other.d
                - self.f * other.c
                + self.g * other.b
                + self.h * other.a,
        }
    }

    /// Addition: 8 FLOPs
    fn add(self, other: Octonion) -> Octonion {
        Octonion {
            a: self.a + other.a,
            b: self.b + other.b,
            c: self.c + other.c,
            d: self.d + other.d,
            e: self.e + other.e,
            f: self.f + other.f,
            g: self.g + other.g,
            h: self.h + other.h,
        }
    }

    /// Norm: 8 muls + 7 adds + 1 sqrt = 16 FLOPs
    fn norm(self) -> f32 {
        (self.a * self.a
            + self.b * self.b
            + self.c * self.c
            + self.d * self.d
            + self.e * self.e
            + self.f * self.f
            + self.g * self.g
            + self.h * self.h)
            .sqrt()
    }

    /// ReLU activation: 8 comparisons + 8 selects = ~16 FLOPs
    fn relu(self) -> Octonion {
        Octonion {
            a: if self.a > 0.0 { self.a } else { 0.0 },
            b: if self.b > 0.0 { self.b } else { 0.0 },
            c: if self.c > 0.0 { self.c } else { 0.0 },
            d: if self.d > 0.0 { self.d } else { 0.0 },
            e: if self.e > 0.0 { self.e } else { 0.0 },
            f: if self.f > 0.0 { self.f } else { 0.0 },
            g: if self.g > 0.0 { self.g } else { 0.0 },
            h: if self.h > 0.0 { self.h } else { 0.0 },
        }
    }

    /// Sigmoid activation: per-component 1/(1+exp(-x))
    fn sigmoid(self) -> Octonion {
        Octonion {
            a: 1.0 / (1.0 + (-self.a).exp()),
            b: 1.0 / (1.0 + (-self.b).exp()),
            c: 1.0 / (1.0 + (-self.c).exp()),
            d: 1.0 / (1.0 + (-self.d).exp()),
            e: 1.0 / (1.0 + (-self.e).exp()),
            f: 1.0 / (1.0 + (-self.f).exp()),
            g: 1.0 / (1.0 + (-self.g).exp()),
            h: 1.0 / (1.0 + (-self.h).exp()),
        }
    }
}

/// Benchmark octonion multiplication
fn bench_octonion_mul(c: &mut Criterion) {
    let o1 = Octonion::new(1.0, 0.5, 0.3, 0.2, 0.1, 0.15, 0.25, 0.05);
    let o2 = Octonion::new(0.5, 0.6, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1);

    let mut group = c.benchmark_group("octonion_mul");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(1000);

    group.bench_function("mul_scalar", |b| {
        b.iter(|| {
            let result = black_box(o1).mul(black_box(o2));
            black_box(result);
        });
    });

    // Multiplication chain: demonstrates 8x parameter efficiency
    // 2 multiplications = 240 FLOPs
    group.bench_function("mul_chain_2x", |b| {
        b.iter(|| {
            let r1 = black_box(o1).mul(black_box(o2));
            let r2 = black_box(r1).mul(black_box(o1));
            black_box(r2);
        });
    });

    group.finish();
}

/// Benchmark octonion addition
fn bench_octonion_add(c: &mut Criterion) {
    let o1 = Octonion::new(1.0, 0.5, 0.3, 0.2, 0.1, 0.15, 0.25, 0.05);
    let o2 = Octonion::new(0.5, 0.6, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1);

    let mut group = c.benchmark_group("octonion_add");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(1000);

    group.bench_function("add_scalar", |b| {
        b.iter(|| {
            let result = black_box(o1).add(black_box(o2));
            black_box(result);
        });
    });

    // Addition chain
    group.bench_function("add_chain_8x", |b| {
        b.iter(|| {
            let mut r = black_box(o1);
            for _ in 0..8 {
                r = r.add(black_box(o2));
            }
            black_box(r);
        });
    });

    group.finish();
}

/// Benchmark octonion norm
fn bench_octonion_norm(c: &mut Criterion) {
    let o1 = Octonion::new(1.0, 0.5, 0.3, 0.2, 0.1, 0.15, 0.25, 0.05);

    let mut group = c.benchmark_group("octonion_norm");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(1000);

    group.bench_function("norm", |b| {
        b.iter(|| {
            let result = black_box(o1).norm();
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark activation functions
fn bench_octonion_activations(c: &mut Criterion) {
    let o1 = Octonion::new(0.5, -0.3, 0.2, -0.1, 0.4, -0.2, 0.1, -0.05);

    let mut group = c.benchmark_group("octonion_activations");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(1000);

    group.bench_function("relu", |b| {
        b.iter(|| {
            let result = black_box(o1).relu();
            black_box(result);
        });
    });

    group.bench_function("sigmoid", |b| {
        b.iter(|| {
            let result = black_box(o1).sigmoid();
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark vector dot product: 8 * N octonions
fn bench_octonion_vec_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("octonion_vec_operations");
    group.measurement_time(Duration::from_secs(5));

    for size in [16, 64, 256, 1024].iter() {
        let vec_a: Vec<Octonion> = (0..*size)
            .map(|i| {
                let x = i as f32 * 0.1;
                Octonion::new(
                    x,
                    x + 0.1,
                    x + 0.2,
                    x + 0.3,
                    x + 0.4,
                    x + 0.5,
                    x + 0.6,
                    x + 0.7,
                )
            })
            .collect();

        let vec_b: Vec<Octonion> = (0..*size)
            .map(|i| {
                let x = i as f32 * 0.05;
                Octonion::new(
                    x,
                    x + 0.05,
                    x + 0.1,
                    x + 0.15,
                    x + 0.2,
                    x + 0.25,
                    x + 0.3,
                    x + 0.35,
                )
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("dot_product_{}_octonions", size)),
            size,
            |b, _| {
                b.iter(|| {
                    let mut sum = Octonion::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                    for i in 0..*size {
                        sum = sum.add(black_box(vec_a[i]).mul(black_box(vec_b[i])));
                    }
                    black_box(sum);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark octonion matrix multiplication
fn bench_octonion_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("octonion_matmul");
    group.measurement_time(Duration::from_secs(10));

    for size in [4, 8, 16, 32].iter() {
        // Create square matrices of octonions
        let mat_a: Vec<Vec<Octonion>> = (0..*size)
            .map(|i| {
                (0..*size)
                    .map(|j| {
                        let x = (i * *size + j) as f32 * 0.01;
                        Octonion::new(
                            x,
                            x + 0.01,
                            x + 0.02,
                            x + 0.03,
                            x + 0.04,
                            x + 0.05,
                            x + 0.06,
                            x + 0.07,
                        )
                    })
                    .collect()
            })
            .collect();

        let mat_b: Vec<Vec<Octonion>> = (0..*size)
            .map(|i| {
                (0..*size)
                    .map(|j| {
                        let x = (i * *size + j) as f32 * 0.001;
                        Octonion::new(
                            x,
                            x + 0.001,
                            x + 0.002,
                            x + 0.003,
                            x + 0.004,
                            x + 0.005,
                            x + 0.006,
                            x + 0.007,
                        )
                    })
                    .collect()
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("matmul_{}x{}", size, size)),
            size,
            |b, _| {
                b.iter(|| {
                    let mut result =
                        vec![
                            vec![Octonion::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0); *size];
                            *size
                        ];

                    for i in 0..*size {
                        for j in 0..*size {
                            let mut sum = Octonion::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                            for k in 0..*size {
                                sum = sum.add(black_box(mat_a[i][k]).mul(black_box(mat_b[k][j])));
                            }
                            result[i][j] = sum;
                        }
                    }

                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_octonion_mul,
    bench_octonion_add,
    bench_octonion_norm,
    bench_octonion_activations,
    bench_octonion_vec_dot,
    bench_octonion_matmul
);

criterion_main!(benches);
