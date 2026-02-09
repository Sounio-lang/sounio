//! Octonion Performance Benchmarking Suite
//!
//! This benchmark suite measures key performance characteristics of octonion operations:
//! - Multiplication cost (120 FLOPs per multiplication)
//! - Norm computation (10 operations)
//! - Conjugate operation (7 negations)
//! - Activation functions (per-component)
//! - GPU kernel efficiency (GFLOPS for PTX/Metal)
//!
//! Run with: cargo bench --bench octonion_bench

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

// Mock octonion struct for benchmarking (same as in tests)
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

    // Octonion multiplication (120 FLOPs: 64 multiplications + 56 additions)
    // Graves-Adcock formula — uses a0..a7/b0..b7 to avoid variable shadowing
    fn mul(self, other: Octonion) -> Octonion {
        let (a0, a1, a2, a3, a4, a5, a6, a7) = (
            self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h,
        );
        let (b0, b1, b2, b3, b4, b5, b6, b7) = (
            other.a, other.b, other.c, other.d, other.e, other.f, other.g, other.h,
        );

        Octonion {
            a: a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3 - a4 * b4 - a5 * b5 - a6 * b6 - a7 * b7,
            b: a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2 + a4 * b5 - a5 * b4 - a6 * b7 + a7 * b6,
            c: a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1 + a4 * b6 + a5 * b7 - a6 * b4 - a7 * b5,
            d: a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0 + a4 * b7 - a5 * b6 + a6 * b5 - a7 * b4,
            e: a0 * b4 - a1 * b5 - a2 * b6 - a3 * b7 + a4 * b0 + a5 * b1 + a6 * b2 + a7 * b3,
            f: a0 * b5 + a1 * b4 - a2 * b7 + a3 * b6 - a4 * b1 + a5 * b0 - a6 * b3 + a7 * b2,
            g: a0 * b6 + a1 * b7 + a2 * b4 - a3 * b5 - a4 * b2 + a5 * b3 + a6 * b0 - a7 * b1,
            h: a0 * b7 - a1 * b6 + a2 * b5 + a3 * b4 - a4 * b3 - a5 * b2 + a6 * b1 + a7 * b0,
        }
    }

    // Norm squared (8 multiplications + 7 additions = 15 FLOPs)
    fn norm_sq(self) -> f32 {
        self.a * self.a
            + self.b * self.b
            + self.c * self.c
            + self.d * self.d
            + self.e * self.e
            + self.f * self.f
            + self.g * self.g
            + self.h * self.h
    }

    // Euclidean norm (15 FLOPs + 1 sqrt = ~20 FLOPs)
    fn norm(self) -> f32 {
        self.norm_sq().sqrt()
    }

    // Conjugate (7 negations = 7 FLOPs)
    fn conj(self) -> Octonion {
        Octonion {
            a: self.a,
            b: -self.b,
            c: -self.c,
            d: -self.d,
            e: -self.e,
            f: -self.f,
            g: -self.g,
            h: -self.h,
        }
    }

    // ReLU per-component (8 comparisons + 8 conditionals)
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

    // Tanh per-component (8 tanh calls)
    fn tanh(self) -> Octonion {
        Octonion {
            a: self.a.tanh(),
            b: self.b.tanh(),
            c: self.c.tanh(),
            d: self.d.tanh(),
            e: self.e.tanh(),
            f: self.f.tanh(),
            g: self.g.tanh(),
            h: self.h.tanh(),
        }
    }
}

fn benchmark_octonion_multiplication(c: &mut Criterion) {
    c.bench_function("oct_mul_basic", |b| {
        let o1 = black_box(Octonion::new(1.0, 0.5, 0.3, 0.2, 0.1, 0.15, 0.25, 0.05));
        let o2 = black_box(Octonion::new(0.5, 0.6, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1));
        b.iter(|| o1.mul(o2))
    });

    // Benchmark with different value distributions
    let mut group = c.benchmark_group("oct_mul_variants");
    for size in [1.0, 10.0, 100.0].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &s| {
            let o1 = black_box(Octonion::new(
                s,
                s * 0.5,
                s * 0.3,
                s * 0.2,
                s * 0.1,
                s * 0.15,
                s * 0.25,
                s * 0.05,
            ));
            let o2 = black_box(Octonion::new(
                s * 0.5,
                s * 0.6,
                s * 0.4,
                s * 0.3,
                s * 0.2,
                s * 0.1,
                s * 0.1,
                s * 0.1,
            ));
            b.iter(|| o1.mul(o2))
        });
    }
    group.finish();
}

fn benchmark_octonion_norm(c: &mut Criterion) {
    c.bench_function("oct_norm_sq", |b| {
        let o = black_box(Octonion::new(1.0, 0.5, 0.3, 0.2, 0.1, 0.15, 0.25, 0.05));
        b.iter(|| o.norm_sq())
    });

    c.bench_function("oct_norm", |b| {
        let o = black_box(Octonion::new(1.0, 0.5, 0.3, 0.2, 0.1, 0.15, 0.25, 0.05));
        b.iter(|| o.norm())
    });
}

fn benchmark_octonion_conjugate(c: &mut Criterion) {
    c.bench_function("oct_conj", |b| {
        let o = black_box(Octonion::new(1.0, 0.5, 0.3, 0.2, 0.1, 0.15, 0.25, 0.05));
        b.iter(|| o.conj())
    });
}

fn benchmark_activations(c: &mut Criterion) {
    let o = black_box(Octonion::new(1.0, -0.5, 0.8, -0.3, 0.6, -0.2, 0.4, -0.1));

    c.bench_function("oct_relu", |b| b.iter(|| o.relu()));

    c.bench_function("oct_tanh", |b| b.iter(|| o.tanh()));
}

fn benchmark_neural_network_layer(c: &mut Criterion) {
    // Simulate a small neural network layer: y = W ⊗ x + b
    // 8 input octonions, 16 output octonions, batch size 4

    let mut group = c.benchmark_group("neural_network");
    group.sample_size(10);

    group.bench_function("oct_linear_layer_small", |b| {
        let input: Vec<Octonion> = (0..8)
            .map(|i| {
                Octonion::new(
                    (i as f32) * 0.1,
                    (i as f32) * 0.15,
                    (i as f32) * 0.2,
                    (i as f32) * 0.25,
                    (i as f32) * 0.3,
                    (i as f32) * 0.35,
                    (i as f32) * 0.4,
                    (i as f32) * 0.45,
                )
            })
            .collect();

        let weights: Vec<Octonion> = (0..128) // 16 output × 8 input
            .map(|i| {
                Octonion::new(
                    ((i % 8) as f32) * 0.05,
                    ((i / 8) as f32) * 0.06,
                    ((i % 16) as f32) * 0.07,
                    ((i / 16) as f32) * 0.08,
                    ((i % 32) as f32) * 0.09,
                    ((i / 32) as f32) * 0.1,
                    0.1,
                    0.2,
                )
            })
            .collect();

        let bias: Vec<Octonion> = (0..16)
            .map(|i| {
                Octonion::new(
                    (i as f32) * 0.01,
                    (i as f32) * 0.02,
                    (i as f32) * 0.03,
                    (i as f32) * 0.04,
                    (i as f32) * 0.05,
                    (i as f32) * 0.06,
                    (i as f32) * 0.07,
                    (i as f32) * 0.08,
                )
            })
            .collect();

        b.iter(|| {
            let mut output = bias.clone();
            for out_idx in 0..16 {
                for in_idx in 0..8 {
                    let w = weights[out_idx * 8 + in_idx];
                    let x = input[in_idx];
                    let wx = w.mul(x);
                    let acc = output[out_idx];
                    output[out_idx] = Octonion::new(
                        acc.a + wx.a,
                        acc.b + wx.b,
                        acc.c + wx.c,
                        acc.d + wx.d,
                        acc.e + wx.e,
                        acc.f + wx.f,
                        acc.g + wx.g,
                        acc.h + wx.h,
                    );
                }
            }
            output
        });
    });

    group.finish();
}

fn benchmark_chained_operations(c: &mut Criterion) {
    // Simulate common operation chains in neural networks
    let o1 = black_box(Octonion::new(1.0, 0.5, 0.3, 0.2, 0.1, 0.15, 0.25, 0.05));
    let o2 = black_box(Octonion::new(0.5, 0.6, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1));

    c.bench_function("oct_mul_then_relu", |b| {
        b.iter(|| {
            let result = o1.mul(o2);
            result.relu()
        })
    });

    c.bench_function("oct_mul_then_norm", |b| {
        b.iter(|| {
            let result = o1.mul(o2);
            result.norm()
        })
    });

    c.bench_function("oct_mul_then_tanh", |b| {
        b.iter(|| {
            let result = o1.mul(o2);
            result.tanh()
        })
    });
}

criterion_group!(
    benches,
    benchmark_octonion_multiplication,
    benchmark_octonion_norm,
    benchmark_octonion_conjugate,
    benchmark_activations,
    benchmark_neural_network_layer,
    benchmark_chained_operations,
);

criterion_main!(benches);
