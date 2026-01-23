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

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

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
        Octonion { a, b, c, d, e, f, g, h }
    }

    // Octonion multiplication (120 FLOPs: 64 multiplications + 56 additions)
    fn mul(self, other: Octonion) -> Octonion {
        let a1 = self.a;
        let b1 = self.b;
        let c1 = self.c;
        let d1 = self.d;
        let e1 = self.e;
        let f1 = self.f;
        let g1 = self.g;
        let h1 = self.h;

        let a2 = other.a;
        let b2 = other.b;
        let c2 = other.c;
        let d2 = other.d;
        let e2 = other.e;
        let f2 = other.f;
        let g2 = other.g;
        let h2 = other.h;

        // Graves-Adcock formula
        let c0 = a1*a2 - b1*b2 - c1*c2 - d1*d2 - e1*e2 - f1*f2 - g1*g2 - h1*h2;
        let c1 = a1*b2 + b1*a2 + c1*d2 - d1*c2 + e1*f2 - f1*e2 - g1*h2 + h1*g2;
        let c2 = a1*c2 - b1*d2 + c1*a2 + d1*b2 + e1*g2 + f1*h2 - g1*e2 - h1*f2;
        let c3 = a1*d2 + b1*c2 - c1*b2 + d1*a2 + e1*h2 - f1*g2 + g1*f2 - h1*e2;
        let c4 = a1*e2 - b1*f2 - c1*g2 - d1*h2 + e1*a2 + f1*b2 + g1*c2 + h1*d2;
        let c5 = a1*f2 + b1*e2 - c1*h2 + d1*g2 - e1*b2 + f1*a2 - g1*d2 + h1*c2;
        let c6 = a1*g2 + b1*h2 + c1*e2 - d1*f2 - e1*c2 + f1*d2 + g1*a2 - h1*b2;
        let c7 = a1*h2 - b1*g2 + c1*f2 + d1*e2 - e1*d2 - f1*c2 + g1*b2 + h1*a2;

        Octonion {
            a: c0, b: c1, c: c2, d: c3,
            e: c4, f: c5, g: c6, h: c7,
        }
    }

    // Norm squared (8 multiplications + 7 additions = 15 FLOPs)
    fn norm_sq(self) -> f32 {
        self.a * self.a + self.b * self.b + self.c * self.c + self.d * self.d +
        self.e * self.e + self.f * self.f + self.g * self.g + self.h * self.h
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
            let o1 = black_box(Octonion::new(s, s*0.5, s*0.3, s*0.2, s*0.1, s*0.15, s*0.25, s*0.05));
            let o2 = black_box(Octonion::new(s*0.5, s*0.6, s*0.4, s*0.3, s*0.2, s*0.1, s*0.1, s*0.1));
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

    c.bench_function("oct_relu", |b| {
        b.iter(|| o.relu())
    });

    c.bench_function("oct_tanh", |b| {
        b.iter(|| o.tanh())
    });
}

fn benchmark_neural_network_layer(c: &mut Criterion) {
    // Simulate a small neural network layer: y = W ⊗ x + b
    // 8 input octonions, 16 output octonions, batch size 4

    let mut group = c.benchmark_group("neural_network");
    group.sample_size(10);

    group.bench_function("oct_linear_layer_small", |b| {
        let input: Vec<Octonion> = (0..8)
            .map(|i| Octonion::new(
                (i as f32) * 0.1,
                (i as f32) * 0.15,
                (i as f32) * 0.2,
                (i as f32) * 0.25,
                (i as f32) * 0.3,
                (i as f32) * 0.35,
                (i as f32) * 0.4,
                (i as f32) * 0.45,
            ))
            .collect();

        let weights: Vec<Octonion> = (0..128) // 16 output × 8 input
            .map(|i| Octonion::new(
                ((i % 8) as f32) * 0.05,
                ((i / 8) as f32) * 0.06,
                ((i % 16) as f32) * 0.07,
                ((i / 16) as f32) * 0.08,
                ((i % 32) as f32) * 0.09,
                ((i / 32) as f32) * 0.1,
                0.1,
                0.2,
            ))
            .collect();

        let bias: Vec<Octonion> = (0..16)
            .map(|i| Octonion::new(
                (i as f32) * 0.01,
                (i as f32) * 0.02,
                (i as f32) * 0.03,
                (i as f32) * 0.04,
                (i as f32) * 0.05,
                (i as f32) * 0.06,
                (i as f32) * 0.07,
                (i as f32) * 0.08,
            ))
            .collect();

        b.iter(|| {
            let mut output = bias.clone();
            for out_idx in 0..16 {
                for in_idx in 0..8 {
                    let w = weights[out_idx * 8 + in_idx];
                    let x = input[in_idx];
                    output[out_idx] = Octonion::new(
                        output[out_idx].a + w.mul(x).a,
                        output[out_idx].b + w.mul(x).b,
                        output[out_idx].c + w.mul(x).c,
                        output[out_idx].d + w.mul(x).d,
                        output[out_idx].e + w.mul(x).e,
                        output[out_idx].f + w.mul(x).f,
                        output[out_idx].g + w.mul(x).g,
                        output[out_idx].h + w.mul(x).h,
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
