//! Quaternion Performance Benchmarking Suite
//!
//! This benchmark suite measures key performance characteristics of quaternion operations:
//! - Hamilton product (28 FLOPs per multiplication)
//! - Rotation operations (2 quaternion multiplications)
//! - Batch operations (SIMD-optimized)
//! - Neural network layer throughput
//!
//! Run with: cargo bench --bench quat_bench

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

/// Quaternion representation: w + xi + yj + zk
#[derive(Clone, Copy, Debug)]
struct Quaternion {
    w: f32,
    x: f32,
    y: f32,
    z: f32,
}

impl Quaternion {
    fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Quaternion { w, x, y, z }
    }

    fn identity() -> Self {
        Quaternion::new(1.0, 0.0, 0.0, 0.0)
    }

    /// Hamilton product: 16 multiplications + 12 additions = 28 FLOPs
    fn mul(self, other: Quaternion) -> Quaternion {
        Quaternion {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }

    /// Conjugate: 3 negations
    fn conj(self) -> Quaternion {
        Quaternion::new(self.w, -self.x, -self.y, -self.z)
    }

    /// Squared norm: 4 multiplications + 3 additions = 7 FLOPs
    fn norm_sq(self) -> f32 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Euclidean norm: 7 FLOPs + 1 sqrt
    fn norm(self) -> f32 {
        self.norm_sq().sqrt()
    }

    /// Normalize to unit quaternion
    fn normalize(self) -> Quaternion {
        let n = self.norm();
        if n > 1e-10 {
            let inv_n = 1.0 / n;
            Quaternion::new(
                self.w * inv_n,
                self.x * inv_n,
                self.y * inv_n,
                self.z * inv_n,
            )
        } else {
            Quaternion::identity()
        }
    }

    /// Rotate a 3D vector by this quaternion: q * v * q^(-1)
    /// For unit quaternions: q * v * conj(q)
    /// Total: 2 quaternion multiplications = 56 FLOPs
    fn rotate_vector(self, v: [f32; 3]) -> [f32; 3] {
        let v_quat = Quaternion::new(0.0, v[0], v[1], v[2]);
        let rotated = self.mul(v_quat).mul(self.conj());
        [rotated.x, rotated.y, rotated.z]
    }

    /// SLERP interpolation between two quaternions
    fn slerp(self, other: Quaternion, t: f32) -> Quaternion {
        let dot = self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z;
        let dot = dot.clamp(-1.0, 1.0);

        if dot.abs() > 0.9995 {
            // Linear interpolation for nearly parallel quaternions
            let result = Quaternion::new(
                self.w + t * (other.w - self.w),
                self.x + t * (other.x - self.x),
                self.y + t * (other.y - self.y),
                self.z + t * (other.z - self.z),
            );
            result.normalize()
        } else {
            let theta = dot.acos();
            let sin_theta = theta.sin();
            let a = ((1.0 - t) * theta).sin() / sin_theta;
            let b = (t * theta).sin() / sin_theta;
            Quaternion::new(
                a * self.w + b * other.w,
                a * self.x + b * other.x,
                a * self.y + b * other.y,
                a * self.z + b * other.z,
            )
        }
    }
}

fn benchmark_quaternion_multiplication(c: &mut Criterion) {
    let q1 = black_box(Quaternion::new(0.707, 0.707, 0.0, 0.0));
    let q2 = black_box(Quaternion::new(0.707, 0.0, 0.707, 0.0));

    c.bench_function("quat_mul_single", |b| b.iter(|| q1.mul(q2)));

    // Benchmark with different quaternion magnitudes
    let mut group = c.benchmark_group("quat_mul_variants");
    for scale in [0.1, 1.0, 10.0].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(scale), scale, |b, &s| {
            let q1 = black_box(Quaternion::new(s * 0.707, s * 0.707, 0.0, 0.0));
            let q2 = black_box(Quaternion::new(s * 0.707, 0.0, s * 0.707, 0.0));
            b.iter(|| q1.mul(q2))
        });
    }
    group.finish();
}

fn benchmark_quaternion_rotation(c: &mut Criterion) {
    // 90-degree rotation around X axis
    let q = black_box(Quaternion::new(0.707, 0.707, 0.0, 0.0));
    let v = black_box([1.0, 0.0, 0.0]);

    c.bench_function("quat_rotate_vector", |b| b.iter(|| q.rotate_vector(v)));
}

fn benchmark_quaternion_slerp(c: &mut Criterion) {
    let q1 = black_box(Quaternion::new(1.0, 0.0, 0.0, 0.0));
    let q2 = black_box(Quaternion::new(0.707, 0.707, 0.0, 0.0));

    c.bench_function("quat_slerp", |b| b.iter(|| q1.slerp(q2, 0.5)));
}

fn benchmark_quaternion_norm(c: &mut Criterion) {
    let q = black_box(Quaternion::new(0.5, 0.5, 0.5, 0.5));

    c.bench_function("quat_norm_sq", |b| b.iter(|| q.norm_sq()));

    c.bench_function("quat_norm", |b| b.iter(|| q.norm()));

    c.bench_function("quat_normalize", |b| b.iter(|| q.normalize()));
}

fn benchmark_quaternion_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("quat_batch_operations");

    for batch_size in [64, 256, 1024, 4096].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("mul", batch_size),
            batch_size,
            |b, &size| {
                let quats: Vec<Quaternion> = (0..size)
                    .map(|i| {
                        let t = (i as f32) * 0.01;
                        Quaternion::new(t.cos(), t.sin(), 0.0, 0.0).normalize()
                    })
                    .collect();

                let q_rot = black_box(Quaternion::new(0.707, 0.707, 0.0, 0.0));

                b.iter(|| quats.iter().map(|q| q.mul(q_rot)).collect::<Vec<_>>())
            },
        );

        group.bench_with_input(
            BenchmarkId::new("rotate", batch_size),
            batch_size,
            |b, &size| {
                let vectors: Vec<[f32; 3]> = (0..size)
                    .map(|i| {
                        let t = (i as f32) * 0.01;
                        [t.cos(), t.sin(), 0.0]
                    })
                    .collect();

                let q = black_box(Quaternion::new(0.707, 0.707, 0.0, 0.0));

                b.iter(|| {
                    vectors
                        .iter()
                        .map(|v| q.rotate_vector(*v))
                        .collect::<Vec<_>>()
                })
            },
        );
    }
    group.finish();
}

fn benchmark_neural_network_layer(c: &mut Criterion) {
    // Simulate a quaternion neural network layer
    // Input: batch of quaternions, Output: transformed quaternions
    let mut group = c.benchmark_group("quat_nn_layer");
    group.sample_size(10);

    // Layer sizes: 16 input quaternions -> 32 output quaternions
    let input_size = 16;
    let output_size = 32;

    let inputs: Vec<Quaternion> = (0..input_size)
        .map(|i| {
            Quaternion::new(
                (i as f32) * 0.1,
                (i as f32) * 0.15,
                (i as f32) * 0.2,
                (i as f32) * 0.25,
            )
            .normalize()
        })
        .collect();

    let weights: Vec<Quaternion> = (0..input_size * output_size)
        .map(|i| {
            Quaternion::new(
                ((i % 4) as f32) * 0.1,
                ((i % 8) as f32) * 0.05,
                ((i % 16) as f32) * 0.03,
                ((i % 32) as f32) * 0.02,
            )
            .normalize()
        })
        .collect();

    let biases: Vec<Quaternion> = (0..output_size)
        .map(|_| Quaternion::new(0.01, 0.0, 0.0, 0.0))
        .collect();

    group.bench_function("quat_linear_forward", |b| {
        b.iter(|| {
            let mut output = biases.clone();
            for out_idx in 0..output_size {
                for in_idx in 0..input_size {
                    let w = weights[out_idx * input_size + in_idx];
                    let x = inputs[in_idx];
                    let wx = w.mul(x);
                    output[out_idx] = Quaternion::new(
                        output[out_idx].w + wx.w,
                        output[out_idx].x + wx.x,
                        output[out_idx].y + wx.y,
                        output[out_idx].z + wx.z,
                    );
                }
            }
            output
        });
    });

    group.finish();
}

fn benchmark_comparison_real_vs_quat(c: &mut Criterion) {
    // Compare quaternion operations to equivalent real operations
    let mut group = c.benchmark_group("real_vs_quat");

    // Real: 3x3 rotation matrix multiplication (27 muls + 18 adds = 45 FLOPs)
    // Quat: 2 quaternion muls (56 FLOPs) but more cache-friendly

    let matrix: [[f32; 3]; 3] = [[0.707, -0.707, 0.0], [0.707, 0.707, 0.0], [0.0, 0.0, 1.0]];
    let v = [1.0f32, 2.0, 3.0];

    group.bench_function("mat3x3_rotate", |b| {
        b.iter(|| {
            let m = black_box(matrix);
            let v = black_box(v);
            [
                m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
                m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
                m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
            ]
        })
    });

    let q = Quaternion::new(0.924, 0.0, 0.0, 0.383); // 45-degree Z rotation

    group.bench_function("quat_rotate", |b| {
        b.iter(|| {
            let q = black_box(q);
            let v = black_box(v);
            q.rotate_vector(v)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_quaternion_multiplication,
    benchmark_quaternion_rotation,
    benchmark_quaternion_slerp,
    benchmark_quaternion_norm,
    benchmark_quaternion_batch,
    benchmark_neural_network_layer,
    benchmark_comparison_real_vs_quat,
);

criterion_main!(benches);
