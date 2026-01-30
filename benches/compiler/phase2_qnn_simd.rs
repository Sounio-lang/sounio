//! Phase 2: Quaternion Neural Network SIMD Benchmarks
//!
//! Benchmarks for quaternion multiplication using different SIMD backends:
//! - Scalar baseline
//! - AVX2 (4x speedup expected)
//! - AVX-512 (6-8x speedup expected)
//! - NEON for ARM (3-4x speedup expected)

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use std::time::Instant;

/// Quaternion representation
#[derive(Clone, Copy, Debug)]
struct Quat {
    w: f32,
    x: f32,
    y: f32,
    z: f32,
}

impl Quat {
    fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { w, x, y, z }
    }

    /// Scalar quaternion multiplication
    fn mul_scalar(&self, other: &Quat) -> Quat {
        Quat {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }

    /// AVX2 quaternion multiplication (simulated)
    fn mul_avx2(&self, other: &Quat) -> Quat {
        // In real implementation, would use _mm256 intrinsics
        // For now, just scalar (actual AVX2 requires unsafe code)
        self.mul_scalar(other)
    }

    /// AVX-512 quaternion multiplication (simulated)
    fn mul_avx512(&self, other: &Quat) -> Quat {
        // In real implementation, would use _mm512 intrinsics
        self.mul_scalar(other)
    }

    /// NEON quaternion multiplication (ARM, simulated)
    fn mul_neon(&self, other: &Quat) -> Quat {
        // In real implementation, would use vdupq_n_f32, vmulq_f32, etc.
        self.mul_scalar(other)
    }
}

fn generate_quaternions(size: usize) -> Vec<Quat> {
    (0..size)
        .map(|i| {
            let f = (i as f32).sin();
            let g = (i as f32 * 0.5).cos();
            let h = (i as f32 * 2.0).sin();
            let i_f = (i as f32).cos();
            Quat::new(f, g, h, i_f)
        })
        .collect()
}

fn bench_quat_mul_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("quat_mul_scalar");

    for size in &[100, 1000, 10000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let quats = black_box(generate_quaternions(size));
            let q1 = black_box(quats[0]);
            let q2 = black_box(quats[size / 2]);

            b.iter(|| {
                let mut result = q1;
                for q in &quats {
                    result = result.mul_scalar(&q);
                }
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_quat_mul_avx2(c: &mut Criterion) {
    let mut group = c.benchmark_group("quat_mul_avx2");

    for size in &[100, 1000, 10000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let quats = black_box(generate_quaternions(size));

            b.iter(|| {
                let mut result = quats[0];
                for q in &quats {
                    result = result.mul_avx2(&q);
                }
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_quat_mul_avx512(c: &mut Criterion) {
    let mut group = c.benchmark_group("quat_mul_avx512");

    for size in &[100, 1000, 10000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let quats = black_box(generate_quaternions(size));

            b.iter(|| {
                let mut result = quats[0];
                for q in &quats {
                    result = result.mul_avx512(&q);
                }
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_quat_mul_neon(c: &mut Criterion) {
    let mut group = c.benchmark_group("quat_mul_neon");

    for size in &[100, 1000, 10000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let quats = black_box(generate_quaternions(size));

            b.iter(|| {
                let mut result = quats[0];
                for q in &quats {
                    result = result.mul_neon(&q);
                }
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_quat_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("quat_batch_operations");

    for batch_size in &[16, 64, 256] {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &batch_size| {
                let quats = black_box(generate_quaternions(batch_size));

                b.iter(|| {
                    let mut results = vec![Quat::new(1.0, 0.0, 0.0, 0.0); batch_size];
                    for (i, q) in quats.iter().enumerate() {
                        results[i] = results[i].mul_scalar(q);
                    }
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

fn bench_quat_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("quat_normalization");

    for size in &[100, 1000, 10000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let quats = black_box(generate_quaternions(size));

            b.iter(|| {
                let normalized: Vec<_> = quats
                    .iter()
                    .map(|q| {
                        let norm = (q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z).sqrt();
                        if norm > 0.0 {
                            Quat::new(q.w / norm, q.x / norm, q.y / norm, q.z / norm)
                        } else {
                            *q
                        }
                    })
                    .collect();
                black_box(normalized);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_quat_mul_scalar,
    bench_quat_mul_avx2,
    bench_quat_mul_avx512,
    bench_quat_mul_neon,
    bench_quat_batch_operations,
    bench_quat_normalization,
);

criterion_main!(benches);
