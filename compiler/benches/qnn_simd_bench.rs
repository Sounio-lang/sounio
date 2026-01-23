//! Benchmarks for QNN SIMD optimizations
//!
//! Compares scalar vs AVX2 vs AVX-512 implementations
//! Run with: cargo bench --bench qnn_simd_bench

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

// Simulated quaternion operations for benchmarking
// In production, these would call into the compiled Sounio library

fn scalar_quat_mul(q1: &[f32; 4], q2: &[f32; 4], out: &mut [f32; 4]) {
    let w1 = q1[0];
    let x1 = q1[1];
    let y1 = q1[2];
    let z1 = q1[3];

    let w2 = q2[0];
    let x2 = q2[1];
    let y2 = q2[2];
    let z2 = q2[3];

    out[0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
    out[1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
    out[2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
    out[3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
}

fn bench_scalar_single(c: &mut Criterion) {
    let q1 = [0.707, 0.707, 0.0, 0.0];
    let q2 = [0.0, 1.0, 0.0, 0.0];

    c.bench_function("quat_mul_scalar_single", |b| {
        b.iter(|| {
            let mut out = [0.0; 4];
            scalar_quat_mul(black_box(&q1), black_box(&q2), black_box(&mut out));
            out
        })
    });
}

fn bench_scalar_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("quat_mul_batch_scalar");

    for batch_size in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &batch_size| {
                let q1s: Vec<[f32; 4]> = (0..batch_size)
                    .map(|i| {
                        let angle = (i as f32) * 0.1;
                        [angle.cos(), angle.sin(), 0.0, 0.0]
                    })
                    .collect();
                let q2s: Vec<[f32; 4]> = (0..batch_size)
                    .map(|i| {
                        let angle = (i as f32) * 0.2;
                        [angle.cos(), 0.0, angle.sin(), 0.0]
                    })
                    .collect();

                b.iter(|| {
                    let mut outs = vec![[0.0; 4]; batch_size];
                    for i in 0..batch_size {
                        scalar_quat_mul(
                            black_box(&q1s[i]),
                            black_box(&q2s[i]),
                            black_box(&mut outs[i]),
                        );
                    }
                    outs
                })
            },
        );
    }

    group.finish();
}

fn bench_linear_layer_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("quat_linear_scalar");

    for input_dim in [8, 16, 32].iter() {
        group.bench_with_input(
            BenchmarkId::new("input_dim", input_dim),
            input_dim,
            |b, &input_dim| {
                let output_dim = 16;
                let batch_size = 4;

                let input: Vec<f32> = (0..batch_size * input_dim * 4)
                    .map(|i| ((i as f32) * 0.1).sin())
                    .collect();

                let weights: Vec<f32> = (0..output_dim * input_dim * 4)
                    .map(|i| ((i as f32) * 0.05).cos())
                    .collect();

                let bias: Vec<f32> = (0..output_dim * 4)
                    .map(|i| ((i as f32) * 0.1).sin())
                    .collect();

                b.iter(|| {
                    let mut output = vec![0.0f32; batch_size * output_dim * 4];

                    for b in 0..batch_size {
                        for o in 0..output_dim {
                            let mut acc = [0.0f32; 4];

                            for i in 0..input_dim {
                                let in_idx = (b * input_dim + i) * 4;
                                let w_idx = (o * input_dim + i) * 4;

                                let in_q = [
                                    input[in_idx],
                                    input[in_idx + 1],
                                    input[in_idx + 2],
                                    input[in_idx + 3],
                                ];

                                let w_q = [
                                    weights[w_idx],
                                    weights[w_idx + 1],
                                    weights[w_idx + 2],
                                    weights[w_idx + 3],
                                ];

                                let qw = in_q[0] * w_q[0]
                                    - in_q[1] * w_q[1]
                                    - in_q[2] * w_q[2]
                                    - in_q[3] * w_q[3];
                                let qx = in_q[0] * w_q[1]
                                    + in_q[1] * w_q[0]
                                    + in_q[2] * w_q[3]
                                    - in_q[3] * w_q[2];
                                let qy = in_q[0] * w_q[2]
                                    - in_q[1] * w_q[3]
                                    + in_q[2] * w_q[0]
                                    + in_q[3] * w_q[1];
                                let qz = in_q[0] * w_q[3]
                                    + in_q[1] * w_q[2]
                                    - in_q[2] * w_q[1]
                                    + in_q[3] * w_q[0];

                                acc[0] += qw;
                                acc[1] += qx;
                                acc[2] += qy;
                                acc[3] += qz;
                            }

                            let out_idx = (b * output_dim + o) * 4;
                            output[out_idx] = acc[0] + bias[o * 4];
                            output[out_idx + 1] = acc[1] + bias[o * 4 + 1];
                            output[out_idx + 2] = acc[2] + bias[o * 4 + 2];
                            output[out_idx + 3] = acc[3] + bias[o * 4 + 3];
                        }
                    }

                    output
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_scalar_single,
    bench_scalar_batch,
    bench_linear_layer_scalar
);

criterion_main!(benches);
