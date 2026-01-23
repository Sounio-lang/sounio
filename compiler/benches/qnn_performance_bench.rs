// QNN Performance Benchmarks
// Measures quaternion operation performance on CPU native backend

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

// Quaternion operations
fn quat_mul(q1: &[f32; 4], q2: &[f32; 4]) -> [f32; 4] {
    let w1 = q1[0];
    let x1 = q1[1];
    let y1 = q1[2];
    let z1 = q1[3];

    let w2 = q2[0];
    let x2 = q2[1];
    let y2 = q2[2];
    let z2 = q2[3];

    [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]
}

fn quat_conj(q: &[f32; 4]) -> [f32; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

fn quat_norm_sq(q: &[f32; 4]) -> f32 {
    q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
}

fn quat_norm(q: &[f32; 4]) -> f32 {
    quat_norm_sq(q).sqrt()
}

fn quat_normalize(q: &[f32; 4]) -> [f32; 4] {
    let norm = quat_norm(q);
    if norm > 1e-10 {
        let inv_norm = 1.0 / norm;
        [
            q[0] * inv_norm,
            q[1] * inv_norm,
            q[2] * inv_norm,
            q[3] * inv_norm,
        ]
    } else {
        [0.0, 0.0, 0.0, 0.0]
    }
}

fn quat_rotate(q: &[f32; 4], v: &[f32; 3]) -> [f32; 3] {
    let w = q[0];
    let x = q[1];
    let y = q[2];
    let z = q[3];

    let vx = v[0];
    let vy = v[1];
    let vz = v[2];

    let p_x = w * vx + y * vz - z * vy;
    let p_y = w * vy - x * vz + z * vx;
    let p_z = w * vz + x * vy - y * vx;
    let p_w = -(x * vx + y * vy + z * vz);

    let out_x = p_w * (-x) + p_x * w + p_y * (-z) - p_z * (-y);
    let out_y = p_w * (-y) - p_x * (-z) + p_y * w + p_z * (-x);
    let out_z = p_w * (-z) + p_x * (-y) - p_y * (-x) + p_z * w;

    [out_x, out_y, out_z]
}

fn quat_linear(x: &[f32], n_in: usize, w: &[f32], b: &[f32], y: &mut [f32], n_out: usize) {
    for i in 0..n_out {
        y[i] = b[i];
        for j in 0..n_in {
            y[i] += w[i * n_in + j] * x[j];
        }
    }
}

// Benchmarks
fn quat_ops_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("quaternion_operations");

    // Conjugate
    group.bench_function("quat_conj", |b| {
        let q = black_box([0.7071, 0.7071, 0.0, 0.0]);
        b.iter(|| quat_conj(&q))
    });

    // Norm squared
    group.bench_function("quat_norm_sq", |b| {
        let q = black_box([0.5, 0.3, 0.2, 0.1]);
        b.iter(|| quat_norm_sq(&q))
    });

    // Norm
    group.bench_function("quat_norm", |b| {
        let q = black_box([0.5, 0.3, 0.2, 0.1]);
        b.iter(|| quat_norm(&q))
    });

    // Normalize
    group.bench_function("quat_normalize", |b| {
        let q = black_box([2.0, 1.0, 0.5, 0.5]);
        b.iter(|| quat_normalize(&q))
    });

    // Hamilton product
    group.bench_function("quat_mul", |b| {
        let q1 = black_box([0.7071, 0.7071, 0.0, 0.0]);
        let q2 = black_box([0.5, 0.5, 0.5, 0.5]);
        b.iter(|| quat_mul(&q1, &q2))
    });

    // Vector rotation
    group.bench_function("quat_rotate", |b| {
        let q = black_box([0.7071, 0.7071, 0.0, 0.0]);
        let v = black_box([0.0, 1.0, 0.0]);
        b.iter(|| quat_rotate(&q, &v))
    });

    group.finish();
}

fn batch_operations_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");

    // Batch multiplication
    for batch_size in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("quat_batch_mul", batch_size),
            batch_size,
            |b, &batch_size| {
                let mut q1s = vec![0.7071; batch_size * 4];
                let mut q2s = vec![0.5; batch_size * 4];
                let mut results = vec![0.0; batch_size * 4];

                b.iter(|| {
                    for i in 0..batch_size {
                        let q1 = [q1s[i*4], q1s[i*4+1], q1s[i*4+2], q1s[i*4+3]];
                        let q2 = [q2s[i*4], q2s[i*4+1], q2s[i*4+2], q2s[i*4+3]];
                        let result = quat_mul(
                            black_box(&q1),
                            black_box(&q2)
                        );
                        results[i*4] = result[0];
                        results[i*4+1] = result[1];
                        results[i*4+2] = result[2];
                        results[i*4+3] = result[3];
                    }
                });
            },
        );
    }

    // Batch rotations
    for batch_size in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("quat_batch_rotate", batch_size),
            batch_size,
            |b, &batch_size| {
                let mut qs = vec![0.7071; batch_size * 4];
                let mut vs = vec![1.0; batch_size * 3];
                let mut results = vec![0.0; batch_size * 3];

                b.iter(|| {
                    for i in 0..batch_size {
                        let q = [qs[i*4], qs[i*4+1], qs[i*4+2], qs[i*4+3]];
                        let v = [vs[i*3], vs[i*3+1], vs[i*3+2]];
                        let result = quat_rotate(
                            black_box(&q),
                            black_box(&v)
                        );
                        results[i*3] = result[0];
                        results[i*3+1] = result[1];
                        results[i*3+2] = result[2];
                    }
                });
            },
        );
    }

    group.finish();
}

fn neural_network_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("neural_network_layers");

    // Linear layer: (batch, in_features*4) -> (out_features*4)
    for (batch, in_feat, out_feat) in [
        (1, 16, 32),
        (10, 32, 64),
        (32, 64, 128),
        (100, 128, 256),
    ].iter() {
        group.bench_with_input(
            BenchmarkId::new(
                "quat_linear_layer",
                format!("batch={},in={},out={}", batch, in_feat, out_feat)
            ),
            &(batch, in_feat, out_feat),
            |b, &(batch, in_feat, out_feat)| {
                let mut x = vec![0.1; batch * in_feat];
                let mut w = vec![0.5; out_feat * in_feat];
                let mut b = vec![0.1; out_feat];
                let mut y = vec![0.0; batch * out_feat];

                b.iter(|| {
                    for batch_idx in 0..batch {
                        quat_linear(
                            black_box(&x[batch_idx*in_feat..(batch_idx+1)*in_feat]),
                            in_feat,
                            black_box(&w),
                            black_box(&b),
                            &mut y[batch_idx*out_feat..(batch_idx+1)*out_feat],
                            out_feat
                        );
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    quat_ops_benchmark,
    batch_operations_benchmark,
    neural_network_benchmark
);
criterion_main!(benches);
