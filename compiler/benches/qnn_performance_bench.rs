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

    // Batch multiplication - small
    group.bench_function("quat_batch_mul_10", |b| {
        let q1s = vec![0.7071; 10 * 4];
        let q2s = vec![0.5; 10 * 4];

        b.iter(|| {
            let mut _result_sum = 0.0;
            for i in 0..10 {
                let q1 = [q1s[i*4], q1s[i*4+1], q1s[i*4+2], q1s[i*4+3]];
                let q2 = [q2s[i*4], q2s[i*4+1], q2s[i*4+2], q2s[i*4+3]];
                let result = quat_mul(black_box(&q1), black_box(&q2));
                _result_sum += result[0];
            }
            black_box(_result_sum)
        });
    });

    // Batch multiplication - medium
    group.bench_function("quat_batch_mul_100", |b| {
        let q1s = vec![0.7071; 100 * 4];
        let q2s = vec![0.5; 100 * 4];

        b.iter(|| {
            let mut _result_sum = 0.0;
            for i in 0..100 {
                let q1 = [q1s[i*4], q1s[i*4+1], q1s[i*4+2], q1s[i*4+3]];
                let q2 = [q2s[i*4], q2s[i*4+1], q2s[i*4+2], q2s[i*4+3]];
                let result = quat_mul(black_box(&q1), black_box(&q2));
                _result_sum += result[0];
            }
            black_box(_result_sum)
        });
    });

    // Batch rotations - small
    group.bench_function("quat_batch_rotate_10", |b| {
        let qs = vec![0.7071; 10 * 4];
        let vs = vec![1.0; 10 * 3];

        b.iter(|| {
            let mut _result_sum = 0.0;
            for i in 0..10 {
                let q = [qs[i*4], qs[i*4+1], qs[i*4+2], qs[i*4+3]];
                let v = [vs[i*3], vs[i*3+1], vs[i*3+2]];
                let result = quat_rotate(black_box(&q), black_box(&v));
                _result_sum += result[0];
            }
            black_box(_result_sum)
        });
    });

    group.finish();
}

fn neural_network_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("neural_network_layers");

    // Linear layer: (batch, in_features*4) -> (out_features*4)
    group.bench_function("quat_linear_layer_small", |b| {
        let x = vec![0.1; 16];
        let w = vec![0.5; 16 * 32];
        let bias = vec![0.1; 32];
        let mut y = vec![0.0; 32];

        b.iter(|| {
            quat_linear(
                black_box(&x),
                16,
                black_box(&w),
                black_box(&bias),
                &mut y,
                32,
            );
        });
    });

    group.bench_function("quat_linear_layer_medium", |b| {
        let x = vec![0.1; 64];
        let w = vec![0.5; 64 * 128];
        let bias = vec![0.1; 128];
        let mut y = vec![0.0; 128];

        b.iter(|| {
            quat_linear(
                black_box(&x),
                64,
                black_box(&w),
                black_box(&bias),
                &mut y,
                128,
            );
        });
    });

    group.bench_function("quat_linear_layer_large", |b| {
        let x = vec![0.1; 256];
        let w = vec![0.5; 256 * 512];
        let bias = vec![0.1; 512];
        let mut y = vec![0.0; 512];

        b.iter(|| {
            quat_linear(
                black_box(&x),
                256,
                black_box(&w),
                black_box(&bias),
                &mut y,
                512,
            );
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    quat_ops_benchmark,
    batch_operations_benchmark,
    neural_network_benchmark
);
criterion_main!(benches);
