//! Phase 2: Quantization Performance and Accuracy Benchmarks
//!
//! Benchmarks for:
//! - INT8 quantization (4x memory reduction, <1% accuracy loss)
//! - INT4 quantization (8x memory reduction)
//! - Per-channel vs per-tensor quantization
//! - Fake quantization for QAT
//!
//! Metrics: Speed, memory, accuracy loss

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

/// Simulate INT8 symmetric quantization
fn quantize_int8_symmetric(values: &[f32], scale: f32) -> Vec<i8> {
    values.iter()
        .map(|&v| ((v / scale).round().clamp(-127.0, 127.0)) as i8)
        .collect()
}

/// Simulate INT8 asymmetric quantization
fn quantize_int8_asymmetric(values: &[f32], scale: f32, zp: i32) -> Vec<u8> {
    values.iter()
        .map(|&v| (((v / scale) + zp as f32).round().clamp(0.0, 255.0)) as u8)
        .collect()
}

/// Simulate INT4 quantization
fn quantize_int4(values: &[f32], scale: f32) -> Vec<u8> {
    values.iter()
        .map(|&v| {
            let q = ((v / scale).round().clamp(0.0, 15.0)) as u8;
            q & 0xF
        })
        .collect()
}

/// Dequantize INT8 symmetric
fn dequantize_int8_symmetric(quantized: &[i8], scale: f32) -> Vec<f32> {
    quantized.iter()
        .map(|&v| (v as f32) * scale)
        .collect()
}

/// Dequantize INT8 asymmetric
fn dequantize_int8_asymmetric(quantized: &[u8], scale: f32, zp: i32) -> Vec<f32> {
    quantized.iter()
        .map(|&v| ((v as f32) - zp as f32) * scale)
        .collect()
}

/// Fake quantization for QAT (quantize + dequantize)
fn fake_quantize(values: &[f32], scale: f32) -> Vec<f32> {
    values.iter()
        .map(|&v| {
            let q = (v / scale).round() * scale;
            q.clamp(-1.0, 1.0)
        })
        .collect()
}

/// Per-channel quantization
fn quantize_per_channel(matrix: &[f32], n: usize) -> (Vec<f32>, Vec<i8>) {
    let mut scales = Vec::new();
    let mut quantized = Vec::new();

    for i in 0..n {
        let row = &matrix[i * n..(i + 1) * n];
        let max_val = row.iter().map(|x| x.abs()).fold(0.0, f32::max);
        let scale = if max_val > 0.0 { max_val / 127.0 } else { 1.0 };
        scales.push(scale);

        for &v in row {
            quantized.push(((v / scale).round().clamp(-127.0, 127.0)) as i8);
        }
    }

    (scales, quantized)
}

/// Calculate quantization error (RMSE)
fn calculate_quantization_error(original: &[f32], reconstructed: &[f32]) -> f32 {
    let mse: f32 = original.iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>() / original.len() as f32;

    mse.sqrt()
}

/// Calculate model size reduction
fn calculate_compression_ratio(original_size: usize, quantized_size: usize) -> f32 {
    original_size as f32 / quantized_size as f32
}

fn generate_neural_net_weights(size: usize) -> Vec<f32> {
    (0..size).map(|i| {
        let seed = i as f32;
        (seed.sin() * seed.cos() * 0.5)
    }).collect()
}

fn bench_int8_symmetric_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize_int8_symmetric");

    for size in &[1000, 10000, 100000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let values = black_box(generate_neural_net_weights(size));
            let scale = 1.0 / 127.0;

            b.iter(|| {
                let quantized = quantize_int8_symmetric(&values, scale);
                black_box(quantized);
            });
        });
    }

    group.finish();
}

fn bench_int8_asymmetric_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize_int8_asymmetric");

    for size in &[1000, 10000, 100000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let values = black_box(generate_neural_net_weights(size));
            let scale = 2.0 / 255.0;
            let zp = 128;

            b.iter(|| {
                let quantized = quantize_int8_asymmetric(&values, scale, zp);
                black_box(quantized);
            });
        });
    }

    group.finish();
}

fn bench_int4_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize_int4");

    for size in &[1000, 10000, 100000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let values = black_box(generate_neural_net_weights(size));
            let scale = 1.0 / 7.0;

            b.iter(|| {
                let quantized = quantize_int4(&values, scale);
                black_box(quantized);
            });
        });
    }

    group.finish();
}

fn bench_dequantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("dequantize");

    let values = generate_neural_net_weights(10000);
    let scale = 1.0 / 127.0;
    let quantized = quantize_int8_symmetric(&values, scale);

    group.bench_function("int8_symmetric_dequant", |b| {
        b.iter(|| {
            let reconstructed = dequantize_int8_symmetric(
                black_box(&quantized),
                scale,
            );
            black_box(reconstructed);
        });
    });

    group.finish();
}

fn bench_fake_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("fake_quantization_qat");

    for size in &[1000, 10000, 100000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let values = black_box(generate_neural_net_weights(size));
            let scale = 0.1;

            b.iter(|| {
                let quantized = fake_quantize(&values, scale);
                black_box(quantized);
            });
        });
    }

    group.finish();
}

fn bench_per_channel_vs_per_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("per_channel_vs_per_tensor");

    for n in &[16, 32, 64] {
        group.bench_with_input(
            BenchmarkId::new("per_tensor", n),
            n,
            |b, &n| {
                let matrix = black_box(generate_neural_net_weights(n * n));
                let scale = 1.0 / 127.0;

                b.iter(|| {
                    let quantized = quantize_int8_symmetric(&matrix, scale);
                    black_box(quantized);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("per_channel", n),
            n,
            |b, &n| {
                let matrix = black_box(generate_neural_net_weights(n * n));

                b.iter(|| {
                    let (_scales, quantized) = quantize_per_channel(&matrix, n);
                    black_box(quantized);
                });
            },
        );
    }

    group.finish();
}

fn bench_quantization_accuracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_accuracy");

    group.bench_function("int8_error_rmse", |b| {
        let original = black_box(generate_neural_net_weights(10000));
        let scale = 1.0 / 127.0;
        let quantized = quantize_int8_symmetric(&original, scale);
        let reconstructed = dequantize_int8_symmetric(&quantized, scale);

        b.iter(|| {
            let error = calculate_quantization_error(&original, &reconstructed);
            black_box(error);
        });
    });

    group.bench_function("int4_error_rmse", |b| {
        let original = black_box(generate_neural_net_weights(10000));
        let scale = 1.0 / 7.0;
        let quantized = quantize_int4(&original, scale);
        let reconstructed: Vec<f32> = quantized.iter()
            .map(|&q| (q as f32) * scale)
            .collect();

        b.iter(|| {
            let error = calculate_quantization_error(&original, &reconstructed);
            black_box(error);
        });
    });

    group.finish();
}

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    group.bench_function("int8_compression", |b| {
        let original = black_box(generate_neural_net_weights(1_000_000));
        let scale = 1.0 / 127.0;

        b.iter(|| {
            let quantized = quantize_int8_symmetric(&original, scale);
            let ratio = calculate_compression_ratio(
                original.len() * 4,  // FP32: 4 bytes
                quantized.len(),      // INT8: 1 byte
            );
            black_box(ratio);
        });
    });

    group.bench_function("int4_compression", |b| {
        let original = black_box(generate_neural_net_weights(1_000_000));
        let scale = 1.0 / 7.0;

        b.iter(|| {
            let quantized = quantize_int4(&original, scale);
            let ratio = calculate_compression_ratio(
                original.len() * 4,
                quantized.len() / 2, // Two INT4 values per byte
            );
            black_box(ratio);
        });
    });

    group.finish();
}

fn bench_calibration(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_calibration");

    group.bench_function("minmax_calibration", |b| {
        let calibration_data: Vec<Vec<f32>> = (0..100)
            .map(|_| black_box(generate_neural_net_weights(10000)))
            .collect();

        b.iter(|| {
            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;

            for batch in &calibration_data {
                for &v in batch {
                    min_val = min_val.min(v);
                    max_val = max_val.max(v);
                }
            }

            let scale = (max_val - min_val) / 255.0;
            black_box((scale, min_val, max_val));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_int8_symmetric_quantization,
    bench_int8_asymmetric_quantization,
    bench_int4_quantization,
    bench_dequantization,
    bench_fake_quantization,
    bench_per_channel_vs_per_tensor,
    bench_quantization_accuracy,
    bench_memory_efficiency,
    bench_calibration,
);

criterion_main!(benches);
