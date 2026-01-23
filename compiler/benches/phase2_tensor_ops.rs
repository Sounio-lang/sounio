//! Phase 2: Tensor Operations Benchmarks
//!
//! Benchmarks for:
//! - Tensor operations (reshape, slice, transpose)
//! - Einsum contraction performance
//! - Cache-friendly patterns
//! - Sparsity patterns (CSR, 2:4 structured)

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

/// Transpose a matrix (row-major)
fn transpose_row_major(matrix: &[f32], n: usize) -> Vec<f32> {
    let mut result = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            result[j * n + i] = matrix[i * n + j];
        }
    }
    result
}

/// Transpose with blocking for cache efficiency
fn transpose_blocked(matrix: &[f32], n: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 64;
    let mut result = vec![0.0; n * n];

    for ii in (0..n).step_by(BLOCK_SIZE) {
        for jj in (0..n).step_by(BLOCK_SIZE) {
            let i_end = (ii + BLOCK_SIZE).min(n);
            let j_end = (jj + BLOCK_SIZE).min(n);

            for i in ii..i_end {
                for j in jj..j_end {
                    result[j * n + i] = matrix[i * n + j];
                }
            }
        }
    }
    result
}

/// Reshape: flatten a 3D tensor to 2D
fn reshape_3d_to_2d(tensor: &[f32], shape: (usize, usize, usize)) -> Vec<f32> {
    tensor.iter().copied().collect()
}

/// Slice operation: extract a sub-tensor
fn slice_tensor(tensor: &[f32], n: usize, start: usize, size: usize) -> Vec<f32> {
    tensor[start * n..(start + size) * n].to_vec()
}

/// Simple 3-way einsum contraction: result[i,j] = sum_k A[i,k] * B[k,j]
fn einsum_ijk_kl_ij(a: &[f32], b: &[f32], k: usize, ij: usize) -> Vec<f32> {
    let mut result = vec![0.0; ij * ij];

    for i in 0..ij {
        for j in 0..ij {
            for k_idx in 0..k {
                result[i * ij + j] += a[i * k + k_idx] * b[k_idx * ij + j];
            }
        }
    }

    result
}

/// Batched matrix multiplication
fn batched_matmul(a: &[f32], b: &[f32], batch: usize, n: usize) -> Vec<f32> {
    let mut result = vec![0.0; batch * n * n];

    for b_idx in 0..batch {
        let a_offset = b_idx * n * n;
        let b_offset = b_idx * n * n;
        let res_offset = b_idx * n * n;

        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += a[a_offset + i * n + k] * b[b_offset + k * n + j];
                }
                result[res_offset + i * n + j] = sum;
            }
        }
    }

    result
}

/// CSR sparse format multiplication simulation
fn sparse_csr_matmul(row_ptrs: &[usize], col_indices: &[usize], values: &[f32],
                     b: &[f32], n: usize, nnz: usize) -> Vec<f32> {
    let mut result = vec![0.0; n * n];

    for i in 0..n {
        for j in n..n {
            let mut sum = 0.0;
            for idx in row_ptrs[i]..row_ptrs[i + 1] {
                let k = col_indices[idx];
                sum += values[idx] * b[k * n + j];
            }
            result[i * n + j] = sum;
        }
    }

    result
}

/// 2:4 structured sparsity pattern check
fn check_2x4_sparsity(pattern: &[bool]) -> bool {
    for chunk in pattern.chunks(4) {
        let nnz = chunk.iter().filter(|&&b| b).count();
        if nnz > 2 {
            return false;
        }
    }
    true
}

/// Apply 2:4 sparsity mask to multiplication result
fn apply_2x4_sparsity_matmul(a: &[f32], b: &[f32], mask: &[bool], n: usize) -> Vec<f32> {
    let mut result = vec![0.0; n * n];

    for i in 0..n {
        for j in 0..n {
            if mask[i * n + j] {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += a[i * n + k] * b[k * n + j];
                }
                result[i * n + j] = sum;
            }
        }
    }

    result
}

fn generate_matrix(n: usize) -> Vec<f32> {
    (0..n * n).map(|i| {
        ((i as f32).sin() * (i as f32).cos()) * 0.5
    }).collect()
}

fn bench_transpose_row_major(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose_row_major");

    for n in &[64, 128, 256] {
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            let matrix = black_box(generate_matrix(n));

            b.iter(|| {
                let transposed = transpose_row_major(&matrix, n);
                black_box(transposed);
            });
        });
    }

    group.finish();
}

fn bench_transpose_blocked(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose_blocked");

    for n in &[64, 128, 256] {
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            let matrix = black_box(generate_matrix(n));

            b.iter(|| {
                let transposed = transpose_blocked(&matrix, n);
                black_box(transposed);
            });
        });
    }

    group.finish();
}

fn bench_reshape(c: &mut Criterion) {
    let mut group = c.benchmark_group("reshape");

    for size in &[1000, 10000, 100000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let tensor = black_box((0..size).map(|i| i as f32).collect::<Vec<_>>());
            let n = (size as f64).sqrt() as usize;
            let shape = (1, n, n);

            b.iter(|| {
                let reshaped = reshape_3d_to_2d(&tensor, shape);
                black_box(reshaped);
            });
        });
    }

    group.finish();
}

fn bench_slice(c: &mut Criterion) {
    let mut group = c.benchmark_group("slice");

    for n in &[64, 128, 256] {
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            let tensor = black_box(generate_matrix(n));

            b.iter(|| {
                let sliced = slice_tensor(&tensor, n, 0, n / 2);
                black_box(sliced);
            });
        });
    }

    group.finish();
}

fn bench_einsum_contraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("einsum_contraction");

    for n in &[32, 64, 128] {
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            let a = black_box(generate_matrix(n));
            let b = black_box(generate_matrix(n));

            b.iter(|| {
                let result = einsum_ijk_kl_ij(&a, &b, n, n);
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_batched_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("batched_matmul");

    for (batch, n) in &[(4, 64), (16, 64), (32, 32)] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}x{}", batch, n, n)),
            &(*batch, *n),
            |b, &(batch, n)| {
                let a = black_box(generate_matrix(batch * n));
                let b_mat = black_box(generate_matrix(batch * n));

                b.iter(|| {
                    let result = batched_matmul(&a, &b_mat, batch, n);
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn bench_sparse_csr_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_csr_matmul");

    let n = 64;
    let nnz = 512; // 50% sparsity

    group.bench_function("csr_64x64_50pct", |b| {
        let row_ptrs = (0..n + 1)
            .map(|i| (i * nnz) / (n + 1))
            .collect::<Vec<_>>();
        let col_indices = (0..nnz)
            .map(|i| (i * n) / nnz)
            .collect::<Vec<_>>();
        let values = black_box(generate_matrix(nnz));
        let b = black_box(generate_matrix(n));

        b.iter(|| {
            let result = sparse_csr_matmul(
                &row_ptrs,
                &col_indices,
                &values,
                &b,
                n,
                nnz,
            );
            black_box(result);
        });
    });

    group.finish();
}

fn bench_2x4_sparsity_check(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparsity_check_2x4");

    for size in &[256, 1024, 4096] {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let pattern = black_box((0..size).map(|i| i % 4 < 2).collect::<Vec<_>>());

            b.iter(|| {
                let valid = check_2x4_sparsity(&pattern);
                black_box(valid);
            });
        });
    }

    group.finish();
}

fn bench_apply_sparsity_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_sparsity_matmul");

    for n in &[64, 128] {
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            let a = black_box(generate_matrix(n));
            let b_mat = black_box(generate_matrix(n));
            let mask = black_box((0..n * n).map(|i| i % 4 < 2).collect::<Vec<_>>());

            b.iter(|| {
                let result = apply_2x4_sparsity_matmul(&a, &b_mat, &mask, n);
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_cache_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_patterns");

    group.bench_function("sequential_access", |b| {
        let data = black_box((0..10000).map(|i| i as f32).collect::<Vec<_>>());

        b.iter(|| {
            let sum: f32 = data.iter().copied().sum();
            black_box(sum);
        });
    });

    group.bench_function("strided_access", |b| {
        let data = black_box((0..10000).map(|i| i as f32).collect::<Vec<_>>());

        b.iter(|| {
            let sum: f32 = data.iter().step_by(64).map(|&x| x).sum();
            black_box(sum);
        });
    });

    group.bench_function("random_access", |b| {
        let data = black_box((0..10000).map(|i| i as f32).collect::<Vec<_>>());
        let indices = black_box((0..1000)
            .map(|i| ((i as f32 * 7919.0) as usize) % 10000)
            .collect::<Vec<_>>());

        b.iter(|| {
            let sum: f32 = indices.iter()
                .map(|&i| data[i])
                .sum();
            black_box(sum);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_transpose_row_major,
    bench_transpose_blocked,
    bench_reshape,
    bench_slice,
    bench_einsum_contraction,
    bench_batched_matmul,
    bench_sparse_csr_matmul,
    bench_2x4_sparsity_check,
    bench_apply_sparsity_matmul,
    bench_cache_patterns,
);

criterion_main!(benches);
