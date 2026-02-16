#!/usr/bin/env bash
set -euo pipefail

# Reproduce the key validation + benchmark artifacts referenced by
# docs/compiler/TECHNICAL_REPORT.{md,tex}.
#
# Notes:
# - GPU *code generation* checks are enabled via `--features gpu`.
# - This script does not require a CUDA/Metal device; it runs CPU tests and
#   (optional) CPU microbenchmarks using Criterion.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "== Sounio Octonion Preprint Reproduction =="
echo
echo "Commit:  $(git rev-parse HEAD 2>/dev/null || echo '<no git>')"
echo "Date:    $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "Rust:    $(rustc --version)"
echo "Cargo:   $(cargo --version)"
echo

echo "== Tests (octonion algebra properties) =="
cargo test -p souc --features gpu --test integration_octonion_moufang -- --nocapture
cargo test -p souc --features gpu --test integration_octonion_numerical -- --nocapture

echo
echo "== Tests (GPU codegen presence checks; no GPU required) =="
cargo test -p souc --features gpu --test integration_octonion_basic -- --nocapture

echo
echo "== Tests (Moufang exhaustive validation) =="
cargo test -p souc --test integration_octonion_moufang_exhaustive -- --nocapture

echo
echo "== Tests (ONN training sanity) =="
cargo test -p souc --test integration_onn_training_sanity -- --nocapture

echo
echo "== Tests (MNIST toy training) =="
cargo test -p souc --test integration_onn_mnist_toy -- --nocapture

echo
echo "== Benchmarks (CPU microbenchmarks; Criterion) =="
echo "Running: cargo bench -p souc --bench octonion_benchmark -- --noplot"
cargo bench -p souc --bench octonion_benchmark -- --noplot

echo
echo "== Benchmarks (f32 baseline; Criterion) =="
echo "Running: cargo bench -p souc --bench octonion_benchmark -- --noplot f32_matmul_baseline"
cargo bench -p souc --bench octonion_benchmark -- --noplot "f32_matmul_baseline"

echo
echo "== Roofline CSV (from Criterion output) =="
python3 scripts/roofline_octonion_matmul.py \
  --criterion-dir target/criterion/octonion_matmul \
  --f32-criterion-dir target/criterion/f32_matmul_baseline \
  --out-csv docs/compiler/figures/octonion_matmul_points.csv

echo
echo "Outputs:"
echo "- target/criterion/* (raw Criterion estimates)"
echo "- target/criterion/f32_matmul_baseline/* (f32 baseline estimates)"
echo "- docs/compiler/figures/octonion_matmul_points.csv (roofline points)"

