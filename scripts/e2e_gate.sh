#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOUC_BIN="$ROOT_DIR/compiler/target/debug/souc"
EXAMPLE="$ROOT_DIR/examples/simple_test.sio"
GPU_FIXTURE="$ROOT_DIR/scripts/fixtures/gpu_minimal.sio"

OUT_DIR="$(mktemp -d -t sounio_e2e.XXXXXX)"
trap 'rm -rf "$OUT_DIR"' EXIT

echo "[e2e] native build + run"
(cd "$ROOT_DIR/compiler" && cargo build -q --bin souc)
"$SOUC_BIN" build "$EXAMPLE" --backend native -o "$OUT_DIR/simple_test_native"
"$OUT_DIR/simple_test_native"

if [[ "${SOUNIO_SKIP_LLVM:-}" != "1" ]]; then
  echo "[e2e] llvm build + run"
  (cd "$ROOT_DIR/compiler" && cargo build -q --features llvm --bin souc)
  "$SOUC_BIN" build "$EXAMPLE" --backend llvm -o "$OUT_DIR/simple_test_llvm"
  "$OUT_DIR/simple_test_llvm"
else
  echo "[e2e] llvm skipped (SOUNIO_SKIP_LLVM=1)"
fi

if [[ "${SOUNIO_SKIP_GPU:-}" != "1" ]]; then
  echo "[e2e] gpu compile-only"
  (cd "$ROOT_DIR/compiler" && cargo build -q --features gpu --bin souc)
  "$SOUC_BIN" build "$GPU_FIXTURE" --backend gpu -o "$OUT_DIR/gpu_minimal.ptx"
  test -s "$OUT_DIR/gpu_minimal.ptx"
  grep -q "\\.entry" "$OUT_DIR/gpu_minimal.ptx"
else
  echo "[e2e] gpu skipped (SOUNIO_SKIP_GPU=1)"
fi

echo "[e2e] ok"
