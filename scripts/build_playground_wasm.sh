#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/website/public/wasm"
TARGET="wasm32-unknown-unknown"

if ! command -v wasm-bindgen >/dev/null 2>&1; then
  echo "wasm-bindgen CLI is required. Install with: cargo install wasm-bindgen-cli"
  exit 1
fi

if ! rustup target list --installed | grep -q "^${TARGET}$"; then
  echo "Installing Rust target ${TARGET}..."
  rustup target add "${TARGET}"
fi

echo "Building souc for ${TARGET}..."
cargo build -p souc --lib --target "${TARGET}" --release --features wasm

mkdir -p "${OUT_DIR}"
echo "Generating wasm-bindgen artifacts in ${OUT_DIR}..."
wasm-bindgen \
  --target web \
  --out-dir "${OUT_DIR}" \
  --out-name sounio_compiler \
  "${ROOT_DIR}/target/${TARGET}/release/sounio.wasm"

echo "Done:"
ls -1 "${OUT_DIR}"/sounio_compiler*
