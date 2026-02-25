#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="${SOUC_BIN:-./target/debug/souc}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-rust-free-seed-proof}"
LOG_DIR="$WORK_DIR/logs"
RUN_LOG="$LOG_DIR/seed-runtime.log"
NO_RUST_MARKER_LOG="$LOG_DIR/no-rust-markers.log"
SEED_PATH="${SOUNIO_BOOTSTRAP_SEED_PATH:-bootstrap/seeds/sounio-bootstrap-linux-x86_64.sio.bin}"
SEED_SHA256_PATH="${SOUNIO_BOOTSTRAP_SEED_SHA256_PATH:-${SEED_PATH}.sha256}"
SEED_SIG_PATH="${SOUNIO_BOOTSTRAP_SEED_SIG_PATH:-${SEED_PATH}.sig}"
SELFHOST_TARGET="${SELFHOST_TARGET:-$ROOT_DIR/self-hosted/}"
ASSERT_NO_RUST_MARKERS_SCRIPT="${ASSERT_NO_RUST_MARKERS_SCRIPT:-$ROOT_DIR/scripts/assert_no_rust_markers.sh}"

mkdir -p "$LOG_DIR"
rm -f "$RUN_LOG" "$NO_RUST_MARKER_LOG"

fail_if_tool_exists() {
  local tool="$1"
  if command -v "$tool" >/dev/null 2>&1; then
    echo "error: expected rust-free environment, but '$tool' is available at $(command -v "$tool")" >&2
    exit 1
  fi
}

echo "RUST_FREE_SEED_RUNTIME_PROOF_START"
echo "souc_bin=$SOUC_BIN"
echo "seed_path=$SEED_PATH"
echo "work_dir=$WORK_DIR"

fail_if_tool_exists rustc
fail_if_tool_exists cargo
fail_if_tool_exists rustup

if [ ! -x "$SOUC_BIN" ]; then
  echo "error: missing executable compiler binary at $SOUC_BIN" >&2
  exit 1
fi

for required_file in "$SEED_PATH" "$SEED_SHA256_PATH" "$SEED_SIG_PATH"; do
  if [ ! -s "$required_file" ]; then
    echo "error: missing required seed file: $required_file" >&2
    exit 1
  fi
done

if [ ! -e "$SELFHOST_TARGET" ]; then
  echo "error: missing self-host target path: $SELFHOST_TARGET" >&2
  exit 1
fi

if [ ! -f "$ASSERT_NO_RUST_MARKERS_SCRIPT" ]; then
  echo "error: missing rust-marker assertion script: $ASSERT_NO_RUST_MARKERS_SCRIPT" >&2
  exit 1
fi

if ! env \
  SOUNIO_BOOTSTRAP_SEED_ENFORCE=1 \
  SOUNIO_BOOTSTRAP_SEED_PATH="$SEED_PATH" \
  SOUNIO_BOOTSTRAP_SEED_SHA256_PATH="$SEED_SHA256_PATH" \
  SOUNIO_BOOTSTRAP_SEED_SIG_PATH="$SEED_SIG_PATH" \
  "$SOUC_BIN" run "$SELFHOST_TARGET" -- version >"$RUN_LOG" 2>&1; then
  echo "error: seed-only self-hosted runtime execution failed (see $RUN_LOG)" >&2
  cat "$RUN_LOG" >&2 || true
  exit 1
fi

if ! grep -q "SELFHOST=seed schema=v1 event=bootstrap_seed status=ok" "$RUN_LOG"; then
  echo "error: expected bootstrap seed marker missing in runtime log: $RUN_LOG" >&2
  cat "$RUN_LOG" >&2 || true
  exit 1
fi

if ! grep -q "SELFHOST=run schema=v1 event=selfhost_preflight status=skipped .*reason=seed_enforced" "$RUN_LOG"; then
  echo "error: expected seed-enforced wrapper preflight skip marker missing: $RUN_LOG" >&2
  cat "$RUN_LOG" >&2 || true
  exit 1
fi

if ! grep -q "Self-hosted compiler - Rust-free build" "$RUN_LOG"; then
  echo "error: expected self-hosted rust-free runtime banner missing: $RUN_LOG" >&2
  cat "$RUN_LOG" >&2 || true
  exit 1
fi

if ! bash "$ASSERT_NO_RUST_MARKERS_SCRIPT" "$RUN_LOG" >"$NO_RUST_MARKER_LOG" 2>&1; then
  echo "error: rust marker leakage detected during rust-free seed runtime proof" >&2
  cat "$NO_RUST_MARKER_LOG" >&2 || true
  exit 1
fi

echo "RUST_FREE_SEED_RUNTIME_PROOF_DONE"
echo "runtime_log=$RUN_LOG"
echo "no_rust_markers_log=$NO_RUST_MARKER_LOG"
