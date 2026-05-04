#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

WORK_DIR="${WORK_DIR:-/tmp/sounio-rust-free-seed-proof}"
LOG_DIR="$WORK_DIR/logs"
RUN_LOG="$LOG_DIR/seed-runtime.log"
NO_RUST_MARKER_LOG="$LOG_DIR/no-rust-markers.log"
SEED_PATH="${SOUNIO_BOOTSTRAP_SEED_PATH:-bootstrap/seeds/sounio-bootstrap-linux-x86_64.sio.bin}"
SEED_SHA256_PATH="${SOUNIO_BOOTSTRAP_SEED_SHA256_PATH:-${SEED_PATH}.sha256}"
SEED_SIG_PATH="${SOUNIO_BOOTSTRAP_SEED_SIG_PATH:-${SEED_PATH}.sig}"
SELFHOST_TARGET="${SELFHOST_TARGET:-$ROOT_DIR/self-hosted/}"
SELFHOST_CANARY="${SELFHOST_CANARY:-$ROOT_DIR/self-hosted/compiler/lean.sio}"
ASSERT_NO_RUST_MARKERS_SCRIPT="${ASSERT_NO_RUST_MARKERS_SCRIPT:-$ROOT_DIR/scripts/ci/assert_no_rust_markers.sh}"
BOOTSTRAP_MANIFEST_METADATA_PATH="${BOOTSTRAP_MANIFEST_METADATA_PATH:-bootstrap/artifacts/manifest.v2.json}"
BOOTSTRAP_SEED_TRUSTED_KEY="${SOUNIO_BOOTSTRAP_SEED_TRUSTED_KEY:-}"

mkdir -p "$LOG_DIR"
rm -f "$RUN_LOG" "$NO_RUST_MARKER_LOG"

resolve_bootstrap_seed_trusted_key() {
  if [[ -n "$BOOTSTRAP_SEED_TRUSTED_KEY" ]]; then
    echo "$BOOTSTRAP_SEED_TRUSTED_KEY"
    return 0
  fi

  if [[ -f "$BOOTSTRAP_MANIFEST_METADATA_PATH" ]] && command -v python3 >/dev/null 2>&1; then
    local manifest_key
    manifest_key="$(
      python3 - "$BOOTSTRAP_MANIFEST_METADATA_PATH" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    payload = json.load(f)
print(payload.get("key_id", ""))
PY
    )"
    if [[ -n "$manifest_key" ]]; then
      echo "$manifest_key"
      return 0
    fi
  fi

  if [[ -f "$BOOTSTRAP_MANIFEST_METADATA_PATH" ]]; then
    local manifest_key
    manifest_key="$(
      sed -n 's/.*"key_id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$BOOTSTRAP_MANIFEST_METADATA_PATH" | head -n1
    )"
    if [[ -n "$manifest_key" ]]; then
      echo "$manifest_key"
      return 0
    fi
  fi

  echo "sounio-dev"
}

BOOTSTRAP_SEED_TRUSTED_KEY="$(resolve_bootstrap_seed_trusted_key)"

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
echo "seed_trusted_key=$BOOTSTRAP_SEED_TRUSTED_KEY"

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

if [ ! -f "$SELFHOST_CANARY" ]; then
  echo "error: missing self-host canary path: $SELFHOST_CANARY" >&2
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
  SOUNIO_BOOTSTRAP_SEED_TRUSTED_KEY="$BOOTSTRAP_SEED_TRUSTED_KEY" \
  "$SOUC_BIN" run "$SELFHOST_CANARY" -- --self-test >"$RUN_LOG" 2>&1; then
  echo "error: seed-only self-hosted runtime execution failed (see $RUN_LOG)" >&2
  cat "$RUN_LOG" >&2 || true
  exit 1
fi

if ! grep -q "souc-lean self-test: ok" "$RUN_LOG"; then
  if ! grep -q "Self-hosted compiler - Rust-free build" "$RUN_LOG" &&
     ! grep -q "SOUNIO SELF-HOSTED COMPILER v" "$RUN_LOG"; then
    echo "error: expected self-hosted runtime banner missing: $RUN_LOG" >&2
    cat "$RUN_LOG" >&2 || true
    exit 1
  fi
fi

if ! grep -q "souc-lean self-test: ok" "$RUN_LOG" &&
   ! grep -E -q 'compiler/main tests: [0-9]+/[0-9]+ passed' "$RUN_LOG"; then
  echo "error: expected self-test summary missing from runtime log: $RUN_LOG" >&2
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
