#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

OUT_DIR="${SOUNIO_NATIVE_V2_AARCH64_PREVIEW_DIR:-$(mktemp -d /tmp/sounio-native-v2-aarch64-preview.XXXXXX)}"
LOG_DIR="$OUT_DIR/logs"
ARTIFACT_DIR="$OUT_DIR/artifacts"
CHECK_LOG="$LOG_DIR/aarch64_macho_preview_emit.check.log"
RUN_LOG="$LOG_DIR/aarch64_macho_preview_emit.run.log"
SMOKE_SRC="tests/native-v2/aarch64_macho_preview_emit.sio"
SMOKE_BIN="artifacts/omega/native_backend_v2_scalar_smoke.aarch64-macos.bin"
COPIED_BIN="$ARTIFACT_DIR/native_backend_v2_scalar_smoke.aarch64-macos.bin"
SUMMARY_JSON="$ARTIFACT_DIR/summary.json"
SMOKE_BACKUP="$OUT_DIR/original-smoke-bin"
HAD_SMOKE_BIN=0

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR" artifacts/omega

if [[ -f "$SMOKE_BIN" ]]; then
  cp "$SMOKE_BIN" "$SMOKE_BACKUP"
  HAD_SMOKE_BIN=1
fi

restore_smoke_bin() {
  if [[ "$HAD_SMOKE_BIN" == "1" ]]; then
    cp "$SMOKE_BACKUP" "$SMOKE_BIN"
  else
    rm -f "$SMOKE_BIN"
  fi
}
trap restore_smoke_bin EXIT

portable_size() {
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

printf '[native-v2-aarch64-preview] souc=%s\n' "$SOUC_BIN"
printf '[native-v2-aarch64-preview] out=%s\n' "$OUT_DIR"

"$SOUC_BIN" check "$SMOKE_SRC" >"$CHECK_LOG" 2>&1

# The preview emitter is multimodule source.  Use the stable lean_single path
# until Madaros imported/native lowering passes the official seed witness.
SOUNIO_SOUC_ENGINE=lean_single "$SOUC_BIN" run "$SMOKE_SRC" >"$RUN_LOG" 2>&1

if [[ ! -f "$SMOKE_BIN" ]]; then
  echo "[native-v2-aarch64-preview] FAIL: missing smoke binary: $SMOKE_BIN" >&2
  tail -n 80 "$RUN_LOG" >&2 || true
  exit 1
fi

byte_len="$(portable_size "$SMOKE_BIN")"
if [[ "$byte_len" != "32768" ]]; then
  echo "[native-v2-aarch64-preview] FAIL: expected byte_len=32768 got=$byte_len" >&2
  exit 1
fi

magic="$(od -An -tx1 -N4 "$SMOKE_BIN" | tr -d ' \n')"
if [[ "$magic" != "cffaedfe" ]]; then
  echo "[native-v2-aarch64-preview] FAIL: expected Mach-O magic cffaedfe got=$magic" >&2
  exit 1
fi

cp "$SMOKE_BIN" "$COPIED_BIN"

cat >"$SUMMARY_JSON" <<EOF
{
  "schema": "sounio.native_v2_aarch64_preview_gate.v1",
  "status": "pass",
  "engine": "lean_single",
  "source": "$SMOKE_SRC",
  "output_path": "$COPIED_BIN",
  "byte_len": $byte_len,
  "magic": "$magic",
  "known_default_engine_blocker": "madaros_imported_native_lower_array_seed_segfault"
}
EOF

echo "[native-v2-aarch64-preview] PASS: Mach-O preview emitted byte_len=$byte_len magic=$magic"
