#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[native-v2-imported-core-abi] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[native-v2-imported-core-abi] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

if [[ -n "${SOUNIO_GATE_SOUC_BIN:-}" ]]; then
  export SOUC_BIN="$SOUNIO_GATE_SOUC_BIN"
elif [[ -n "${SOUC_BIN:-}" && "$SOUC_BIN" != "$ROOT_DIR"/* ]]; then
  echo "[native-v2-imported-core-abi] ignoring external SOUC_BIN outside this worktree: $SOUC_BIN"
  unset SOUC_BIN
fi

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

if [[ -n "${SOUNIO_GATE_STDLIB_PATH:-}" ]]; then
  export SOUNIO_STDLIB_PATH="$SOUNIO_GATE_STDLIB_PATH"
elif [[ -z "${SOUNIO_STDLIB_PATH:-}" || "$SOUNIO_STDLIB_PATH" != "$ROOT_DIR"/* ]]; then
  export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"
fi

OUT_DIR="${SOUNIO_NATIVE_V2_IMPORTED_CORE_ABI_DIR:-$(mktemp -d /tmp/sounio-native-v2-imported-core-abi.XXXXXX)}"
LOG_DIR="$OUT_DIR/logs"
ARTIFACT_DIR="$OUT_DIR/artifacts"
PROGRAM="tests/selfhost/native_runtime/import_core_abi_42.sio"
ELF="$ARTIFACT_DIR/import_core_abi_42.native"
SUMMARY_JSON="$ARTIFACT_DIR/native_v2_imported_core_abi.v2.json"

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

echo "[native-v2-imported-core-abi] souc=$SOUC_BIN"
echo "[native-v2-imported-core-abi] out=$OUT_DIR"
echo "[native-v2-imported-core-abi] program=$PROGRAM"

if [[ ! -f "$PROGRAM" ]]; then
  echo "[native-v2-imported-core-abi] FAIL: missing program $PROGRAM" >&2
  exit 1
fi

if grep -Eq 'module_frontend_try_fold_imported_struct_array_i64_main|module_frontend_patch_main_constant_and_stubs' \
  self-hosted/compiler/module_frontend.sio; then
  echo "[native-v2-imported-core-abi] FAIL: retired imported-core ABI whole-module patch symbol is present" >&2
  exit 1
fi

run_log() {
  local name="$1"
  shift
  echo "[native-v2-imported-core-abi] running $name"
  "$@" >"$LOG_DIR/$name.log" 2>&1
}

run_log frontend_convergence bash scripts/ci/native_v2_frontend_convergence_gate.sh

run_log imported_ir_summary \
  bash scripts/lib/run_selfhost_fresh.sh "$SOUC_BIN" self-hosted/compiler/lean.sio -- --ir-summary "$PROGRAM"

if ! grep -q 'Merged IR: 6' "$LOG_DIR/imported_ir_summary.log" ||
   ! grep -q 'souc-lean ir-summary: functions=6' "$LOG_DIR/imported_ir_summary.log"; then
  echo "[native-v2-imported-core-abi] FAIL: modular lean IR summary did not prove 6-function imported handoff" >&2
  cat "$LOG_DIR/imported_ir_summary.log" >&2 || true
  exit 1
fi

run_log native_compile \
  bash scripts/lib/run_selfhost_fresh.sh "$SOUC_BIN" self-hosted/compiler/lean.sio -- --native-compile "$PROGRAM" -o "$ELF"

if grep -q 'native_prebundle:' "$LOG_DIR/native_compile.log"; then
  echo "[native-v2-imported-core-abi] FAIL: direct native path used native_prebundle" >&2
  cat "$LOG_DIR/native_compile.log" >&2 || true
  exit 1
fi

if ! grep -q 'module_native_driver: imported source uses modular IR path' "$LOG_DIR/native_compile.log" ||
   ! grep -q 'Merged IR: 6' "$LOG_DIR/native_compile.log"; then
  echo "[native-v2-imported-core-abi] FAIL: native compile did not use imported modular IR path" >&2
  cat "$LOG_DIR/native_compile.log" >&2 || true
  exit 1
fi

if [[ ! -f "$ELF" ]]; then
  echo "[native-v2-imported-core-abi] FAIL: native ELF not produced" >&2
  cat "$LOG_DIR/native_compile.log" >&2 || true
  exit 1
fi

chmod +x "$ELF" 2>/dev/null || true

if command -v file >/dev/null 2>&1; then
  file "$ELF" >"$LOG_DIR/import_core_abi_42.file.txt"
  if ! grep -q 'ELF 64-bit LSB executable, x86-64' "$LOG_DIR/import_core_abi_42.file.txt"; then
    echo "[native-v2-imported-core-abi] FAIL: unexpected native artifact kind" >&2
    cat "$LOG_DIR/import_core_abi_42.file.txt" >&2
    exit 1
  fi
fi

set +e
"$ELF" >"$LOG_DIR/import_core_abi_42.stdout" 2>"$LOG_DIR/import_core_abi_42.stderr"
runtime_rc=$?
set -e

if [[ "$runtime_rc" -ne 42 ]]; then
  echo "[native-v2-imported-core-abi] FAIL: expected runtime exit 42, got $runtime_rc" >&2
  cat "$LOG_DIR/import_core_abi_42.stdout" >&2 || true
  cat "$LOG_DIR/import_core_abi_42.stderr" >&2 || true
  exit 1
fi

# Emit summary JSON via pure-Sounio kretikos json-emit (replaces python json.dump heredoc).
# Schema sounio.native_v2_imported_core_abi.v2. Args ordered alphabetically by key.
ELF_SHA256="$(sha256sum "$ELF" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$ELF" | awk '{print $1}')"
GENERATED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%S.%6NZ)"
"$ROOT_DIR/bin/kretikos" json-emit \
    --string "artifact_dir=$OUT_DIR" \
    --string "compiler_entrypoint=self-hosted/compiler/lean.sio" \
    --string "compiler_resolved=$SOUC_BIN" \
    --string "fallback_path=none" \
    --string "generated_at_utc=$GENERATED_AT_UTC" \
    --string "host_callback=none" \
    --bool   "imported_prebundle_native=false" \
    --string "native_elf=$ELF" \
    --string "native_elf_sha256=$ELF_SHA256" \
    --string "program=$PROGRAM" \
    --int    "runtime_exit=42" \
    --string "schema=sounio.native_v2_imported_core_abi.v2" \
    --int    "source_markers=0" \
    --string "status=pass" \
    --string "target=x86_64-linux" \
    > "$SUMMARY_JSON"

echo "[native-v2-imported-core-abi] PASS: imported core ABI uses modular IR native driver directly"
echo "[native-v2-imported-core-abi] summary=$SUMMARY_JSON"
