#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[native-v2-imported-captured-closure-boundary] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[native-v2-imported-captured-closure-boundary] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

if [[ -n "${SOUC_BIN:-}" && "$SOUC_BIN" != "$ROOT_DIR"/* ]]; then
  echo "[native-v2-imported-captured-closure-boundary] ignoring external SOUC_BIN outside this worktree: $SOUC_BIN"
  unset SOUC_BIN
fi

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

if [[ -n "${SOUNIO_GATE_STDLIB_PATH:-}" ]]; then
  export SOUNIO_STDLIB_PATH="$SOUNIO_GATE_STDLIB_PATH"
elif [[ -z "${SOUNIO_STDLIB_PATH:-}" || "$SOUNIO_STDLIB_PATH" != "$ROOT_DIR"/* ]]; then
  export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"
fi

OUT_DIR="${SOUNIO_NATIVE_V2_IMPORTED_CAPTURED_CLOSURE_BOUNDARY_DIR:-$(mktemp -d /tmp/sounio-native-v2-imported-captured-closure-boundary.XXXXXX)}"
LOG_DIR="$OUT_DIR/logs"
ARTIFACT_DIR="$OUT_DIR/artifacts"
PROGRAM="tests/selfhost/native_runtime/import_captured_closure_boundary_42.sio"
MODULE="tests/selfhost/native_runtime/import_captured_closure_boundary/mod.sio"
ELF="$ARTIFACT_DIR/import_captured_closure_boundary_42.native"
SUMMARY_JSON="$ARTIFACT_DIR/native_v2_imported_captured_closure_boundary.v1.json"

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

echo "[native-v2-imported-captured-closure-boundary] souc=$SOUC_BIN"
echo "[native-v2-imported-captured-closure-boundary] out=$OUT_DIR"
echo "[native-v2-imported-captured-closure-boundary] program=$PROGRAM"

if [[ ! -f "$PROGRAM" || ! -f "$MODULE" ]]; then
  echo "[native-v2-imported-captured-closure-boundary] FAIL: missing imported captured closure witness files" >&2
  exit 1
fi

run_log() {
  local name="$1"
  shift
  echo "[native-v2-imported-captured-closure-boundary] running $name"
  "$@" >"$LOG_DIR/$name.log" 2>&1
}

run_log imported_closure_boundary bash scripts/ci/native_v2_imported_closure_boundary_gate.sh

run_log ir_summary \
  bash scripts/lib/run_selfhost_fresh.sh "$SOUC_BIN" self-hosted/compiler/lean.sio -- --ir-summary "$PROGRAM"

if ! grep -q 'Merged IR:' "$LOG_DIR/ir_summary.log" ||
   ! grep -q 'souc-lean ir-summary: functions=3' "$LOG_DIR/ir_summary.log"; then
  echo "[native-v2-imported-captured-closure-boundary] FAIL: IR summary did not prove 3-function imported captured closure handoff" >&2
  cat "$LOG_DIR/ir_summary.log" >&2 || true
  exit 1
fi

run_log native_compile \
  bash scripts/lib/run_selfhost_fresh.sh "$SOUC_BIN" self-hosted/compiler/lean.sio -- --native-compile "$PROGRAM" -o "$ELF"

if grep -q 'native_prebundle:' "$LOG_DIR/native_compile.log"; then
  echo "[native-v2-imported-captured-closure-boundary] FAIL: direct native path used native_prebundle" >&2
  cat "$LOG_DIR/native_compile.log" >&2 || true
  exit 1
fi

if grep -q 'falling back to full IR path' "$LOG_DIR/native_compile.log"; then
  echo "[native-v2-imported-captured-closure-boundary] FAIL: compact imported path fell back to full IR" >&2
  cat "$LOG_DIR/native_compile.log" >&2 || true
  exit 1
fi

if ! grep -q 'module_native_driver: imported source uses modular IR path' "$LOG_DIR/native_compile.log" ||
   ! grep -q 'module_native_driver: imported source uses compact modular IR table path' "$LOG_DIR/native_compile.log" ||
   ! grep -q 'Merged IR: 3' "$LOG_DIR/native_compile.log"; then
  echo "[native-v2-imported-captured-closure-boundary] FAIL: native compile did not use imported compact modular IR path" >&2
  cat "$LOG_DIR/native_compile.log" >&2 || true
  exit 1
fi

if [[ ! -f "$ELF" ]]; then
  echo "[native-v2-imported-captured-closure-boundary] FAIL: native ELF not produced" >&2
  cat "$LOG_DIR/native_compile.log" >&2 || true
  exit 1
fi

chmod +x "$ELF" 2>/dev/null || true

if command -v file >/dev/null 2>&1; then
  file "$ELF" >"$LOG_DIR/import_captured_closure_boundary_42.file.txt"
  if ! grep -q 'ELF 64-bit LSB executable, x86-64' "$LOG_DIR/import_captured_closure_boundary_42.file.txt"; then
    echo "[native-v2-imported-captured-closure-boundary] FAIL: unexpected native artifact kind" >&2
    cat "$LOG_DIR/import_captured_closure_boundary_42.file.txt" >&2
    exit 1
  fi
fi

set +e
"$ELF" >"$LOG_DIR/import_captured_closure_boundary_42.stdout" 2>"$LOG_DIR/import_captured_closure_boundary_42.stderr"
runtime_rc=$?
set -e

if [[ "$runtime_rc" -ne 42 ]]; then
  echo "[native-v2-imported-captured-closure-boundary] FAIL: expected runtime exit 42, got $runtime_rc" >&2
  cat "$LOG_DIR/import_captured_closure_boundary_42.stdout" >&2 || true
  cat "$LOG_DIR/import_captured_closure_boundary_42.stderr" >&2 || true
  exit 1
fi

# Pure-bash summary JSON emitter (replaces python3 hashlib + json.dump heredoc).
elf_sha="$(sha256sum "$ELF" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$ELF" | awk '{print $1}')"
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"$ROOT_DIR/bin/kretikos" json-emit \
  --string "artifact_dir=$OUT_DIR" \
  --string "capture_lowering=shape_specialized_i64_factory_no_heap_env" \
  --bool   "captured_closure_environments=true" \
  --string "compiler_entrypoint=self-hosted/compiler/lean.sio" \
  --string "compiler_resolved=$SOUC_BIN" \
  --string "fallback_path=none" \
  --int    "functions=3" \
  --string "generated_at_utc=$ts" \
  --string "host_callback=none" \
  --string "imported_module=$MODULE" \
  --bool   "imported_prebundle_native=false" \
  --string "native_elf=$ELF" \
  --string "native_elf_sha256=$elf_sha" \
  --string "program=$PROGRAM" \
  --int    "runtime_exit=42" \
  --string "schema=sounio.native_v2_imported_captured_closure_boundary.v1" \
  --string "scope=single_i64_capture_imported_factory_only" \
  --string "status=pass" \
  --string "target=x86_64-linux" \
  --array-strings "unsupported=general_heap_closure_env|linear_captures|epistemic_captures|multi_capture_envs" \
  > "$SUMMARY_JSON"

echo "[native-v2-imported-captured-closure-boundary] PASS: imported captured i64 closure factory uses modular IR native driver directly"
echo "[native-v2-imported-captured-closure-boundary] summary=$SUMMARY_JSON"
