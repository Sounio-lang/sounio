#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[native-v2-imported-closure-boundary] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[native-v2-imported-closure-boundary] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

if [[ -n "${SOUC_BIN:-}" && "$SOUC_BIN" != "$ROOT_DIR"/* ]]; then
  echo "[native-v2-imported-closure-boundary] ignoring external SOUC_BIN outside this worktree: $SOUC_BIN"
  unset SOUC_BIN
fi

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

OUT_DIR="${SOUNIO_NATIVE_V2_IMPORTED_CLOSURE_BOUNDARY_DIR:-$(mktemp -d /tmp/sounio-native-v2-imported-closure-boundary.XXXXXX)}"
LOG_DIR="$OUT_DIR/logs"
ARTIFACT_DIR="$OUT_DIR/artifacts"
PROGRAM="tests/selfhost/native_runtime/import_closure_boundary_42.sio"
MODULE="tests/selfhost/native_runtime/import_closure_boundary/mod.sio"
ELF="$ARTIFACT_DIR/import_closure_boundary_42.native"
SUMMARY_JSON="$ARTIFACT_DIR/native_v2_imported_closure_boundary.v1.json"

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

echo "[native-v2-imported-closure-boundary] souc=$SOUC_BIN"
echo "[native-v2-imported-closure-boundary] out=$OUT_DIR"
echo "[native-v2-imported-closure-boundary] program=$PROGRAM"

if [[ ! -f "$PROGRAM" || ! -f "$MODULE" ]]; then
  echo "[native-v2-imported-closure-boundary] FAIL: missing imported closure witness files" >&2
  exit 1
fi

run_log() {
  local name="$1"
  shift
  echo "[native-v2-imported-closure-boundary] running $name"
  "$@" >"$LOG_DIR/$name.log" 2>&1
}

run_log imported_body_lowering bash scripts/ci/native_v2_imported_body_lowering_gate.sh
run_log hof_closure bash scripts/ci/native_v2_hof_closure_gate.sh

run_log ir_summary \
  bash scripts/lib/run_selfhost_fresh.sh "$SOUC_BIN" self-hosted/compiler/lean.sio -- --ir-summary "$PROGRAM"

if ! grep -q 'Merged IR:' "$LOG_DIR/ir_summary.log" ||
   ! grep -q 'souc-lean ir-summary: functions=' "$LOG_DIR/ir_summary.log"; then
  echo "[native-v2-imported-closure-boundary] FAIL: IR summary did not prove imported handoff" >&2
  cat "$LOG_DIR/ir_summary.log" >&2 || true
  exit 1
fi

functions_count="$(grep 'souc-lean ir-summary: functions=' "$LOG_DIR/ir_summary.log" | sed -E 's/.*functions=([0-9]+).*/\1/' | tail -1)"
if [[ -z "$functions_count" || "$functions_count" -lt 1 ]]; then
  echo "[native-v2-imported-closure-boundary] FAIL: could not parse IR function count" >&2
  cat "$LOG_DIR/ir_summary.log" >&2 || true
  exit 1
fi

run_log native_compile \
  bash scripts/lib/run_selfhost_fresh.sh "$SOUC_BIN" self-hosted/compiler/lean.sio -- --native-compile "$PROGRAM" -o "$ELF"

if grep -q 'native_prebundle:' "$LOG_DIR/native_compile.log"; then
  echo "[native-v2-imported-closure-boundary] FAIL: direct native path used native_prebundle" >&2
  cat "$LOG_DIR/native_compile.log" >&2 || true
  exit 1
fi

if grep -q 'falling back to full IR path' "$LOG_DIR/native_compile.log"; then
  echo "[native-v2-imported-closure-boundary] FAIL: compact imported path fell back to full IR" >&2
  cat "$LOG_DIR/native_compile.log" >&2 || true
  exit 1
fi

if ! grep -q 'module_native_driver: imported source uses modular IR path' "$LOG_DIR/native_compile.log" ||
   ! grep -q 'module_native_driver: imported source uses compact modular IR table path' "$LOG_DIR/native_compile.log"; then
  echo "[native-v2-imported-closure-boundary] FAIL: native compile did not use imported compact modular IR path" >&2
  cat "$LOG_DIR/native_compile.log" >&2 || true
  exit 1
fi

if [[ ! -f "$ELF" ]]; then
  echo "[native-v2-imported-closure-boundary] FAIL: native ELF not produced" >&2
  cat "$LOG_DIR/native_compile.log" >&2 || true
  exit 1
fi

chmod +x "$ELF" 2>/dev/null || true

if command -v file >/dev/null 2>&1; then
  file "$ELF" >"$LOG_DIR/import_closure_boundary_42.file.txt"
  if ! grep -q 'ELF 64-bit LSB executable, x86-64' "$LOG_DIR/import_closure_boundary_42.file.txt"; then
    echo "[native-v2-imported-closure-boundary] FAIL: unexpected native artifact kind" >&2
    cat "$LOG_DIR/import_closure_boundary_42.file.txt" >&2
    exit 1
  fi
fi

set +e
"$ELF" >"$LOG_DIR/import_closure_boundary_42.stdout" 2>"$LOG_DIR/import_closure_boundary_42.stderr"
runtime_rc=$?
set -e

if [[ "$runtime_rc" -ne 42 ]]; then
  echo "[native-v2-imported-closure-boundary] FAIL: expected runtime exit 42, got $runtime_rc" >&2
  cat "$LOG_DIR/import_closure_boundary_42.stdout" >&2 || true
  cat "$LOG_DIR/import_closure_boundary_42.stderr" >&2 || true
  exit 1
fi

python3 - "$SUMMARY_JSON" "$SOUC_BIN" "$PROGRAM" "$MODULE" "$ELF" "$OUT_DIR" "$functions_count" <<'PY'
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

summary_path = pathlib.Path(sys.argv[1])
souc_bin = sys.argv[2]
program = sys.argv[3]
module = sys.argv[4]
elf = pathlib.Path(sys.argv[5])
out_dir = pathlib.Path(sys.argv[6])
functions_count = int(sys.argv[7])

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

payload = {
    "schema": "sounio.native_v2_imported_closure_boundary.v1",
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "status": "pass",
    "compiler_resolved": souc_bin,
    "compiler_entrypoint": "self-hosted/compiler/lean.sio",
    "target": "x86_64-linux",
    "program": program,
    "imported_module": module,
    "native_elf": str(elf),
    "native_elf_sha256": sha256(elf),
    "runtime_exit": 42,
    "functions": functions_count,
    "fallback_path": "none",
    "host_callback": "none",
    "imported_prebundle_native": False,
    "scope": "noncapturing_closure_literals_only",
    "captured_closure_environments": False,
    "artifact_dir": str(out_dir),
}
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "[native-v2-imported-closure-boundary] PASS: imported noncapturing closure boundary uses modular IR native driver directly"
echo "[native-v2-imported-closure-boundary] summary=$SUMMARY_JSON"
