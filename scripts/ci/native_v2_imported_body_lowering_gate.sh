#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[native-v2-imported-body-lowering] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[native-v2-imported-body-lowering] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

if [[ -n "${SOUC_BIN:-}" && "$SOUC_BIN" != "$ROOT_DIR"/* ]]; then
  echo "[native-v2-imported-body-lowering] ignoring external SOUC_BIN outside this worktree: $SOUC_BIN"
  unset SOUC_BIN
fi

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

OUT_DIR="${SOUNIO_NATIVE_V2_IMPORTED_BODY_LOWERING_DIR:-$(mktemp -d /tmp/sounio-native-v2-imported-body-lowering.XXXXXX)}"
LOG_DIR="$OUT_DIR/logs"
ARTIFACT_DIR="$OUT_DIR/artifacts"
SUMMARY_JSON="$ARTIFACT_DIR/native_v2_imported_body_lowering.v1.json"

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

echo "[native-v2-imported-body-lowering] souc=$SOUC_BIN"
echo "[native-v2-imported-body-lowering] out=$OUT_DIR"

if grep -R -q 'SOUNIO_IMPORTED_HOF_ABI_V1' \
  self-hosted/compiler/module_frontend.sio \
  tests/selfhost/native_runtime/import_hof_abi_42.sio \
  tests/selfhost/native_runtime/import_hof_abi/mod.sio \
  tests/selfhost/native_runtime/import_body_lowering_42.sio \
  tests/selfhost/native_runtime/import_body_lowering/mod.sio; then
  echo "[native-v2-imported-body-lowering] FAIL: retired imported-HOF marker is still present" >&2
  exit 1
fi

if grep -Eq 'module_frontend_try_fold_imported_struct_array_i64_main|module_frontend_patch_main_constant_and_stubs' \
  self-hosted/compiler/module_frontend.sio; then
  echo "[native-v2-imported-body-lowering] FAIL: retired imported-core ABI whole-module patch symbol is present" >&2
  exit 1
fi

run_log() {
  local name="$1"
  shift
  echo "[native-v2-imported-body-lowering] running $name"
  "$@" >"$LOG_DIR/$name.log" 2>&1
}

run_case() {
  local name="$1"
  local program="$2"
  local expected_functions="$3"
  local elf="$ARTIFACT_DIR/${name}.native"

  if [[ ! -f "$program" ]]; then
    echo "[native-v2-imported-body-lowering] FAIL: missing program $program" >&2
    exit 1
  fi

  run_log "${name}.ir_summary" \
    bash scripts/lib/run_selfhost_fresh.sh "$SOUC_BIN" self-hosted/compiler/lean.sio -- --ir-summary "$program"

  if ! grep -q "Merged IR: ${expected_functions}" "$LOG_DIR/${name}.ir_summary.log" ||
     ! grep -q "souc-lean ir-summary: functions=${expected_functions}" "$LOG_DIR/${name}.ir_summary.log"; then
    echo "[native-v2-imported-body-lowering] FAIL: $name IR summary did not prove ${expected_functions}-function imported handoff" >&2
    cat "$LOG_DIR/${name}.ir_summary.log" >&2 || true
    exit 1
  fi

  run_log "${name}.native_compile" \
    bash scripts/lib/run_selfhost_fresh.sh "$SOUC_BIN" self-hosted/compiler/lean.sio -- --native-compile "$program" -o "$elf"

  if grep -q 'native_prebundle:' "$LOG_DIR/${name}.native_compile.log"; then
    echo "[native-v2-imported-body-lowering] FAIL: $name used native_prebundle" >&2
    cat "$LOG_DIR/${name}.native_compile.log" >&2 || true
    exit 1
  fi

  if ! grep -q 'module_native_driver: imported source uses modular IR path' "$LOG_DIR/${name}.native_compile.log" ||
     ! grep -q 'module_native_driver: imported source uses compact modular IR table path' "$LOG_DIR/${name}.native_compile.log" ||
     ! grep -q "Merged IR: ${expected_functions}" "$LOG_DIR/${name}.native_compile.log"; then
    echo "[native-v2-imported-body-lowering] FAIL: $name native compile did not use imported compact modular IR path" >&2
    cat "$LOG_DIR/${name}.native_compile.log" >&2 || true
    exit 1
  fi

  if grep -q 'falling back to full IR path' "$LOG_DIR/${name}.native_compile.log"; then
    echo "[native-v2-imported-body-lowering] FAIL: $name compact imported path fell back to full IR" >&2
    cat "$LOG_DIR/${name}.native_compile.log" >&2 || true
    exit 1
  fi

  if [[ ! -f "$elf" ]]; then
    echo "[native-v2-imported-body-lowering] FAIL: $name native ELF not produced" >&2
    cat "$LOG_DIR/${name}.native_compile.log" >&2 || true
    exit 1
  fi

  chmod +x "$elf" 2>/dev/null || true

  if command -v file >/dev/null 2>&1; then
    file "$elf" >"$LOG_DIR/${name}.file.txt"
    if ! grep -q 'ELF 64-bit LSB executable, x86-64' "$LOG_DIR/${name}.file.txt"; then
      echo "[native-v2-imported-body-lowering] FAIL: $name unexpected native artifact kind" >&2
      cat "$LOG_DIR/${name}.file.txt" >&2
      exit 1
    fi
  fi

  set +e
  "$elf" >"$LOG_DIR/${name}.stdout" 2>"$LOG_DIR/${name}.stderr"
  local runtime_rc=$?
  set -e

  if [[ "$runtime_rc" -ne 42 ]]; then
    echo "[native-v2-imported-body-lowering] FAIL: $name expected runtime exit 42, got $runtime_rc" >&2
    cat "$LOG_DIR/${name}.stdout" >&2 || true
    cat "$LOG_DIR/${name}.stderr" >&2 || true
    exit 1
  fi

  printf '%s\t%s\t%s\t%s\n' "$name" "$program" "$expected_functions" "$elf" >>"$ARTIFACT_DIR/cases.tsv"
}

: >"$ARTIFACT_DIR/cases.tsv"

run_case imported_core tests/selfhost/native_runtime/import_core_abi_42.sio 6
run_case imported_hof tests/selfhost/native_runtime/import_hof_abi_42.sio 6
run_case imported_mixed tests/selfhost/native_runtime/import_body_lowering_42.sio 7

python3 - "$SUMMARY_JSON" "$SOUC_BIN" "$OUT_DIR" "$ARTIFACT_DIR/cases.tsv" <<'PY'
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

summary_path = pathlib.Path(sys.argv[1])
souc_bin = sys.argv[2]
out_dir = pathlib.Path(sys.argv[3])
cases_path = pathlib.Path(sys.argv[4])

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

cases = []
for line in cases_path.read_text(encoding="utf-8").splitlines():
    name, program, functions, elf = line.split("\t")
    elf_path = pathlib.Path(elf)
    cases.append({
        "name": name,
        "program": program,
        "functions": int(functions),
        "native_elf": str(elf_path),
        "native_elf_sha256": sha256(elf_path),
        "runtime_exit": 42,
    })

payload = {
    "schema": "sounio.native_v2_imported_body_lowering.v1",
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "status": "pass",
    "compiler_resolved": souc_bin,
    "compiler_entrypoint": "self-hosted/compiler/lean.sio",
    "target": "x86_64-linux",
    "case_count": len(cases),
    "pass_count": len(cases),
    "fail_count": 0,
    "cases": cases,
    "fallback_path": "none",
    "host_callback": "none",
    "imported_prebundle_native": False,
    "imported_body_lowering": "compact_imported_body_lowering_v1_no_source_marker",
    "scope": "i64 imported summaries, small struct returns, named fn refs, fn-typed params/returns, no captures",
    "source_markers": 0,
    "artifact_dir": str(out_dir),
}
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "[native-v2-imported-body-lowering] PASS: imported body lowering v1 covers core, HOF, and mixed witnesses"
echo "[native-v2-imported-body-lowering] summary=$SUMMARY_JSON"
