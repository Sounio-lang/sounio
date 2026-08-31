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

if [[ -n "${SOUNIO_GATE_STDLIB_PATH:-}" ]]; then
  export SOUNIO_STDLIB_PATH="$SOUNIO_GATE_STDLIB_PATH"
elif [[ -z "${SOUNIO_STDLIB_PATH:-}" || "$SOUNIO_STDLIB_PATH" != "$ROOT_DIR"/* ]]; then
  export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"
fi

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

# Pure-bash summary JSON emitter (replaces python3 hashlib + json.dump heredoc).
# Reads cases.tsv (name\tprogram\tfunctions\telf), computes sha256 per ELF,
# builds cases array with keys sorted alphabetically, then emits summary JSON.
CASES_JSON="$(
  first=true
  while IFS=$'\t' read -r name program functions elf_path; do
    [[ -z "$name" ]] && continue
    elf_sha="$(sha256sum "$elf_path" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$elf_path" | awk '{print $1}')"
    [[ "$first" == true ]] && first=false || printf ','
    printf '{"functions":%d,"name":"%s","native_elf":"%s","native_elf_sha256":"%s","program":"%s","runtime_exit":42}' \
      "$functions" "$name" "$elf_path" "$elf_sha" "$program"
  done < "$ARTIFACT_DIR/cases.tsv"
)"
CASES_JSON="[$CASES_JSON]"
# `grep -c` prints 0 itself on an empty file, so `|| echo 0` would append a
# second 0 (a two-line count) -- `|| true` rescues the exit code only.
case_count="$(grep -c . "$ARTIFACT_DIR/cases.tsv" 2>/dev/null || true)"
# Plan floor: the three run_case calls above each append exactly one row or
# exit 1; anything else at summary time means the plan did not complete and
# the summary would certify a run that never happened.
EXPECTED_CASES=3
if [[ "$case_count" -ne "$EXPECTED_CASES" ]]; then
  echo "[native-v2-imported-body-lowering] FAIL: cases.tsv has $case_count rows, expected $EXPECTED_CASES -- the plan did not complete" >&2
  exit 1
fi
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"$ROOT_DIR/bin/kretikos" json-emit \
  --string "artifact_dir=$OUT_DIR" \
  --int    "case_count=$case_count" \
  --raw-json "cases=$CASES_JSON" \
  --string "compiler_entrypoint=self-hosted/compiler/lean.sio" \
  --string "compiler_resolved=$SOUC_BIN" \
  --int    "fail_count=0" \
  --string "fallback_path=none" \
  --string "generated_at_utc=$ts" \
  --string "host_callback=none" \
  --string "imported_body_lowering=compact_imported_body_lowering_v1_no_source_marker" \
  --bool   "imported_prebundle_native=false" \
  --int    "pass_count=$case_count" \
  --string "schema=sounio.native_v2_imported_body_lowering.v1" \
  --string "scope=i64 imported summaries, small struct returns, named fn refs, fn-typed params/returns, no captures" \
  --int    "source_markers=0" \
  --string "status=pass" \
  --string "target=x86_64-linux" \
  > "$SUMMARY_JSON"

echo "[native-v2-imported-body-lowering] PASS: imported body lowering v1 covers core, HOF, and mixed witnesses"
echo "[native-v2-imported-body-lowering] summary=$SUMMARY_JSON"
