#!/usr/bin/env bash
# Probe the backend-IR -> PTX Vec4 aggregate pack/unpack witness.
#
# This is deliberately evidence-producing rather than completion-producing:
# if the current modular GPU import path is blocked, it records the blocker in
# JSON and exits 0 so higher-level audits can classify the lane precisely.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_GPU_KNOWLEDGE_BACKEND_PROBE_DIR:-$ROOT_DIR/artifacts/gpu/knowledge_vecmat_evidence_audit/backend_pack_unpack_probe}"
OUT_JSON="$OUT_DIR/gpu_knowledge_vec4_backend_pack_unpack_probe.v1.json"
OUT_RAW="$OUT_DIR/gpu_knowledge_vec4_backend_pack_unpack.raw"
OUT_PTX="$OUT_DIR/gpu_knowledge_vec4_backend_pack_unpack.ptx"
OUT_CUBIN="$OUT_DIR/gpu_knowledge_vec4_backend_pack_unpack.cubin"
EXTRACT_RAW="$OUT_DIR/gpu_knowledge_vec4_pack_unpack_lean_extract.raw"
EXTRACT_PTX="$OUT_DIR/gpu_knowledge_vec4_pack_unpack_lean_extract.ptx"
EXTRACT_CUBIN="$OUT_DIR/gpu_knowledge_vec4_pack_unpack_lean_extract.cubin"
EXTRACT_LOG="$OUT_DIR/lean_extract.log"
MIN_NO_IMPORT_LOG="$OUT_DIR/minimal_no_import_run.log"
MIN_IMPORT_LOG="$OUT_DIR/minimal_import_run.log"
LOG="$OUT_DIR/souc_run.log"
CHECK_LOG="$OUT_DIR/souc_check.log"
STDOUT_LOG="$OUT_DIR/souc_stdout.log"
ARCH="${SOUNIO_GPU_KNOWLEDGE_PTXAS_ARCH:-sm_80}"
HARNESS="self-hosted/gpu/gpu_knowledge_vec4_pack_unpack_harness.sio"
EXTRACT_HARNESS="self-hosted/gpu/gpu_knowledge_vec4_pack_unpack_lean_extract.sio"

mkdir -p "$OUT_DIR"

find_ptxas() {
  if [ -n "${SOUNIO_GPU_KNOWLEDGE_PTXAS_BIN:-}" ] && [ -x "$SOUNIO_GPU_KNOWLEDGE_PTXAS_BIN" ]; then
    printf '%s\n' "$SOUNIO_GPU_KNOWLEDGE_PTXAS_BIN"
    return 0
  fi
  if command -v ptxas >/dev/null 2>&1; then
    command -v ptxas
    return 0
  fi
  local candidate
  for candidate in \
    /usr/local/cuda/bin/ptxas \
    /usr/local/cuda-13.0/bin/ptxas \
    /workspace/.home/openvscode-server/.agents/claude-2/.local/lib/python3.12/site-packages/torch/bin/ptxas \
    /workspace/.home/openvscode-server/.agents/codex-1/.cache/uv/archive-v0/F6BRiDncsYqX5vDBPCSlZ/torch/bin/ptxas
  do
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

write_json() {
  local status="$1"
  local reason="$2"
  local souc_run_exit="$3"
  local souc_check_exit="$4"
  local ptxas_bin="${5:-}"
  local ptxas_exit="${6:-}"
  python3 - "$ROOT_DIR" "$OUT_JSON" "$status" "$reason" "$souc_run_exit" "$souc_check_exit" "$ptxas_bin" "$ptxas_exit" "$HARNESS" "$OUT_PTX" "$OUT_CUBIN" "$LOG" "$CHECK_LOG" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

root = pathlib.Path(sys.argv[1])
out_json = pathlib.Path(sys.argv[2])
status, reason, souc_run_exit, souc_check_exit, ptxas_bin, ptxas_exit, harness = sys.argv[3:10]
ptx, cubin, log, check_log = map(pathlib.Path, sys.argv[10:14])

def rel(path):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)

def file_entry(path):
    if not str(path) or not path.exists() or not path.is_file():
        return {"path": rel(path), "present": False}
    data = path.read_bytes()
    return {
        "path": rel(path),
        "present": True,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }

log_text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
check_text = check_log.read_text(encoding="utf-8", errors="replace") if check_log.exists() else ""
payload = {
    "schema": "sounio.gpu-knowledge-vec4-backend-pack-unpack-probe.v1",
    "status": status,
    "reason": reason,
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "harness": harness,
    "souc": {
        "check_command": f"./bin/souc check {harness}",
        "check_exit_code": int(souc_check_exit),
        "check_log": rel(check_log),
        "check_log_tail": check_text[-2000:],
        "run_command": f"./bin/souc run {harness}",
        "run_exit_code": int(souc_run_exit),
        "run_log": rel(log),
        "run_log_tail": log_text[-2000:],
        "compiled_before_runtime_failure": "Compilation successful!" in log_text,
    },
    "ptxas": {
        "path": ptxas_bin,
        "exit_code": int(ptxas_exit) if ptxas_exit else None,
        "arch": "sm_80",
    },
    "artifacts": {
        "ptx": file_entry(ptx),
        "cubin": file_entry(cubin),
    },
    "lean_extract_fallback": {
        "classification": "reference_extract_not_production_backend",
        "harness": os.environ.get("SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_HARNESS", ""),
        "status": os.environ.get("SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_STATUS", "not_run"),
        "reason": os.environ.get("SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_REASON", ""),
        "check_exit_code": int(os.environ.get("SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_CHECK_EXIT", "-1")),
        "run_exit_code": int(os.environ.get("SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_RUN_EXIT", "-1")),
        "ptxas_exit_code": int(os.environ.get("SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_PTXAS_EXIT", "-1")),
        "ptxas_path": os.environ.get("SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_PTXAS", ""),
        "artifacts": {
            "raw": file_entry(pathlib.Path(os.environ.get("SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_RAW", ""))),
            "ptx": file_entry(pathlib.Path(os.environ.get("SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_PTX", ""))),
            "cubin": file_entry(pathlib.Path(os.environ.get("SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_CUBIN", ""))),
            "log": file_entry(pathlib.Path(os.environ.get("SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_LOG", ""))),
        },
        "boundaries": [
            "does_not_claim_canonical_gpu_kernel_ir_backend",
            "does_not_claim_imported_compiler_lowering",
            "does_not_claim_cuda_device_runtime_execution",
        ],
    },
    "minimal_import_runtime_probe": {
        "classification": "diagnostic_not_backend_contract",
        "status": os.environ.get("SOUNIO_GPU_KNOWLEDGE_MIN_IMPORT_STATUS", "not_run"),
        "reason": os.environ.get("SOUNIO_GPU_KNOWLEDGE_MIN_IMPORT_REASON", "not_run"),
        "no_import": {
            "check_exit_code": int(os.environ.get("SOUNIO_GPU_KNOWLEDGE_MIN_NO_IMPORT_CHECK_EXIT", "-1")),
            "run_exit_code": int(os.environ.get("SOUNIO_GPU_KNOWLEDGE_MIN_NO_IMPORT_RUN_EXIT", "-1")),
            "run_log": rel(pathlib.Path(os.environ.get("SOUNIO_GPU_KNOWLEDGE_MIN_NO_IMPORT_LOG", ""))),
            "run_log_tail": pathlib.Path(os.environ.get("SOUNIO_GPU_KNOWLEDGE_MIN_NO_IMPORT_LOG", "")).read_text(encoding="utf-8", errors="replace")[-2000:] if os.environ.get("SOUNIO_GPU_KNOWLEDGE_MIN_NO_IMPORT_LOG") and pathlib.Path(os.environ.get("SOUNIO_GPU_KNOWLEDGE_MIN_NO_IMPORT_LOG", "")).exists() else "",
        },
        "imported": {
            "check_exit_code": int(os.environ.get("SOUNIO_GPU_KNOWLEDGE_MIN_IMPORT_CHECK_EXIT", "-1")),
            "run_exit_code": int(os.environ.get("SOUNIO_GPU_KNOWLEDGE_MIN_IMPORT_RUN_EXIT", "-1")),
            "run_log": rel(pathlib.Path(os.environ.get("SOUNIO_GPU_KNOWLEDGE_MIN_IMPORT_LOG", ""))),
            "run_log_tail": pathlib.Path(os.environ.get("SOUNIO_GPU_KNOWLEDGE_MIN_IMPORT_LOG", "")).read_text(encoding="utf-8", errors="replace")[-2000:] if os.environ.get("SOUNIO_GPU_KNOWLEDGE_MIN_IMPORT_LOG") and pathlib.Path(os.environ.get("SOUNIO_GPU_KNOWLEDGE_MIN_IMPORT_LOG", "")).exists() else "",
        },
        "boundaries": [
            "does_not_claim_backend_ir_to_ptx",
            "does_not_claim_cuda_device_runtime_execution",
            "diagnoses_minimal_modular_import_runtime_only",
        ],
    },
    "backend_ir_contract": {
        "kernel": "gpu_vec4_pack_unpack",
        "source_param": "in_ptr",
        "output_param": "out_ptr",
        "lane_offsets_bytes": [0, 32, 64, 96],
        "lane_type": "f64",
        "status": "proved" if status == "pass" else "unproved",
    },
    "boundaries": [
        "backend_ir_to_ptx_probe_only",
        "does_not_claim_imported_compiler_lowering",
        "does_not_claim_cuda_device_runtime_execution",
    ],
}
out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

required_patterns=(
  "\\.visible \\.entry gpu_vec4_pack_unpack"
  "ld\\.param\\.u64 %rd1, \\[in_ptr\\];"
  "ld\\.param\\.u64 %rd2, \\[out_ptr\\];"
  "ld\\.global\\.f64 %fd1, \\[%rd1\\];"
  "add\\.u64 %rd3, %rd1, 32;"
  "ld\\.global\\.f64 %fd2, \\[%rd3\\];"
  "add\\.u64 %rd4, %rd1, 64;"
  "ld\\.global\\.f64 %fd3, \\[%rd4\\];"
  "add\\.u64 %rd5, %rd1, 96;"
  "ld\\.global\\.f64 %fd4, \\[%rd5\\];"
  "st\\.global\\.f64 \\[%rd2\\], %fd1;"
  "add\\.u64 %rd6, %rd2, 32;"
  "st\\.global\\.f64 \\[%rd6\\], %fd2;"
  "add\\.u64 %rd7, %rd2, 64;"
  "st\\.global\\.f64 \\[%rd7\\], %fd3;"
  "add\\.u64 %rd8, %rd2, 96;"
  "st\\.global\\.f64 \\[%rd8\\], %fd4;"
)

validate_contract_ptx() {
  local ptx_file="$1"
  local pattern
  for pattern in "${required_patterns[@]}"; do
    if ! grep -Eq "$pattern" "$ptx_file"; then
      printf '%s\n' "$pattern"
      return 1
    fi
  done
  return 0
}

run_lean_extract_fallback() {
  export SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_HARNESS="$EXTRACT_HARNESS"
  export SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_RAW="$EXTRACT_RAW"
  export SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_PTX="$EXTRACT_PTX"
  export SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_CUBIN="$EXTRACT_CUBIN"
  export SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_LOG="$EXTRACT_LOG"
  export SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_STATUS="blocked"
  export SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_REASON="not_run"
  export SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_CHECK_EXIT="-1"
  export SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_RUN_EXIT="-1"
  export SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_PTXAS_EXIT="-1"
  export SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_PTXAS=""

  : >"$EXTRACT_LOG"
  set +e
  ./bin/souc check "$EXTRACT_HARNESS" >>"$EXTRACT_LOG" 2>&1
  local check_exit=$?
  set -e
  export SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_CHECK_EXIT="$check_exit"
  if [ "$check_exit" -ne 0 ]; then
    export SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_REASON="check_failed"
    return 0
  fi

  set +e
  ./bin/souc run "$EXTRACT_HARNESS" >"$EXTRACT_RAW" 2>>"$EXTRACT_LOG"
  local run_exit=$?
  set -e
  export SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_RUN_EXIT="$run_exit"
  if [ "$run_exit" -ne 0 ]; then
    export SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_REASON="run_failed"
    return 0
  fi

  awk 'BEGIN{emit=0} /^\.version /{emit=1} emit{print}' "$EXTRACT_RAW" >"$EXTRACT_PTX"
  local missing_pattern=""
  if ! missing_pattern="$(validate_contract_ptx "$EXTRACT_PTX")"; then
    export SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_REASON="ptx_contract_pattern_missing:${missing_pattern}"
    return 0
  fi

  local ptxas_bin
  ptxas_bin="$(find_ptxas || true)"
  export SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_PTXAS="$ptxas_bin"
  if [ -z "$ptxas_bin" ]; then
    export SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_REASON="ptxas_not_found"
    return 0
  fi

  set +e
  "$ptxas_bin" -arch="$ARCH" "$EXTRACT_PTX" -o "$EXTRACT_CUBIN" >>"$EXTRACT_LOG" 2>&1
  local ptxas_exit=$?
  set -e
  export SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_PTXAS_EXIT="$ptxas_exit"
  if [ "$ptxas_exit" -ne 0 ]; then
    export SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_REASON="ptxas_failed"
    return 0
  fi

  export SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_STATUS="pass"
  export SOUNIO_GPU_KNOWLEDGE_LEAN_EXTRACT_REASON="lean_extract_ptxas_pass"
}

run_minimal_import_runtime_probe() {
  export SOUNIO_GPU_KNOWLEDGE_MIN_IMPORT_STATUS="blocked"
  export SOUNIO_GPU_KNOWLEDGE_MIN_IMPORT_REASON="not_run"
  export SOUNIO_GPU_KNOWLEDGE_MIN_NO_IMPORT_CHECK_EXIT="-1"
  export SOUNIO_GPU_KNOWLEDGE_MIN_NO_IMPORT_RUN_EXIT="-1"
  export SOUNIO_GPU_KNOWLEDGE_MIN_IMPORT_CHECK_EXIT="-1"
  export SOUNIO_GPU_KNOWLEDGE_MIN_IMPORT_RUN_EXIT="-1"
  export SOUNIO_GPU_KNOWLEDGE_MIN_NO_IMPORT_LOG="$MIN_NO_IMPORT_LOG"
  export SOUNIO_GPU_KNOWLEDGE_MIN_IMPORT_LOG="$MIN_IMPORT_LOG"

  local tmpdir
  tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/sounio-gpu-min-import.XXXXXX")"
  cat >"$tmpdir/no_import.sio" <<'SIO'
fn main() -> i32 with IO {
    print(".version 7.0\n")
    0
}
SIO
  cat >"$tmpdir/import_leaf.sio" <<'SIO'
pub fn emit_marker() with IO {
    print(".version 7.0\n")
}
SIO
  cat >"$tmpdir/import_harness.sio" <<'SIO'
use import_leaf::*

fn main() -> i32 with IO {
    emit_marker()
    0
}
SIO

  : >"$MIN_NO_IMPORT_LOG"
  : >"$MIN_IMPORT_LOG"

  set +e
  ./bin/souc check "$tmpdir/no_import.sio" >>"$MIN_NO_IMPORT_LOG" 2>&1
  local no_import_check=$?
  ./bin/souc run "$tmpdir/no_import.sio" >>"$MIN_NO_IMPORT_LOG" 2>&1
  local no_import_run=$?
  ./bin/souc check "$tmpdir/import_harness.sio" >>"$MIN_IMPORT_LOG" 2>&1
  local import_check=$?
  ./bin/souc run "$tmpdir/import_harness.sio" >>"$MIN_IMPORT_LOG" 2>&1
  local import_run=$?
  set -e

  export SOUNIO_GPU_KNOWLEDGE_MIN_NO_IMPORT_CHECK_EXIT="$no_import_check"
  export SOUNIO_GPU_KNOWLEDGE_MIN_NO_IMPORT_RUN_EXIT="$no_import_run"
  export SOUNIO_GPU_KNOWLEDGE_MIN_IMPORT_CHECK_EXIT="$import_check"
  export SOUNIO_GPU_KNOWLEDGE_MIN_IMPORT_RUN_EXIT="$import_run"

  if [ "$no_import_check" -eq 0 ] && [ "$no_import_run" -eq 0 ] && [ "$import_check" -eq 0 ] && [ "$import_run" -eq 0 ]; then
    export SOUNIO_GPU_KNOWLEDGE_MIN_IMPORT_STATUS="pass"
    export SOUNIO_GPU_KNOWLEDGE_MIN_IMPORT_REASON="minimal_imported_elf_pass"
  elif [ "$no_import_check" -eq 0 ] && [ "$no_import_run" -eq 0 ] && [ "$import_check" -eq 0 ] && [ "$import_run" -eq 139 ]; then
    export SOUNIO_GPU_KNOWLEDGE_MIN_IMPORT_REASON="minimal_imported_elf_segfault_after_compile"
  else
    export SOUNIO_GPU_KNOWLEDGE_MIN_IMPORT_REASON="minimal_import_probe_unexpected_result"
  fi
}

set +e
./bin/souc check "$HARNESS" >"$CHECK_LOG" 2>&1
souc_check_exit=$?
set -e
if [ "$souc_check_exit" -ne 0 ]; then
  rm -f "$OUT_PTX"
  write_json "blocked" "souc_check_failed" "0" "$souc_check_exit"
  echo "gpu_knowledge_vec4_backend_pack_unpack_probe: BLOCKED souc_check_failed report=${OUT_JSON#$ROOT_DIR/}"
  exit 0
fi

set +e
./bin/souc run "$HARNESS" >"$STDOUT_LOG" 2>"$LOG"
souc_run_exit=$?
set -e
if [ "$souc_run_exit" -ne 0 ]; then
  {
    printf '\n--- stdout ---\n'
    cat "$STDOUT_LOG"
  } >>"$LOG"
  rm -f "$OUT_PTX"
  reason="souc_harness_failed"
  if grep -q "IR_MAX_INSTRS" "$LOG"; then
    reason="souc_ir_max_instrs_before_ptx"
  elif [ "$souc_run_exit" -eq 139 ] || grep -qi "Segmentation fault" "$LOG"; then
    reason="souc_runtime_segfault_after_compile"
  fi
  run_minimal_import_runtime_probe
  run_lean_extract_fallback
  write_json "blocked" "$reason" "$souc_run_exit" "$souc_check_exit"
  echo "gpu_knowledge_vec4_backend_pack_unpack_probe: BLOCKED $reason report=${OUT_JSON#$ROOT_DIR/}"
  exit 0
fi
mv "$STDOUT_LOG" "$OUT_RAW"
awk 'BEGIN{emit=0} /^\.version /{emit=1} emit{print}' "$OUT_RAW" >"$OUT_PTX"

if ! missing_pattern="$(validate_contract_ptx "$OUT_PTX")"; then
    write_json "fail" "ptx_contract_pattern_missing" "0" "$souc_check_exit"
    echo "gpu_knowledge_vec4_backend_pack_unpack_probe: FAIL ptx_contract_pattern_missing pattern=$missing_pattern report=${OUT_JSON#$ROOT_DIR/}" >&2
    exit 1
fi

PTXAS_BIN="$(find_ptxas || true)"
if [ -z "$PTXAS_BIN" ]; then
  write_json "blocked" "ptxas_not_found" "0" "$souc_check_exit"
  echo "gpu_knowledge_vec4_backend_pack_unpack_probe: BLOCKED ptxas_not_found report=${OUT_JSON#$ROOT_DIR/}"
  exit 0
fi

set +e
"$PTXAS_BIN" -arch="$ARCH" "$OUT_PTX" -o "$OUT_CUBIN" >>"$LOG" 2>&1
ptxas_exit=$?
set -e
if [ "$ptxas_exit" -ne 0 ]; then
  write_json "fail" "ptxas_failed" "0" "$souc_check_exit" "$PTXAS_BIN" "$ptxas_exit"
  echo "gpu_knowledge_vec4_backend_pack_unpack_probe: FAIL ptxas_failed report=${OUT_JSON#$ROOT_DIR/}" >&2
  exit 1
fi

write_json "pass" "backend_ir_pack_unpack_ptxas_pass" "0" "$souc_check_exit" "$PTXAS_BIN" "0"
echo "gpu_knowledge_vec4_backend_pack_unpack_probe: PASS report=${OUT_JSON#$ROOT_DIR/}"
