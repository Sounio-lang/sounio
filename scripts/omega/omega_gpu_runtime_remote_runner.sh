#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

NONCE="${OMEGA_GPU_ATTEST_NONCE:-${1:-}}"
MANIFEST_B64="${OMEGA_GPU_ATTEST_MANIFEST_B64:-${2:-}}"
EXPECTED_VERSION="${SOUNIO_SOUC_VERSION:-unknown}"
BOOTSTRAPPED_SOUC="${OMEGA_GPU_ATTEST_BOOTSTRAPPED_SOUC:-}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
RESULTS_TSV="$TMP_DIR/results.tsv"

PROBE_LITERAL="tests/selfhost/gpu_runtime/test_gpu_runtime_literal.sio"
PROBE_TEXT="tests/selfhost/gpu_runtime/test_gpu_runtime_file_text.sio"
PROBE_BINARY="tests/selfhost/gpu_runtime/test_gpu_runtime_file_binary.sio"
PROBE_SLICE="tests/selfhost/gpu_runtime/test_gpu_runtime_dynamic_slice.sio"
GPU_COMPILE_FIXTURE="scripts/fixtures/gpu_minimal.sio"
GPU_RUNTIME_SMOKE="tests/run-pass/gpu_kernel_basic.sio"

append_result() {
  local name="$1"
  local status="$2"
  local rc="$3"
  local duration_ms="$4"
  local digest="$5"
  local excerpt="$6"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$status" "$rc" "$duration_ms" "$digest" "$excerpt" >> "$RESULTS_TSV"
}

sha256_file() {
  local path="$1"
  python3 - "$path" <<'PY'
import hashlib
from pathlib import Path
import sys
p = Path(sys.argv[1])
if not p.exists():
    print("")
    raise SystemExit(0)
print(hashlib.sha256(p.read_bytes()).hexdigest())
PY
}

probe_status_from_markers() {
  local out_path="$1"
  local start_marker="$2"
  local after_marker="$3"
  local check_rc="$4"
  local run_rc="$5"

  if [[ "$check_rc" -eq 125 && "$run_rc" -eq 125 ]]; then
    printf 'not_run\n'
    return
  fi
  if [[ "$check_rc" -ne 0 || "$run_rc" -ne 0 ]]; then
    printf 'fail\n'
    return
  fi
  if ! rg -q "$start_marker" "$out_path" 2>/dev/null; then
    printf 'fail\n'
    return
  fi
  if ! rg -q "$after_marker" "$out_path" 2>/dev/null; then
    printf 'fail\n'
    return
  fi
  printf 'pass\n'
}

run_probe() {
  local name="$1"
  local src="$2"
  local start_marker="$3"
  local after_marker="$4"
  local out_path="$TMP_DIR/${name}.out"
  local check_rc=125
  local run_rc=125

  local t0
  t0="$(python3 - <<'PY'
import time
print(int(time.time()*1000))
PY
)"

  if [[ -z "${SOUC_BIN:-}" || ! -x "${SOUC_BIN:-}" ]]; then
    append_result "$name" "not_run" "125" "0" "" "souc_unavailable"
    return
  fi

  set +e
  "$SOUC_BIN" check "$src" >"$out_path" 2>&1
  check_rc=$?
  if [[ $check_rc -eq 0 ]]; then
    "$SOUC_BIN" run "$src" >>"$out_path" 2>&1
    run_rc=$?
  fi
  set -e

  local status
  status="$(probe_status_from_markers "$out_path" "$start_marker" "$after_marker" "$check_rc" "$run_rc")"
  local rc="$run_rc"
  if [[ "$run_rc" -eq 125 ]]; then
    rc="$check_rc"
  fi

  local t1
  t1="$(python3 - <<'PY'
import time
print(int(time.time()*1000))
PY
)"
  local duration_ms=$((t1 - t0))

  local digest
  digest="$(sha256_file "$out_path")"
  local excerpt
  excerpt="$(head -n 4 "$out_path" | tr '\n' '|' | sed 's/|$//' | sed 's/[[:space:]]\+/ /g')"

  append_result "$name" "$status" "$rc" "$duration_ms" "$digest" "$excerpt"
}

run_gpu_compile_smoke() {
  local name="gpu_backend_compile_smoke"
  local out_path="$TMP_DIR/${name}.out"
  local rc=125

  local t0
  t0="$(python3 - <<'PY'
import time
print(int(time.time()*1000))
PY
)"

  if [[ -z "${SOUC_BIN:-}" || ! -x "${SOUC_BIN:-}" ]]; then
    append_result "$name" "not_run" "125" "0" "" "souc_unavailable"
    return
  fi

  set +e
  "$SOUC_BIN" build "$GPU_COMPILE_FIXTURE" --backend gpu -o "$TMP_DIR/gpu_minimal.ptx" >"$out_path" 2>&1
  rc=$?
  if [[ $rc -eq 0 ]]; then
    test -s "$TMP_DIR/gpu_minimal.ptx" >>"$out_path" 2>&1
    rc=$?
  fi
  if [[ $rc -eq 0 ]]; then
    rg -q "\\.entry" "$TMP_DIR/gpu_minimal.ptx" >>"$out_path" 2>&1
    rc=$?
  fi
  set -e

  local status="fail"
  if [[ $rc -eq 0 ]]; then
    status="pass"
  fi

  local t1
  t1="$(python3 - <<'PY'
import time
print(int(time.time()*1000))
PY
)"
  local duration_ms=$((t1 - t0))
  local digest
  digest="$(sha256_file "$out_path")"
  local excerpt
  excerpt="$(head -n 4 "$out_path" | tr '\n' '|' | sed 's/|$//' | sed 's/[[:space:]]\+/ /g')"

  append_result "$name" "$status" "$rc" "$duration_ms" "$digest" "$excerpt"
}

run_gpu_runtime_smoke() {
  local name="gpu_kernel_basic_runtime_smoke"
  local out_path="$TMP_DIR/${name}.out"
  local check_rc=125
  local run_rc=125

  local t0
  t0="$(python3 - <<'PY'
import time
print(int(time.time()*1000))
PY
)"

  if [[ -z "${SOUC_BIN:-}" || ! -x "${SOUC_BIN:-}" ]]; then
    append_result "$name" "not_run" "125" "0" "" "souc_unavailable"
    return
  fi

  set +e
  "$SOUC_BIN" check "$GPU_RUNTIME_SMOKE" >"$out_path" 2>&1
  check_rc=$?
  if [[ $check_rc -eq 0 ]]; then
    "$SOUC_BIN" run "$GPU_RUNTIME_SMOKE" >>"$out_path" 2>&1
    run_rc=$?
  fi
  set -e

  local status="fail"
  if [[ $check_rc -eq 125 && $run_rc -eq 125 ]]; then
    status="not_run"
  elif [[ $check_rc -eq 0 && $run_rc -eq 0 ]]; then
    status="pass"
  fi

  local rc="$run_rc"
  if [[ $run_rc -eq 125 ]]; then
    rc="$check_rc"
  fi

  local t1
  t1="$(python3 - <<'PY'
import time
print(int(time.time()*1000))
PY
)"
  local duration_ms=$((t1 - t0))
  local digest
  digest="$(sha256_file "$out_path")"
  local excerpt
  excerpt="$(head -n 4 "$out_path" | tr '\n' '|' | sed 's/|$//' | sed 's/[[:space:]]\+/ /g')"

  append_result "$name" "$status" "$rc" "$duration_ms" "$digest" "$excerpt"
}

SOUC_BIN=""
SOUC_VERSION="unknown"
RESOLVER_STATUS="fail"
RESOLVER_LOG="$TMP_DIR/resolver.log"

if [[ -n "$BOOTSTRAPPED_SOUC" ]]; then
  if [[ "$BOOTSTRAPPED_SOUC" = /* ]]; then
    candidate_souc="$BOOTSTRAPPED_SOUC"
  else
    candidate_souc="$ROOT_DIR/$BOOTSTRAPPED_SOUC"
  fi
  if [[ -x "$candidate_souc" ]]; then
    SOUC_BIN="$candidate_souc"
    RESOLVER_STATUS="bootstrapped"
  fi
fi

if [[ -z "$SOUC_BIN" && -x "$ROOT_DIR/scripts/omega/omega_resolve_souc_bin.sh" ]]; then
  set +e
  SOUC_BIN="$(SOUNIO_SOUC_VERSION="$EXPECTED_VERSION" OMEGA_SOUC_REQUIRE_PINNED=1 OMEGA_SOUC_ALLOW_LOCAL_FALLBACK=1 "$ROOT_DIR/scripts/omega/omega_resolve_souc_bin.sh" --print-path 2>"$RESOLVER_LOG")"
  resolver_rc=$?
  set -e
  if [[ $resolver_rc -eq 0 && -x "$SOUC_BIN" ]]; then
    RESOLVER_STATUS="pass"
  fi
fi

if [[ -z "$SOUC_BIN" || ! -x "$SOUC_BIN" ]]; then
  if command -v souc >/dev/null 2>&1; then
    SOUC_BIN="$(command -v souc)"
    RESOLVER_STATUS="fallback_local"
  fi
fi

if [[ -n "$SOUC_BIN" && -x "$SOUC_BIN" ]]; then
  SOUC_VERSION="$($SOUC_BIN --version 2>/dev/null | awk 'NR==1 {print $2}')"
  if [[ -z "$SOUC_VERSION" ]]; then
    SOUC_VERSION="unknown"
  fi
fi

GPU_INFO="nvidia-smi-unavailable"
if command -v nvidia-smi >/dev/null 2>&1; then
  set +e
  GPU_INFO="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n 4 | tr '\n' ';' | sed 's/;*$//')"
  gpu_info_rc=$?
  set -e
  if [[ $gpu_info_rc -ne 0 || -z "$GPU_INFO" ]]; then
    set +e
    GPU_INFO="$(nvidia-smi 2>/dev/null | head -n 8 | tr '\n' ';' | sed 's/;*$//')"
    gpu_info_fallback_rc=$?
    set -e
    if [[ $gpu_info_fallback_rc -ne 0 || -z "$GPU_INFO" ]]; then
      GPU_INFO="nvidia-smi-present-no-data"
    fi
  fi
fi

run_probe "gpu_runtime_literal" "$PROBE_LITERAL" "GPU_RUNTIME_LITERAL_START" "GPU_RUNTIME_LITERAL_AFTER"
run_probe "gpu_runtime_file_text" "$PROBE_TEXT" "GPU_RUNTIME_FILE_TEXT_START" "GPU_RUNTIME_FILE_TEXT_AFTER"
run_probe "gpu_runtime_file_binary" "$PROBE_BINARY" "GPU_RUNTIME_FILE_BINARY_START" "GPU_RUNTIME_FILE_BINARY_AFTER"
run_probe "gpu_runtime_dynamic_slice" "$PROBE_SLICE" "GPU_RUNTIME_SLICE_START" "GPU_RUNTIME_SLICE_AFTER"
run_gpu_compile_smoke
run_gpu_runtime_smoke

python3 - "$ROOT_DIR" "$RESULTS_TSV" "$NONCE" "$MANIFEST_B64" "$SOUC_BIN" "$SOUC_VERSION" "$EXPECTED_VERSION" "$RESOLVER_STATUS" "$GPU_INFO" <<'PY'
import base64
import datetime
import hashlib
import json
from pathlib import Path
import socket
import sys

root = Path(sys.argv[1]).resolve()
results_tsv = Path(sys.argv[2]).resolve()
nonce = sys.argv[3]
manifest_b64 = sys.argv[4]
souc_bin = sys.argv[5]
souc_version = sys.argv[6]
expected_version = sys.argv[7]
resolver_status = sys.argv[8]
gpu_info = sys.argv[9]

def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

manifest = {}
if manifest_b64:
    try:
        manifest = json.loads(base64.b64decode(manifest_b64.encode("ascii")).decode("utf-8"))
    except Exception:
        manifest = {}

expected_probe_hashes = manifest.get("probe_hashes") if isinstance(manifest.get("probe_hashes"), dict) else {}

probe_hashes_local = {}
probe_hash_mismatches = []
for rel, expected in expected_probe_hashes.items():
    p = (root / rel).resolve()
    if not p.exists():
        probe_hash_mismatches.append({"path": rel, "reason": "missing_remote_file", "expected": expected, "actual": ""})
        continue
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    probe_hashes_local[rel] = actual
    if actual != expected:
        probe_hash_mismatches.append({"path": rel, "reason": "hash_mismatch", "expected": expected, "actual": actual})

rows = []
if results_tsv.exists():
    for raw in results_tsv.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip():
            continue
        parts = raw.split("\t")
        if len(parts) != 6:
            continue
        name, status, rc, duration_ms, digest, excerpt = parts
        rows.append(
            {
                "name": name,
                "status": status,
                "rc": int(rc),
                "duration_ms": int(duration_ms),
                "stdout_digest": digest,
                "error_excerpt": excerpt[:400],
            }
        )

blockers = []
if not nonce:
    blockers.append("remote_env_missing")
if not souc_bin:
    blockers.append("remote_env_missing")
if resolver_status == "fail":
    blockers.append("remote_env_missing")
if probe_hash_mismatches:
    blockers.append("attestation_invalid")
for row in rows:
    if row.get("name") == "gpu_backend_compile_smoke" and row.get("status") == "fail":
        excerpt = str(row.get("error_excerpt", "")).lower()
        if "gpu backend not enabled" in excerpt:
            blockers.append("gpu_backend_unavailable")
if any(r.get("status") == "fail" for r in rows):
    blockers.append("runtime_test_fail")

if rows and all(r.get("status") == "pass" for r in rows) and not blockers:
    status_summary = "pass"
elif not rows:
    status_summary = "not_run"
else:
    status_summary = "fail"

payload = {
    "schema": "sounio.omega.gpu_runtime_attestation.remote.v1",
    "generated_at_utc": now_utc(),
    "nonce": nonce,
    "remote": {
        "hostname": socket.gethostname(),
        "root_dir": str(root),
    },
    "souc_path": souc_bin,
    "souc_version": souc_version,
    "souc_version_expected": expected_version,
    "resolver_status": resolver_status,
    "gpu_info": gpu_info,
    "probe_hashes_local": probe_hashes_local,
    "probe_hash_mismatches": probe_hash_mismatches,
    "tests": rows,
    "blockers": sorted(set(blockers)),
    "status_summary": status_summary,
}
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
signature = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
wrapper = {
    "payload": payload,
    "signature_algo": "sha256-detached-v1",
    "signature": signature,
}
print(json.dumps(wrapper, separators=(",", ":"), ensure_ascii=True))
PY
