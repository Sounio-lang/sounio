#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

MODE="${OMEGA_GPU_RUNTIME_GATE_MODE:-auto}"
OUT_JSON="${OMEGA_GPU_RUNTIME_GATE_OUT:-$ROOT_DIR/artifacts/omega/gpu_runtime_attest_gate.v1.json}"
LOG_PATH="${OMEGA_GPU_RUNTIME_GATE_LOG:-$ROOT_DIR/artifacts/omega/gpu_runtime_attest_gate.log}"
PARITY_ARTIFACT="${OMEGA_GPU_CODEGEN_PARITY_JSON:-$ROOT_DIR/artifacts/omega/gpu_codegen_parity.v1.json}"
BINARY_ATTEST_ARTIFACT="${OMEGA_GPU_BINARY_ATTEST_JSON:-$ROOT_DIR/artifacts/omega/gpu_binary_attestation.v1.json}"
REQUIRED_NATIVE_LANES="${OMEGA_REQUIRED_NATIVE_LANES:-onn,qnn,snn,spnn,quantnn,hyper_math,exceptional}"
GPU_HOST="${GPU_HOST:-10.100.100.215}"
GPU_USER="${GPU_USER:-demetrios}"
REMOTE_DIR="${REMOTE_DIR:-~/work/sounio}"
REMOTE_BOOTSTRAP="${OMEGA_GPU_RUNTIME_BOOTSTRAP:-1}"
REMOTE_BOOTSTRAPPED_SOUC_REL="${OMEGA_GPU_REMOTE_BOOTSTRAPPED_SOUC_REL:-artifacts/omega/souc-bin/souc-linux-x86_64}"
FRESHNESS_MAX_SEC="${OMEGA_GPU_ATTEST_FRESHNESS_MAX_SEC:-900}"
EXPECTED_VERSION_SOURCE_PATH="$ROOT_DIR/scripts/omega/omega_resolve_souc_bin.sh"
EXPECTED_VERSION_SOURCE="env:SOUNIO_SOUC_VERSION"
if [[ -n "${SOUNIO_SOUC_VERSION:-}" ]]; then
  EXPECTED_VERSION="${SOUNIO_SOUC_VERSION}"
else
  EXPECTED_VERSION="$(python3 - "$EXPECTED_VERSION_SOURCE_PATH" <<'PY'
import re
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.exists():
    print("")
    raise SystemExit(0)
src = path.read_text(encoding="utf-8", errors="replace")
match = re.search(r'SOUC_VERSION="\$\{SOUNIO_SOUC_VERSION:-([^}]+)\}"', src)
print(match.group(1) if match else "")
PY
)"
  if [[ -n "$EXPECTED_VERSION" ]]; then
    EXPECTED_VERSION_SOURCE="resolver_default"
  else
    EXPECTED_VERSION="1.0.0-beta.5"
    EXPECTED_VERSION_SOURCE="fallback_default"
  fi
fi
LOCAL_PINNED_SOUC_PATH=""
if [[ -x "$EXPECTED_VERSION_SOURCE_PATH" ]]; then
  set +e
  LOCAL_PINNED_SOUC_PATH="$(
    SOUNIO_SOUC_VERSION="$EXPECTED_VERSION" \
    OMEGA_SOUC_REQUIRE_PINNED=1 \
    OMEGA_SOUC_ALLOW_LOCAL_FALLBACK=1 \
      "$EXPECTED_VERSION_SOURCE_PATH" --print-path 2>>"$LOG_PATH"
  )"
  local_souc_rc=$?
  set -e
  if [[ $local_souc_rc -ne 0 || ! -x "$LOCAL_PINNED_SOUC_PATH" ]]; then
    LOCAL_PINNED_SOUC_PATH=""
  fi
fi

GPU_RUNTIME_LOCAL_SOUC_BIN="${OMEGA_GPU_RUNTIME_LOCAL_SOUC_BIN:-}"
if [[ -z "$GPU_RUNTIME_LOCAL_SOUC_BIN" ]]; then
  GPU_RUNTIME_LOCAL_SOUC_BIN="$LOCAL_PINNED_SOUC_PATH"
fi
if resolved_gpu_souc="$(sounio_resolve_gpu_souc "$ROOT_DIR/scripts/fixtures/gpu_minimal.sio" "$GPU_RUNTIME_LOCAL_SOUC_BIN" 2>/dev/null)"; then
  LOCAL_PINNED_SOUC_PATH="$resolved_gpu_souc"
fi

PROBE_FILES=(
  "tests/selfhost/gpu_runtime/test_gpu_runtime_literal.sio"
  "tests/selfhost/gpu_runtime/test_gpu_runtime_file_text.sio"
  "tests/selfhost/gpu_runtime/test_gpu_runtime_file_binary.sio"
  "tests/selfhost/gpu_runtime/test_gpu_runtime_dynamic_slice.sio"
)
REQUIRED_TESTS=(
  "gpu_runtime_literal"
  "gpu_runtime_file_text"
  "gpu_runtime_file_binary"
  "gpu_runtime_dynamic_slice"
  "gpu_backend_compile_smoke"
  "gpu_kernel_basic_runtime_smoke"
)
BOOTSTRAP_FILES=(
  "scripts/omega/omega_gpu_runtime_remote_runner.sh"
  "scripts/omega/omega_resolve_souc_bin.sh"
  "scripts/omega/omega_verify_release_souc.py"
  "scripts/fixtures/gpu_minimal.sio"
  "tests/selfhost/gpu_runtime/test_gpu_runtime_literal.sio"
  "tests/selfhost/gpu_runtime/test_gpu_runtime_file_text.sio"
  "tests/selfhost/gpu_runtime/test_gpu_runtime_file_binary.sio"
  "tests/selfhost/gpu_runtime/test_gpu_runtime_dynamic_slice.sio"
  "tests/run-pass/gpu_kernel_basic.sio"
  "tests/fixtures/fmri/tiny_real_motion.tsv"
  "tests/fixtures/fmri/tiny_real_slice_nifti1.nii"
  "keys/bootstrap_ed25519.pub"
)

case "$MODE" in
  auto|required|off) ;;
  *)
    echo "error: OMEGA_GPU_RUNTIME_GATE_MODE must be auto|required|off (got '$MODE')" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$OUT_JSON")" "$(dirname "$LOG_PATH")"

remote_bootstrap_sync() {
  local host="$1"
  local dir="$2"
  local -a ssh_opts=("${@:3}")
  local file

  for file in "${BOOTSTRAP_FILES[@]}"; do
    if [[ ! -f "$ROOT_DIR/$file" ]]; then
      return 12
    fi
  done

  if ! ssh "${ssh_opts[@]}" "$host" "mkdir -p ${dir}" >/dev/null 2>&1; then
    return 13
  fi

  if ! tar -cf - -C "$ROOT_DIR" "${BOOTSTRAP_FILES[@]}" 2>>"$LOG_PATH" | ssh "${ssh_opts[@]}" "$host" "mkdir -p ${dir} && tar -xf - -C ${dir}" >>"$LOG_PATH" 2>&1; then
    return 14
  fi

  if ! ssh "${ssh_opts[@]}" "$host" "cd ${dir} && test -f scripts/omega/omega_gpu_runtime_remote_runner.sh" >/dev/null 2>&1; then
    return 15
  fi

  if [[ -n "${LOCAL_PINNED_SOUC_PATH:-}" && -x "${LOCAL_PINNED_SOUC_PATH}" ]]; then
    if ! ssh "${ssh_opts[@]}" "$host" "mkdir -p ${dir}/artifacts/omega/souc-bin" >/dev/null 2>&1; then
      return 16
    fi
    if ! cat "${LOCAL_PINNED_SOUC_PATH}" | ssh "${ssh_opts[@]}" "$host" "cat > ${dir}/${REMOTE_BOOTSTRAPPED_SOUC_REL} && chmod +x ${dir}/${REMOTE_BOOTSTRAPPED_SOUC_REL}" >>"$LOG_PATH" 2>&1; then
      return 17
    fi
  fi

  return 0
}

emit_status_json() {
  local status="$1"
  local reason="$2"
  local signature_valid="$3"
  local souc_path="$4"
  local souc_version="$5"
  local remote_json="$6"
  local blockers_json="$7"
  local tests_json="$8"
  local gpu_info="$9"
  local nonce="${10}"
  local parity_artifact_path="$PARITY_ARTIFACT"
  local binary_attest_artifact_path="$BINARY_ATTEST_ARTIFACT"
  python3 - "$OUT_JSON" "$status" "$reason" "$MODE" "$GPU_HOST" "$GPU_USER" "$REMOTE_DIR" "$signature_valid" "$souc_path" "$souc_version" "$EXPECTED_VERSION" "$remote_json" "$blockers_json" "$tests_json" "$gpu_info" "$nonce" "$LOG_PATH" "$EXPECTED_VERSION_SOURCE" "$EXPECTED_VERSION_SOURCE_PATH" "$LOCAL_PINNED_SOUC_PATH" "$REMOTE_BOOTSTRAPPED_SOUC_REL" "$parity_artifact_path" "$binary_attest_artifact_path" <<'PY'
import datetime
import json
from pathlib import Path
import sys

out_path = Path(sys.argv[1])
status = sys.argv[2]
reason = sys.argv[3]
mode = sys.argv[4]
gpu_host = sys.argv[5]
gpu_user = sys.argv[6]
remote_dir = sys.argv[7]
signature_valid = sys.argv[8] == "true"
souc_path = sys.argv[9]
souc_version = sys.argv[10]
expected_version = sys.argv[11]
remote_json = sys.argv[12]
blockers_json = sys.argv[13]
tests_json = sys.argv[14]
gpu_info = sys.argv[15]
nonce = sys.argv[16]
log_path = Path(sys.argv[17])
expected_version_source = sys.argv[18]
expected_version_source_path = sys.argv[19]
local_pinned_souc_path = sys.argv[20]
remote_bootstrapped_souc_rel = sys.argv[21]
parity_artifact_path = Path(sys.argv[22]) if sys.argv[22] else None
binary_attest_artifact_path = Path(sys.argv[23]) if sys.argv[23] else None

try:
    remote_obj = json.loads(remote_json) if remote_json else {}
except Exception:
    remote_obj = {}
try:
    blockers = json.loads(blockers_json) if blockers_json else []
except Exception:
    blockers = []
try:
    tests = json.loads(tests_json) if tests_json else []
except Exception:
    tests = []

# CUDA toolchain provenance (ptxas/nvcc/driver). Captured on the remote GPU host
# by omega_capture_toolchain.sh and threaded in via OMEGA_GPU_TOOLCHAIN_JSON.
# Empty/not_captured when the host was unreachable — keeps the artifact comparable
# across toolkit upgrades (ptxas codegen + cuBLAS baselines change across versions).
import os
try:
    toolchain = json.loads(os.environ.get("OMEGA_GPU_TOOLCHAIN_JSON", "") or "{}")
    if not isinstance(toolchain, dict):
        toolchain = {}
except Exception:
    toolchain = {}
if not toolchain:
    toolchain = {"capture_status": "not_captured"}

parity_obj = {}
if parity_artifact_path is not None and parity_artifact_path.exists():
    try:
        parity_obj = json.loads(parity_artifact_path.read_text(encoding="utf-8"))
    except Exception:
        parity_obj = {}

binary_attest_obj = {}
if binary_attest_artifact_path is not None and binary_attest_artifact_path.exists():
    try:
        binary_attest_obj = json.loads(binary_attest_artifact_path.read_text(encoding="utf-8"))
    except Exception:
        binary_attest_obj = {}

target_profiles = []
for row in parity_obj.get("targets", []) if isinstance(parity_obj.get("targets"), list) else []:
    if not isinstance(row, dict):
        continue
    target_profiles.append({
        "lane": row.get("lane", ""),
        "target": row.get("target", ""),
        "binary_format": row.get("binary_format", ""),
        "status": row.get("status", ""),
        "reason": row.get("reason", ""),
        "output_path": row.get("output_path", ""),
        "output_sha256": row.get("output_sha256", ""),
    })

native_lanes = parity_obj.get("native_lanes", []) if isinstance(parity_obj.get("native_lanes"), list) else []

hash_chain = binary_attest_obj.get("hash_chain", {}) if isinstance(binary_attest_obj.get("hash_chain"), dict) else {}
binary_provenance_entries = binary_attest_obj.get("target_provenance", []) if isinstance(binary_attest_obj.get("target_provenance"), list) else []

obj = {
    "schema": "sounio.omega.gpu_runtime_attest_gate.v1",
    "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    "mode": mode,
    "status_summary": status,
    "remote": {
        "gpu_host": gpu_host,
        "gpu_user": gpu_user,
        "remote_dir": remote_dir,
        "attested": remote_obj,
    },
    "nonce": nonce,
    "souc_path": souc_path,
    "souc_version": souc_version,
    "souc_version_expected": expected_version,
    "version_provenance": {
        "expected_source": expected_version_source,
        "expected_source_path": expected_version_source_path,
        "resolved_expected_version": expected_version,
        "local_pinned_souc_path": local_pinned_souc_path,
        "remote_bootstrapped_souc_rel": remote_bootstrapped_souc_rel,
    },
    "signature_valid": signature_valid,
    "gpu_info": gpu_info,
    "toolchain": toolchain,
    "target_profiles": target_profiles,
    "native_lanes": native_lanes,
    "binary_provenance": {
        "parity_artifact": str(parity_artifact_path) if parity_artifact_path is not None else "",
        "binary_attest_artifact": str(binary_attest_artifact_path) if binary_attest_artifact_path is not None else "",
        "entries": binary_provenance_entries,
    },
    "hash_chain": hash_chain,
    "tests": tests,
    "blockers": blockers,
    "notes": [
        f"reason={reason}",
        f"log_path={str(log_path)}",
        "required_test_set=gpu_runtime_literal,gpu_runtime_file_text,gpu_runtime_file_binary,gpu_runtime_dynamic_slice,gpu_backend_compile_smoke,gpu_kernel_basic_runtime_smoke",
        f"expected_version_source={expected_version_source}",
        f"expected_version_source_path={expected_version_source_path}",
        f"resolved_expected_version={expected_version}",
        f"local_pinned_souc_path={local_pinned_souc_path}",
        f"remote_bootstrapped_souc_rel={remote_bootstrapped_souc_rel}",
        f"parity_artifact={str(parity_artifact_path) if parity_artifact_path is not None else ''}",
        f"binary_attest_artifact={str(binary_attest_artifact_path) if binary_attest_artifact_path is not None else ''}",
    ],
}
out_path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
PY
}

if [[ "$MODE" == "off" ]]; then
  emit_status_json "not_run" "gate_disabled" "false" "" "" "{}" "[]" "[]" "" ""
  echo "omega_gpu_runtime_attest_gate: status=not_run reason=gate_disabled report=$OUT_JSON"
  exit 0
fi

for f in "${PROBE_FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    emit_status_json "fail" "probe_file_missing" "false" "" "" "{}" "[\"remote_env_missing\"]" "[]" "" ""
    echo "error: missing required probe source: $f" >&2
    exit 2
  fi
done

NONCE="$(python3 - <<'PY'
import secrets
print(secrets.token_hex(16))
PY
)"

MANIFEST_B64="$(python3 - "$ROOT_DIR" "$NONCE" "$EXPECTED_VERSION" "${PROBE_FILES[@]}" "${REQUIRED_TESTS[@]}" <<'PY'
import base64
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
nonce = sys.argv[2]
expected_version = sys.argv[3]
probe_files = sys.argv[4:8]
required_tests = sys.argv[8:]
probe_hashes = {}
for rel in probe_files:
    p = (root / rel).resolve()
    probe_hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
obj = {
    "schema": "sounio.omega.gpu_runtime_attest_manifest.v1",
    "nonce": nonce,
    "souc_version_expected": expected_version,
    "probe_hashes": probe_hashes,
    "required_tests": required_tests,
}
payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
print(base64.b64encode(payload).decode("ascii"))
PY
)"

SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o StrictHostKeyChecking=no
)

if ! ssh "${SSH_OPTS[@]}" "${GPU_USER}@${GPU_HOST}" "echo ok" >/dev/null 2>&1; then
  if [[ "$MODE" == "required" ]]; then
    emit_status_json "fail" "ssh_unreachable" "false" "" "" "{}" "[\"ssh_unreachable\"]" "[]" "" "$NONCE"
    echo "omega_gpu_runtime_attest_gate: status=fail reason=ssh_unreachable report=$OUT_JSON" >&2
    exit 2
  fi
  emit_status_json "not_run" "ssh_unreachable" "false" "" "" "{}" "[\"ssh_unreachable\"]" "[]" "" "$NONCE"
  echo "omega_gpu_runtime_attest_gate: status=not_run reason=ssh_unreachable report=$OUT_JSON"
  exit 0
fi

# Host is reachable → capture CUDA toolchain provenance (ptxas/nvcc/driver) on the
# GPU host so the attestation records WHICH toolchain produced/ran the artifacts.
# Best-effort: failure leaves OMEGA_GPU_TOOLCHAIN_JSON unset (→ capture_status=not_captured).
TOOLCHAIN_CAPTURE_OUT="$(SSH_OPTS='-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no' "$ROOT_DIR/scripts/omega/omega_capture_toolchain.sh" "${GPU_USER}@${GPU_HOST}" 2>/dev/null || true)"
if [[ -n "$TOOLCHAIN_CAPTURE_OUT" ]]; then
  export OMEGA_GPU_TOOLCHAIN_JSON="$TOOLCHAIN_CAPTURE_OUT"
fi

if [[ "$REMOTE_BOOTSTRAP" == "1" ]]; then
  if ! remote_bootstrap_sync "${GPU_USER}@${GPU_HOST}" "${REMOTE_DIR}" "${SSH_OPTS[@]}"; then
    if [[ "$MODE" == "required" ]]; then
      emit_status_json "fail" "remote_bootstrap_failed" "false" "" "" "{}" "[\"remote_env_missing\"]" "[]" "" "$NONCE"
      echo "omega_gpu_runtime_attest_gate: status=fail reason=remote_bootstrap_failed report=$OUT_JSON" >&2
      exit 2
    fi
    emit_status_json "not_run" "remote_bootstrap_failed" "false" "" "" "{}" "[\"remote_env_missing\"]" "[]" "" "$NONCE"
    echo "omega_gpu_runtime_attest_gate: status=not_run reason=remote_bootstrap_failed report=$OUT_JSON"
    exit 0
  fi
fi

REMOTE_OUT="$(mktemp)"
trap 'rm -f "$REMOTE_OUT"' EXIT

set +e
ssh "${SSH_OPTS[@]}" "${GPU_USER}@${GPU_HOST}" \
  "cd ${REMOTE_DIR} && OMEGA_GPU_ATTEST_NONCE='${NONCE}' OMEGA_GPU_ATTEST_MANIFEST_B64='${MANIFEST_B64}' OMEGA_GPU_ATTEST_BOOTSTRAPPED_SOUC='${REMOTE_BOOTSTRAPPED_SOUC_REL}' SOUNIO_SOUC_VERSION='${EXPECTED_VERSION}' bash scripts/omega/omega_gpu_runtime_remote_runner.sh" \
  >"$REMOTE_OUT" 2>"$LOG_PATH"
remote_rc=$?
set -e

if [[ $remote_rc -ne 0 ]]; then
  if [[ "$MODE" == "required" ]]; then
    emit_status_json "fail" "remote_runner_failed" "false" "" "" "{}" "[\"remote_env_missing\"]" "[]" "" "$NONCE"
    echo "omega_gpu_runtime_attest_gate: status=fail reason=remote_runner_failed rc=$remote_rc report=$OUT_JSON" >&2
    exit 2
  fi
  emit_status_json "not_run" "remote_runner_failed" "false" "" "" "{}" "[\"remote_env_missing\"]" "[]" "" "$NONCE"
  echo "omega_gpu_runtime_attest_gate: status=not_run reason=remote_runner_failed rc=$remote_rc report=$OUT_JSON"
  exit 0
fi

VERDICT_FILE="$(mktemp)"
trap 'rm -f "$REMOTE_OUT" "$VERDICT_FILE"' EXIT

python3 - "$REMOTE_OUT" "$NONCE" "$EXPECTED_VERSION" "$FRESHNESS_MAX_SEC" "$PARITY_ARTIFACT" "$REQUIRED_NATIVE_LANES" "${REQUIRED_TESTS[@]}" >"$VERDICT_FILE" <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import sys

wrapper_path = Path(sys.argv[1]).resolve()
nonce = sys.argv[2]
expected_version = sys.argv[3]
max_age_sec = int(sys.argv[4])
parity_artifact = Path(sys.argv[5]).resolve()
required_native_lanes = [lane.strip() for lane in sys.argv[6].split(",") if lane.strip()]
required_tests = sys.argv[7:]

raw = wrapper_path.read_text(encoding="utf-8", errors="replace").strip()
wrapper = json.loads(raw)
payload = wrapper.get("payload") if isinstance(wrapper.get("payload"), dict) else {}
signature_algo = wrapper.get("signature_algo", "")
signature = wrapper.get("signature", "")

canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
expected_sig = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
signature_valid = signature_algo == "sha256-detached-v1" and signature == expected_sig

blockers = set(payload.get("blockers") or [])
status = payload.get("status_summary", "not_run")

generated = payload.get("generated_at_utc", "")
now = datetime.datetime.now(datetime.timezone.utc)
try:
    ts = datetime.datetime.fromisoformat(generated.replace("Z", "+00:00"))
    age = (now - ts).total_seconds()
    freshness_ok = age >= 0 and age <= max_age_sec
except Exception:
    freshness_ok = False

if not signature_valid or not freshness_ok:
    blockers.add("attestation_invalid")

if payload.get("nonce") != nonce:
    blockers.add("attestation_invalid")

if payload.get("souc_version_expected", "") != expected_version:
    blockers.add("pinned_version_mismatch")
if payload.get("souc_version", "") != expected_version:
    blockers.add("pinned_version_mismatch")

rows = payload.get("tests") if isinstance(payload.get("tests"), list) else []
by_name = {row.get("name", ""): row for row in rows if isinstance(row, dict)}
for test_name in required_tests:
    row = by_name.get(test_name)
    if row is None or row.get("status") != "pass":
        blockers.add("runtime_test_fail")

if status != "pass":
    blockers.add("runtime_test_fail")

native_lanes = []
if parity_artifact.exists():
    try:
        parity_obj = json.loads(parity_artifact.read_text(encoding="utf-8", errors="replace"))
        native_lanes = parity_obj.get("native_lanes") if isinstance(parity_obj.get("native_lanes"), list) else []
    except Exception:
        native_lanes = []
else:
    blockers.add("native_lane_missing")

native_by_lane = {}
for lane_row in native_lanes:
    if isinstance(lane_row, dict):
        lane = str(lane_row.get("lane", ""))
        if lane:
            native_by_lane[lane] = lane_row

def lane_blocker(status_text: str, reason_text: str, fallback_text: str) -> str:
    text = (reason_text or "").lower()
    if status_text == "not_run" or "missing" in text or "not_run" in text:
        return "native_lane_missing"
    if "compile" in text or "check_failed" in text:
        return "native_lane_compile_fail"
    if "parity" in text or "golden" in text:
        return "native_lane_parity_fail"
    if "perf" in text or "throughput" in text:
        return "native_lane_perf_regression"
    if fallback_text:
        return fallback_text
    return "native_lane_runtime_fail"

alias = {"hyper_math": "math"}
for lane in required_native_lanes:
    row = native_by_lane.get(lane)
    if row is None:
        mapped = alias.get(lane, lane)
        row = native_by_lane.get(mapped)
    if row is None:
        blockers.add("native_lane_missing")
        continue
    lane_status = str(row.get("status", "not_run"))
    if lane_status == "pass":
        continue
    blocker = lane_blocker(lane_status, str(row.get("reason", "")), str(row.get("blocker", "")))
    blockers.add(blocker)

final_status = "pass" if not blockers else "fail"

out = {
    "final_status": final_status,
    "signature_valid": signature_valid,
    "payload": payload,
    "blockers": sorted(blockers),
    "tests": rows,
    "native_lanes": native_lanes,
    "gpu_info": payload.get("gpu_info", ""),
    "souc_path": payload.get("souc_path", ""),
    "souc_version": payload.get("souc_version", ""),
}
print(json.dumps(out))
PY

VERDICT_JSON="$(cat "$VERDICT_FILE")"

FINAL_STATUS="$(python3 - "$VERDICT_JSON" <<'PY'
import json
import sys
obj = json.loads(sys.argv[1])
print(obj.get("final_status", "fail"))
PY
)"

SIGNATURE_VALID="$(python3 - "$VERDICT_JSON" <<'PY'
import json
import sys
obj = json.loads(sys.argv[1])
print("true" if obj.get("signature_valid") else "false")
PY
)"

BLOCKERS_JSON="$(python3 - "$VERDICT_JSON" <<'PY'
import json
import sys
obj = json.loads(sys.argv[1])
print(json.dumps(obj.get("blockers", []), separators=(",", ":")))
PY
)"

TESTS_JSON="$(python3 - "$VERDICT_JSON" <<'PY'
import json
import sys
obj = json.loads(sys.argv[1])
print(json.dumps(obj.get("tests", []), separators=(",", ":")))
PY
)"

ATTESTED_REMOTE_JSON="$(python3 - "$VERDICT_JSON" <<'PY'
import json
import sys
obj = json.loads(sys.argv[1])
payload = obj.get("payload", {})
remote = payload.get("remote", {})
print(json.dumps(remote, separators=(",", ":")))
PY
)"

GPU_INFO_VALUE="$(python3 - "$VERDICT_JSON" <<'PY'
import json
import sys
obj = json.loads(sys.argv[1])
print(obj.get("gpu_info", ""))
PY
)"

SOUC_PATH_VALUE="$(python3 - "$VERDICT_JSON" <<'PY'
import json
import sys
obj = json.loads(sys.argv[1])
print(obj.get("souc_path", ""))
PY
)"

SOUC_VERSION_VALUE="$(python3 - "$VERDICT_JSON" <<'PY'
import json
import sys
obj = json.loads(sys.argv[1])
print(obj.get("souc_version", ""))
PY
)"

if [[ "$FINAL_STATUS" == "pass" ]]; then
  emit_status_json "pass" "attestation_pass" "$SIGNATURE_VALID" "$SOUC_PATH_VALUE" "$SOUC_VERSION_VALUE" "$ATTESTED_REMOTE_JSON" "$BLOCKERS_JSON" "$TESTS_JSON" "$GPU_INFO_VALUE" "$NONCE"
  echo "omega_gpu_runtime_attest_gate: status=pass report=$OUT_JSON"
  exit 0
fi

if [[ "$MODE" == "auto" ]]; then
  emit_status_json "not_run" "attestation_non_pass_auto" "$SIGNATURE_VALID" "$SOUC_PATH_VALUE" "$SOUC_VERSION_VALUE" "$ATTESTED_REMOTE_JSON" "$BLOCKERS_JSON" "$TESTS_JSON" "$GPU_INFO_VALUE" "$NONCE"
  echo "omega_gpu_runtime_attest_gate: status=not_run reason=attestation_non_pass_auto report=$OUT_JSON"
  exit 0
fi

emit_status_json "fail" "attestation_failed" "$SIGNATURE_VALID" "$SOUC_PATH_VALUE" "$SOUC_VERSION_VALUE" "$ATTESTED_REMOTE_JSON" "$BLOCKERS_JSON" "$TESTS_JSON" "$GPU_INFO_VALUE" "$NONCE"
echo "omega_gpu_runtime_attest_gate: status=fail report=$OUT_JSON" >&2
exit 2
