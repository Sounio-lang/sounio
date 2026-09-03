#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: required dependency missing: jq" >&2
  exit 2
fi

MODE="${OMEGA_GPU_BINARY_ATTEST_MODE:-auto}"
OUT_JSON="${OMEGA_GPU_BINARY_ATTEST_OUT:-$ROOT_DIR/artifacts/omega/gpu_binary_attestation.v1.json}"
LOG_PATH="${OMEGA_GPU_BINARY_ATTEST_LOG:-$ROOT_DIR/artifacts/omega/gpu_binary_attestation.log}"
PARITY_JSON="${OMEGA_GPU_CODEGEN_PARITY_JSON:-$ROOT_DIR/artifacts/omega/gpu_codegen_parity.v1.json}"
REQUIRED_TARGETS="${OMEGA_GPU_BINARY_ATTEST_REQUIRE_TARGETS:-cuda,rocm}"

case "$MODE" in
  auto|required|off) ;;
  *)
    echo "error: OMEGA_GPU_BINARY_ATTEST_MODE must be auto|required|off (got '$MODE')" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$OUT_JSON")" "$(dirname "$LOG_PATH")"
: >"$LOG_PATH"

to_rel() {
  local path="$1"
  if [[ "$path" == "$ROOT_DIR/"* ]]; then
    printf '%s' "${path#$ROOT_DIR/}"
  else
    printf '%s' "$path"
  fi
}

emit_status_json() {
  local status="$1"
  local reason="$2"
  local blockers_json="$3"
  local binaries_json="$4"
  local hash_chain_json="$5"
  local provenance_json="$6"
  local native_lanes_json="$7"

  jq -cn \
    --arg generated_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg mode "$MODE" \
    --arg status "$status" \
    --arg reason "$reason" \
    --arg parity_json "$(to_rel "$PARITY_JSON")" \
    --arg log_path "$(to_rel "$LOG_PATH")" \
    --argjson blockers "$blockers_json" \
    --argjson binaries "$binaries_json" \
    --argjson hash_chain "$hash_chain_json" \
    --argjson provenance "$provenance_json" \
    --argjson native_lanes "$native_lanes_json" \
    --argjson toolchain "$TOOLCHAIN_JSON" \
    '{
      schema: "sounio.omega.gpu_binary_attestation.v1",
      generated_at_utc: $generated_at_utc,
      mode: $mode,
      status_summary: $status,
      reason: $reason,
      source_parity_artifact: $parity_json,
      binaries: $binaries,
      hash_chain: $hash_chain,
      target_provenance: $provenance,
      native_lanes: $native_lanes,
      toolchain: $toolchain,
      blockers: $blockers,
      log_path: $log_path
    }' >"$OUT_JSON"
}

# CUDA toolchain provenance of the host that assembled/hashed these binaries
# (ptxas determines CUBIN bytes → the hash_chain is only comparable across toolkit
# versions with the toolchain recorded). Honors a pre-set env (orchestrator may pass
# a remote capture); else captures locally; else not_captured. Never fails the gate.
TOOLCHAIN_JSON="${OMEGA_GPU_TOOLCHAIN_JSON:-}"
[[ -z "$TOOLCHAIN_JSON" ]] && TOOLCHAIN_JSON="$("$ROOT_DIR/scripts/omega/omega_capture_toolchain.sh" 2>/dev/null || true)"
[[ -z "$TOOLCHAIN_JSON" ]] && TOOLCHAIN_JSON='{"capture_status":"not_captured"}'

if [[ "$MODE" == "off" ]]; then
  emit_status_json "not_run" "gate_disabled" "[]" "[]" '{"algorithm":"sha256-chain-v1","entries":[],"head":""}' '[]' '[]'
  echo "omega_gpu_binary_attest_gate: status=not_run reason=gate_disabled report=$OUT_JSON"
  exit 0
fi

analysis_json="$(python3 - "$ROOT_DIR" "$PARITY_JSON" "$REQUIRED_TARGETS" <<'PY'
import csv
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
parity_path = Path(sys.argv[2])
required_targets = [x.strip() for x in sys.argv[3].split(',') if x.strip()]

blockers = []
binaries = []
provenance = []
hash_entries = []
head = ""

if not parity_path.exists():
    blockers.append("target_unavailable")
    print(json.dumps({
        "blockers": blockers,
        "binaries": binaries,
        "provenance": provenance,
        "hash_chain": {"algorithm": "sha256-chain-v1", "entries": hash_entries, "head": head},
        "reason": "parity_artifact_missing"
    }, separators=(",", ":")))
    raise SystemExit(0)

obj = json.loads(parity_path.read_text(encoding="utf-8", errors="replace"))
target_rows = obj.get("targets") if isinstance(obj.get("targets"), list) else []

rows_by_lane = {}
for row in target_rows:
    if not isinstance(row, dict):
        continue
    lane = str(row.get("lane", ""))
    if lane:
        rows_by_lane[lane] = row

def blocker_from_target_reason(reason: str) -> str:
    text = (reason or "").strip().lower()
    if text in {"isa_encode_unsupported", "binary_pack_fail", "driver_reject", "parity_fail", "attestation_invalid"}:
        return text
    if text in {"gpu_backend_unavailable", "target_unavailable", "souc_unavailable"}:
        return "target_unavailable"
    if "isa" in text or "encode" in text:
        return "isa_encode_unsupported"
    if "pack" in text or "binary" in text:
        return "binary_pack_fail"
    if "driver" in text or "runtime" in text or "launch" in text:
        return "driver_reject"
    if "parity" in text:
        return "parity_fail"
    return "target_unavailable"

for lane in required_targets:
    row = rows_by_lane.get(lane)
    if row is None:
        if "target_unavailable" not in blockers:
            blockers.append("target_unavailable")
        continue
    if row.get("status") != "pass":
        blocker = blocker_from_target_reason(str(row.get("reason", "")))
        if blocker not in blockers:
            blockers.append(blocker)

pass_rows = [row for row in target_rows if isinstance(row, dict) and row.get("status") == "pass"]
for row in pass_rows:
    rel = str(row.get("output_path", ""))
    lane = str(row.get("lane", ""))
    target = str(row.get("target", ""))
    binary_format = str(row.get("binary_format", ""))
    expected_sha = str(row.get("output_sha256", ""))

    if not rel:
        if "binary_pack_fail" not in blockers:
            blockers.append("binary_pack_fail")
        continue

    abs_path = (root / rel).resolve()
    if not abs_path.exists() or not abs_path.is_file():
        if "binary_pack_fail" not in blockers:
            blockers.append("binary_pack_fail")
        continue

    payload = abs_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha and expected_sha != digest and "attestation_invalid" not in blockers:
        blockers.append("attestation_invalid")

    binaries.append({
        "lane": lane,
        "target": target,
        "binary_format": binary_format,
        "path": rel,
        "size_bytes": len(payload),
        "sha256": digest,
    })
    provenance.append({
        "lane": lane,
        "target": target,
        "binary_format": binary_format,
        "artifact_path": rel,
        "sha256": digest,
    })

binaries.sort(key=lambda x: (x.get("lane", ""), x.get("target", "")))
for idx, item in enumerate(binaries):
    seed = f"{idx}:{item['lane']}:{item['target']}:{item['sha256']}"
    if idx == 0:
        chain_hash = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    else:
        chain_hash = hashlib.sha256((head + ":" + seed).encode("utf-8")).hexdigest()
    head = chain_hash
    hash_entries.append({
        "index": idx,
        "lane": item["lane"],
        "target": item["target"],
        "sha256": item["sha256"],
        "chain_hash": chain_hash,
    })

if not binaries and "target_unavailable" not in blockers:
    blockers.append("target_unavailable")

native_lanes = obj.get("native_lanes") if isinstance(obj.get("native_lanes"), list) else []

def native_blocker(status: str, reason: str, fallback: str) -> str:
    text = (reason or "").lower()
    if status == "not_run" or "missing" in text or "not_run" in text:
        return "native_lane_missing"
    if "compile" in text or "check_failed" in text:
        return "native_lane_compile_fail"
    if "parity" in text or "golden" in text:
        return "native_lane_parity_fail"
    if "perf" in text or "throughput" in text:
        return "native_lane_perf_regression"
    if fallback:
        return fallback
    return "native_lane_runtime_fail"

for lane_row in native_lanes:
    if not isinstance(lane_row, dict):
        continue
    if not lane_row.get("required", False):
        continue
    lane_status = str(lane_row.get("status", "not_run"))
    if lane_status == "pass":
        continue
    blocker = native_blocker(lane_status, str(lane_row.get("reason", "")), str(lane_row.get("blocker", "")))
    if blocker not in blockers:
        blockers.append(blocker)

reason = "attestation_pass"
if blockers:
    reason = blockers[0]

print(json.dumps({
    "blockers": blockers,
    "binaries": binaries,
    "provenance": provenance,
    "native_lanes": native_lanes,
    "hash_chain": {
        "algorithm": "sha256-chain-v1",
        "entries": hash_entries,
        "head": head,
    },
    "reason": reason,
}, separators=(",", ":")))
PY
)"

blockers_json="$(python3 - "$analysis_json" <<'PY'
import json
import sys
obj = json.loads(sys.argv[1])
print(json.dumps(obj.get("blockers", []), separators=(",", ":")))
PY
)"

binaries_json="$(python3 - "$analysis_json" <<'PY'
import json
import sys
obj = json.loads(sys.argv[1])
print(json.dumps(obj.get("binaries", []), separators=(",", ":")))
PY
)"

hash_chain_json="$(python3 - "$analysis_json" <<'PY'
import json
import sys
obj = json.loads(sys.argv[1])
print(json.dumps(obj.get("hash_chain", {}), separators=(",", ":")))
PY
)"

provenance_json="$(python3 - "$analysis_json" <<'PY'
import json
import sys
obj = json.loads(sys.argv[1])
print(json.dumps(obj.get("provenance", []), separators=(",", ":")))
PY
)"

native_lanes_json="$(python3 - "$analysis_json" <<'PY'
import json
import sys
obj = json.loads(sys.argv[1])
print(json.dumps(obj.get("native_lanes", []), separators=(",", ":")))
PY
)"

candidate_reason="$(python3 - "$analysis_json" <<'PY'
import json
import sys
obj = json.loads(sys.argv[1])
print(obj.get("reason", "attestation_failed"))
PY
)"

status="pass"
reason="attestation_pass"
if [[ "$blockers_json" != "[]" ]]; then
  if [[ "$MODE" == "required" ]]; then
    status="fail"
    reason="$candidate_reason"
  else
    status="not_run"
    reason="attestation_non_pass_auto"
  fi
fi

emit_status_json "$status" "$reason" "$blockers_json" "$binaries_json" "$hash_chain_json" "$provenance_json" "$native_lanes_json"

if [[ "$status" == "pass" ]]; then
  echo "omega_gpu_binary_attest_gate: status=pass report=$OUT_JSON"
  exit 0
fi
if [[ "$status" == "not_run" ]]; then
  echo "omega_gpu_binary_attest_gate: status=not_run reason=$reason report=$OUT_JSON"
  exit 0
fi

echo "omega_gpu_binary_attest_gate: status=fail reason=$reason report=$OUT_JSON" >&2
exit 2
