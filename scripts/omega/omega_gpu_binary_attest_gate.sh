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
      blockers: $blockers,
      log_path: $log_path
    }' >"$OUT_JSON"
}

if [[ "$MODE" == "off" ]]; then
  emit_status_json "not_run" "gate_disabled" "[]" "[]" '{"algorithm":"sha256-chain-v1","entries":[],"head":""}' '[]'
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

for lane in required_targets:
    row = rows_by_lane.get(lane)
    if row is None or row.get("status") != "pass":
        if "target_unavailable" not in blockers:
            blockers.append("target_unavailable")

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

reason = "attestation_pass"
if blockers:
    reason = blockers[0]

print(json.dumps({
    "blockers": blockers,
    "binaries": binaries,
    "provenance": provenance,
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

emit_status_json "$status" "$reason" "$blockers_json" "$binaries_json" "$hash_chain_json" "$provenance_json"

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
