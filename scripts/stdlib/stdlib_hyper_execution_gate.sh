#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_JSON="${STDLIB_HYPER_STATUS_OUT:-$ROOT_DIR/artifacts/stdlib/stdlib_hyper_execution_status.v1.json}"
GOLDEN_JSON="${STDLIB_HYPER_GOLDEN:-$ROOT_DIR/tests/fixtures/hyper/pipeline_golden.v1.json}"
INVENTORY_JSON="${STDLIB_HYPER_INVENTORY_JSON:-$ROOT_DIR/artifacts/stdlib/stdlib_inventory.v1.json}"
NATIVE_LANE_MATRIX_JSON="${STDLIB_NATIVE_LANE_MATRIX_OUT:-$ROOT_DIR/artifacts/stdlib/native_lane_matrix.v1.json}"
MODE="${STDLIB_HYPER_GATE_MODE:-required}"

LANES=("nn" "onn" "qnn" "snn" "spnn" "quantnn" "math")

# shellcheck source=lib/resolve_souc.sh
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

REQUIRED_TESTS=(
  "tests/stdlib/nn/test_hyper_quaternion_e2e.sio"
  "tests/stdlib/onn/test_hyper_onn_e2e.sio"
  "tests/stdlib/qnn/test_hyper_qnn_e2e.sio"
  "tests/stdlib/snn/test_snn_e2e.sio"
  "tests/stdlib/spnn/test_spiking_e2e.sio"
  "tests/stdlib/quantnn/test_quantum_e2e.sio"
  "tests/stdlib/math/test_hyper_math_e2e.sio"
)

usage() {
  cat <<'USAGE'
Usage: bash scripts/stdlib_hyper_execution_gate.sh [--out-json PATH] [--golden PATH] [--lane-matrix-json PATH]

Runs strict STDLIB hyper execution gate:
  1) verifies no //@ ignore on required hyper tests
  2) executes required run-pass hyper tests
  3) parses HYPER_METRIC lines and compares against pinned golden
  4) emits artifacts/stdlib/stdlib_hyper_execution_status.v1.json
  5) emits artifacts/stdlib/native_lane_matrix.v1.json
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-json)
      OUT_JSON="$2"
      shift 2
      ;;
    --golden)
      GOLDEN_JSON="$2"
      shift 2
      ;;
    --lane-matrix-json)
      NATIVE_LANE_MATRIX_JSON="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "$MODE" in
  auto|required|off) ;;
  *)
    echo "error: STDLIB_HYPER_GATE_MODE must be auto|required|off (got '$MODE')" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$OUT_JSON")" "$(dirname "$NATIVE_LANE_MATRIX_JSON")"

emit_not_run() {
  local reason="$1"
  python3 - "$OUT_JSON" "$NATIVE_LANE_MATRIX_JSON" "$reason" "$MODE" <<'PY'
import datetime
import json
from pathlib import Path
import sys

out = Path(sys.argv[1])
lane_out = Path(sys.argv[2])
reason = sys.argv[3]
mode = sys.argv[4]
lanes = ["nn", "onn", "qnn", "snn", "spnn", "quantnn", "math"]

def lane_test_path(lane: str) -> str:
    if lane == "nn":
        return "tests/stdlib/nn/test_hyper_quaternion_e2e.sio"
    if lane == "onn":
        return "tests/stdlib/onn/test_hyper_onn_e2e.sio"
    if lane == "qnn":
        return "tests/stdlib/qnn/test_hyper_qnn_e2e.sio"
    if lane == "snn":
        return "tests/stdlib/snn/test_snn_e2e.sio"
    if lane == "spnn":
        return "tests/stdlib/spnn/test_spiking_e2e.sio"
    if lane == "quantnn":
        return "tests/stdlib/quantnn/test_quantum_e2e.sio"
    return "tests/stdlib/math/test_hyper_math_e2e.sio"

lane_rows = []
for lane in lanes:
    lane_rows.append({
        "lane": lane,
        "required": True,
        "test_path": lane_test_path(lane),
        "status": "not_run",
        "reason": reason,
        "blockers": ["native_lane_missing"],
        "metrics_status": "not_run",
    })

status_obj = {
    "schema": "sounio.stdlib.hyper_execution_status.v1",
    "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    "command": "bash scripts/stdlib/stdlib_hyper_execution_gate.sh",
    "mode": mode,
    "status_summary": "not_run",
    "totals": {"pass": 0, "fail": 0, "skip": 0, "total": 0},
    "inventory": {
        "hyper_active_files": 0,
        "hyper_disabled_files": 0,
        "hyper_stub_mod_files": 0,
    },
    "failures": [{"test_path": "", "category": "not_run", "error_excerpt": reason}],
    "golden_comparison": {"status": "not_run", "mismatches": []},
    "ignore_policy": {"status": "not_run", "violations": []},
    "native_lane_matrix_artifact": str(lane_out),
    "lane_statuses": lane_rows,
    "blockers": ["native_lane_missing"],
    "notes": [reason],
}

lane_obj = {
    "schema": "sounio.stdlib.native_lane_matrix.v1",
    "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    "mode": mode,
    "status_summary": "not_run",
    "lanes": lane_rows,
    "blockers": ["native_lane_missing"],
    "notes": [reason],
}

out.write_text(json.dumps(status_obj, indent=2) + "\n", encoding="utf-8")
lane_out.write_text(json.dumps(lane_obj, indent=2) + "\n", encoding="utf-8")
PY
}

if [[ "$MODE" == "off" ]]; then
  emit_not_run "gate_disabled"
  echo "[stdlib-hyper-gate] status_summary=not_run (gate_disabled)"
  exit 0
fi

for t in "${REQUIRED_TESTS[@]}"; do
  if [[ ! -f "$t" ]]; then
    emit_not_run "required_test_missing:$t"
    echo "error: required test missing: $t" >&2
    exit 1
  fi
done

if [[ ! -f "$GOLDEN_JSON" ]]; then
  emit_not_run "golden_missing"
  echo "error: missing hyper golden file: $GOLDEN_JSON" >&2
  exit 1
fi

sounio_require_souc
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
IGNORE_HITS="$TMP_DIR/ignore_hits.txt"
RESULTS_TSV="$TMP_DIR/results.tsv"
OUT_DIR="$TMP_DIR/out"
mkdir -p "$OUT_DIR"

rg -n '^[[:space:]]*//@[[:space:]]*ignore([[:space:]].*)?$' "${REQUIRED_TESTS[@]}" >"$IGNORE_HITS" || true

run_one() {
  local test_path="$1"
  local lane="$2"
  local marker="$3"
  local out_file="$OUT_DIR/${lane}.out"
  local check_rc=125
  local run_rc=125

  set +e
  "$SOUC_BIN" check "$test_path" >"$out_file" 2>&1
  check_rc=$?
  if [[ $check_rc -eq 0 ]]; then
    "$SOUC_BIN" run "$test_path" >>"$out_file" 2>&1
    run_rc=$?
  fi
  set -e

  local status="fail"
  local reason="run_failed"
  if [[ $check_rc -eq 125 && $run_rc -eq 125 ]]; then
    status="skip"
    reason="not_run"
  elif [[ $check_rc -ne 0 ]]; then
    status="fail"
    reason="check_failed"
  elif [[ $run_rc -ne 0 ]]; then
    status="fail"
    reason="run_failed"
  elif ! rg -q "$marker" "$out_file"; then
    status="fail"
    reason="missing_marker"
  else
    status="pass"
    reason="ok"
  fi

  local excerpt
  excerpt="$(head -n 4 "$out_file" | tr '\n' '|' | sed 's/|$//')"
  printf '%s\t%s\t%s\t%s\t%s\n' "$test_path" "$lane" "$status" "$reason" "$excerpt" >> "$RESULTS_TSV"
}

run_one "tests/stdlib/nn/test_hyper_quaternion_e2e.sio" "nn" "HYPER_NN_OK"
run_one "tests/stdlib/onn/test_hyper_onn_e2e.sio" "onn" "HYPER_ONN_OK"
run_one "tests/stdlib/qnn/test_hyper_qnn_e2e.sio" "qnn" "HYPER_QNN_OK"
run_one "tests/stdlib/snn/test_snn_e2e.sio" "snn" "HYPER_SNN_OK"
run_one "tests/stdlib/spnn/test_spiking_e2e.sio" "spnn" "HYPER_SPNN_OK"
run_one "tests/stdlib/quantnn/test_quantum_e2e.sio" "quantnn" "HYPER_QUANTNN_OK"
run_one "tests/stdlib/math/test_hyper_math_e2e.sio" "math" "HYPER_MATH_OK"

set +e
bash "$ROOT_DIR/scripts/stdlib/scan_stdlib.sh" --json-out "$INVENTORY_JSON" --quiet
scan_rc=$?
set -e
if [[ $scan_rc -ne 0 || ! -s "$INVENTORY_JSON" ]]; then
  emit_not_run "inventory_scan_failed"
  echo "error: stdlib inventory scan failed" >&2
  exit 1
fi

python3 - "$ROOT_DIR" "$OUT_JSON" "$RESULTS_TSV" "$OUT_DIR" "$GOLDEN_JSON" "$IGNORE_HITS" "$INVENTORY_JSON" "$MODE" "$NATIVE_LANE_MATRIX_JSON" <<'PY'
import datetime
import json
import math
import re
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
out_path = Path(sys.argv[2]).resolve()
results_tsv = Path(sys.argv[3]).resolve()
out_dir = Path(sys.argv[4]).resolve()
golden_path = Path(sys.argv[5]).resolve()
ignore_hits_path = Path(sys.argv[6]).resolve()
inventory_path = Path(sys.argv[7]).resolve()
mode = sys.argv[8]
lane_matrix_path = Path(sys.argv[9]).resolve()
lanes = ["nn", "onn", "qnn", "snn", "spnn", "quantnn", "math"]

rows = []
for line in results_tsv.read_text(encoding="utf-8", errors="replace").splitlines():
    if not line.strip():
        continue
    parts = line.split("\t")
    if len(parts) != 5:
        continue
    test_path, lane, status, reason, excerpt = parts
    rows.append({
        "test_path": test_path,
        "lane": lane,
        "status": status,
        "reason": reason,
        "excerpt": excerpt,
    })

ignore_hits = []
if ignore_hits_path.exists():
    ignore_hits = [line.strip() for line in ignore_hits_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]

lane_outputs = {}
for lane in lanes:
    p = out_dir / f"{lane}.out"
    lane_outputs[lane] = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""

metric_pattern = re.compile(r"HYPER_METRIC\s+(nn|onn|qnn|snn|spnn|quantnn|math)\s+([A-Za-z0-9_]+)\s+([-+0-9.eE]+)")
observed = {lane: {} for lane in lanes}
for lane, text in lane_outputs.items():
    for m in metric_pattern.findall(text):
        _, key, raw_val = m
        try:
            observed[lane][key] = float(raw_val)
        except Exception:
            pass

try:
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
except Exception:
    golden = {}

abs_tol = float((golden.get("tolerance") or {}).get("abs", 1e-6))
rel_tol = float((golden.get("tolerance") or {}).get("rel", 1e-6))

mismatches = []
for lane in lanes:
    exp_metrics = ((golden.get(lane) or {}).get("metrics") or {})
    obs_metrics = observed.get(lane, {})
    if not exp_metrics:
        mismatches.append({"lane": lane, "metric": "_lane", "reason": "missing_lane_in_golden"})
        continue
    for key, exp in exp_metrics.items():
        if key not in obs_metrics:
            mismatches.append({"lane": lane, "metric": key, "reason": "missing_metric", "expected": exp})
            continue
        obs = obs_metrics[key]
        if not math.isfinite(obs):
            mismatches.append({"lane": lane, "metric": key, "reason": "non_finite", "observed": obs})
            continue
        tol = abs_tol + rel_tol * abs(float(exp))
        delta = abs(obs - float(exp))
        if delta > tol:
            mismatches.append({
                "lane": lane,
                "metric": key,
                "reason": "mismatch",
                "expected": float(exp),
                "observed": obs,
                "delta": delta,
                "tolerance": tol,
            })

pass_count = sum(1 for r in rows if r["status"] == "pass")
fail_count = sum(1 for r in rows if r["status"] == "fail")
skip_count = sum(1 for r in rows if r["status"] == "skip")

rows_by_lane = {lane: None for lane in lanes}
for row in rows:
    rows_by_lane[row["lane"]] = row

mismatch_by_lane = {lane: [] for lane in lanes}
for mm in mismatches:
    lane = mm.get("lane", "")
    if lane in mismatch_by_lane:
        mismatch_by_lane[lane].append(mm)

def lane_test_path(lane: str) -> str:
    if lane == "nn":
        return "tests/stdlib/nn/test_hyper_quaternion_e2e.sio"
    if lane == "onn":
        return "tests/stdlib/onn/test_hyper_onn_e2e.sio"
    if lane == "qnn":
        return "tests/stdlib/qnn/test_hyper_qnn_e2e.sio"
    if lane == "snn":
        return "tests/stdlib/snn/test_snn_e2e.sio"
    if lane == "spnn":
        return "tests/stdlib/spnn/test_spiking_e2e.sio"
    if lane == "quantnn":
        return "tests/stdlib/quantnn/test_quantum_e2e.sio"
    return "tests/stdlib/math/test_hyper_math_e2e.sio"

lane_rows = []
lane_blockers = set()
for lane in lanes:
    row = rows_by_lane.get(lane)
    status = "not_run"
    reason = "native_lane_missing"
    blockers = ["native_lane_missing"]
    metrics_status = "not_run"

    if row is not None:
        if row["status"] == "pass":
            status = "pass"
            reason = "ok"
            blockers = []
            metrics_status = "pass"
        elif row["status"] == "skip":
            status = "not_run"
            reason = "native_lane_missing"
            blockers = ["native_lane_missing"]
            metrics_status = "not_run"
        elif row["reason"] == "check_failed":
            status = "fail"
            reason = "native_lane_compile_fail"
            blockers = ["native_lane_compile_fail"]
            metrics_status = "fail"
        else:
            status = "fail"
            reason = "native_lane_runtime_fail"
            blockers = ["native_lane_runtime_fail"]
            metrics_status = "fail"

    if mismatch_by_lane[lane]:
        status = "fail"
        reason = "native_lane_parity_fail"
        blockers = ["native_lane_parity_fail"]
        metrics_status = "fail"

    for blocker in blockers:
        lane_blockers.add(blocker)

    lane_rows.append({
        "lane": lane,
        "required": True,
        "test_path": lane_test_path(lane),
        "status": status,
        "reason": reason,
        "blockers": blockers,
        "metrics_status": metrics_status,
    })

failures = []
for r in rows:
    if r["status"] == "fail":
        failures.append({
            "test_path": r["test_path"],
            "category": r["reason"],
            "error_excerpt": r["excerpt"][:400],
        })
    if r["status"] == "skip":
        failures.append({
            "test_path": r["test_path"],
            "category": "native_lane_missing",
            "error_excerpt": r["excerpt"][:400],
        })

for mm in mismatches:
    failures.append({
        "test_path": f"tests/stdlib/{mm['lane']}",
        "category": "golden_mismatch",
        "error_excerpt": json.dumps(mm, separators=(",", ":")),
    })

if ignore_hits:
    failures.append({
        "test_path": "tests/stdlib/hyper",
        "category": "ignore_policy_violation",
        "error_excerpt": ignore_hits[0][:400],
    })

inv = {}
try:
    inv = json.loads(inventory_path.read_text(encoding="utf-8"))
except Exception:
    inv = {}
counts = inv.get("counts") if isinstance(inv.get("counts"), dict) else {}

status_summary = "pass"
if fail_count > 0 or mismatches or ignore_hits:
    status_summary = "fail"

lane_non_pass = [row for row in lane_rows if row["status"] != "pass"]
if mode == "required" and lane_non_pass:
    status_summary = "fail"
elif status_summary == "pass" and lane_non_pass:
    status_summary = "not_run"

blockers = set(lane_blockers)
if ignore_hits:
    blockers.add("native_lane_runtime_fail")

lane_matrix_obj = {
    "schema": "sounio.stdlib.native_lane_matrix.v1",
    "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    "mode": mode,
    "status_summary": status_summary,
    "lanes": lane_rows,
    "blockers": sorted(blockers),
    "notes": [
        f"golden={golden_path.relative_to(root).as_posix() if golden_path.exists() else str(golden_path)}",
        f"inventory={inventory_path.relative_to(root).as_posix() if inventory_path.exists() else str(inventory_path)}",
    ],
}
lane_matrix_path.write_text(json.dumps(lane_matrix_obj, indent=2) + "\n", encoding="utf-8")

obj = {
    "schema": "sounio.stdlib.hyper_execution_status.v1",
    "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    "command": "bash scripts/stdlib/stdlib_hyper_execution_gate.sh",
    "mode": mode,
    "status_summary": status_summary,
    "totals": {
        "pass": pass_count,
        "fail": fail_count,
        "skip": skip_count,
        "total": pass_count + fail_count + skip_count,
    },
    "inventory": {
        "hyper_active_files": int(counts.get("hyper_active_files", 0)),
        "hyper_disabled_files": int(counts.get("hyper_disabled_files", 0)),
        "hyper_stub_mod_files": int(counts.get("hyper_stub_mod_files", 0)),
    },
    "failures": failures,
    "golden_comparison": {
        "status": "pass" if not mismatches else "fail",
        "mismatches": mismatches,
    },
    "ignore_policy": {
        "status": "pass" if not ignore_hits else "fail",
        "violations": ignore_hits,
    },
    "native_lane_matrix_artifact": str(lane_matrix_path),
    "lane_statuses": lane_rows,
    "blockers": sorted(blockers),
    "notes": [
        f"required_tests={len(rows)}",
        f"golden={golden_path.relative_to(root).as_posix() if golden_path.exists() else str(golden_path)}",
        f"inventory={inventory_path.relative_to(root).as_posix() if inventory_path.exists() else str(inventory_path)}",
    ],
}
out_path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
PY

status_summary="$(python3 - "$OUT_JSON" <<'PY'
import json
import pathlib
import sys
obj = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))
print(obj.get('status_summary', 'not_run'))
PY
)"

echo "[stdlib-hyper-gate] status_json=${OUT_JSON#$ROOT_DIR/}"
echo "[stdlib-hyper-gate] lane_matrix_json=${NATIVE_LANE_MATRIX_JSON#$ROOT_DIR/}"
echo "[stdlib-hyper-gate] status_summary=$status_summary"

if [[ "$status_summary" == "pass" ]]; then
  echo "STDLIB_HYPER_EXECUTION_GATE_PASS"
  exit 0
fi

echo "error: STDLIB_HYPER_EXECUTION_GATE_${status_summary^^}" >&2
exit 1
