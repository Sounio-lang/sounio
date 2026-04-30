#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[native-v2-cpu-compiler] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[native-v2-cpu-compiler] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

OUT_DIR="${SOUNIO_NATIVE_V2_CPU_COMPILER_DIR:-$(mktemp -d /tmp/sounio-native-v2-cpu-compiler.XXXXXX)}"
LOG_DIR="$OUT_DIR/logs"
ARTIFACT_DIR="$OUT_DIR/artifacts"
RESULTS_TSV="$ARTIFACT_DIR/gates.tsv"
SUMMARY_JSON="${SOUNIO_NATIVE_V2_CPU_COMPILER_SUMMARY:-$ARTIFACT_DIR/native_v2_cpu_compiler_umbrella.v1.json}"

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

cat >"$RESULTS_TSV" <<'EOF'
gate	rc	out_dir	summary_json	log_path
EOF

append_gate_row() {
  local gate="$1"
  local rc="$2"
  local out_dir="$3"
  local summary_json="$4"
  local log_path="$5"

  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$gate" "$rc" "$out_dir" "$summary_json" "$log_path" >>"$RESULTS_TSV"
}

run_driver_self_compile() {
  local gate="driver_self_compile"
  local gate_dir="$OUT_DIR/$gate"
  local log_path="$LOG_DIR/$gate.log"

  mkdir -p "$gate_dir"
  echo "[native-v2-cpu-compiler] running $gate"
  set +e
  SOUNIO_NATIVE_V2_DRIVER_SELF_COMPILE_DIR="$gate_dir" \
    bash scripts/ci/native_v2_driver_self_compile_gate.sh >"$log_path" 2>&1
  local rc=$?
  set -e
  append_gate_row "$gate" "$rc" "$gate_dir" "-" "$log_path"
}

run_science_spine() {
  local gate="science_spine"
  local gate_dir="$OUT_DIR/$gate"
  local log_path="$LOG_DIR/$gate.log"
  local summary_json="$gate_dir/artifacts/summary.json"

  mkdir -p "$gate_dir"
  echo "[native-v2-cpu-compiler] running $gate"
  set +e
  SOUNIO_NATIVE_V2_SCIENCE_SPINE_DIR="$gate_dir" \
  SOUNIO_NATIVE_V2_DRIVER_SELF_COMPILE_DIR="$gate_dir/driver-self" \
    bash scripts/ci/native_v2_epistemic_science_spine_gate.sh >"$log_path" 2>&1
  local rc=$?
  set -e
  append_gate_row "$gate" "$rc" "$gate_dir" "$summary_json" "$log_path"
}

run_f64_ladder() {
  local gate="f64_ladder"
  local gate_dir="$OUT_DIR/$gate"
  local log_path="$LOG_DIR/$gate.log"
  local summary_json="$gate_dir/artifacts/summary.json"

  mkdir -p "$gate_dir"
  echo "[native-v2-cpu-compiler] running $gate"
  set +e
  SOUNIO_NATIVE_V2_F64_LADDER_DIR="$gate_dir" \
  SOUNIO_NATIVE_V2_DRIVER_SELF_COMPILE_DIR="$gate_dir/driver-self" \
    bash scripts/ci/native_v2_f64_ladder_gate.sh >"$log_path" 2>&1
  local rc=$?
  set -e
  append_gate_row "$gate" "$rc" "$gate_dir" "$summary_json" "$log_path"
}

run_gum_primitives() {
  local gate="gum_primitives"
  local gate_dir="$OUT_DIR/$gate"
  local log_path="$LOG_DIR/$gate.log"
  local summary_json="$gate_dir/artifacts/summary.json"

  mkdir -p "$gate_dir"
  echo "[native-v2-cpu-compiler] running $gate"
  set +e
  SOUNIO_NATIVE_V2_GUM_DIR="$gate_dir" \
  SOUNIO_NATIVE_V2_DRIVER_SELF_COMPILE_DIR="$gate_dir/driver-self" \
    bash scripts/ci/native_v2_gum_primitives_gate.sh >"$log_path" 2>&1
  local rc=$?
  set -e
  append_gate_row "$gate" "$rc" "$gate_dir" "$summary_json" "$log_path"
}

run_semantic_hardening() {
  local gate="semantic_hardening"
  local gate_dir="$OUT_DIR/$gate"
  local log_path="$LOG_DIR/$gate.log"
  local summary_json="$gate_dir/artifacts/summary.json"

  mkdir -p "$gate_dir"
  echo "[native-v2-cpu-compiler] running $gate"
  set +e
  SOUNIO_NATIVE_V2_SEMANTIC_HARDENING_DIR="$gate_dir" \
  SOUNIO_NATIVE_V2_DRIVER_SELF_COMPILE_DIR="$gate_dir/driver-self" \
    bash scripts/ci/native_v2_semantic_hardening_gate.sh >"$log_path" 2>&1
  local rc=$?
  set -e
  append_gate_row "$gate" "$rc" "$gate_dir" "$summary_json" "$log_path"
}

run_lean_single_fixed_point() {
  local gate="lean_single_fixed_point"
  local gate_dir="$OUT_DIR/$gate"
  local log_path="$LOG_DIR/$gate.log"

  mkdir -p "$gate_dir"
  echo "[native-v2-cpu-compiler] running $gate"
  set +e
  SOUNIO_LEAN_SINGLE_FP_DIR="$gate_dir" \
    bash scripts/ci/lean_single_fixed_point_gate.sh >"$log_path" 2>&1
  local rc=$?
  set -e
  append_gate_row "$gate" "$rc" "$gate_dir" "-" "$log_path"
}

emit_summary_json() {
  python3 - "$SUMMARY_JSON" "$RESULTS_TSV" "$SOUC_BIN" "$OUT_DIR" <<'PY'
import csv
import json
import pathlib
import re
import sys
from datetime import datetime, timezone

summary_path = pathlib.Path(sys.argv[1])
results_path = pathlib.Path(sys.argv[2])
souc_bin = sys.argv[3]
out_dir = pathlib.Path(sys.argv[4])

def read_json(path):
    try:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "missing", "reason": f"summary_read_failed:{exc}"}

def read_text(path):
    try:
        return pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

rows = []
with open(results_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

gates = []
fail_reasons = []

for row in rows:
    gate = row["gate"]
    rc = int(row["rc"])
    log_path = row["log_path"]
    summary_json = row["summary_json"]
    entry = {
        "gate": gate,
        "status": "pass" if rc == 0 else "fail",
        "rc": rc,
        "out_dir": row["out_dir"],
        "summary_json": None if summary_json == "-" else summary_json,
        "log_path": log_path,
    }

    if summary_json != "-":
        payload = read_json(summary_json)
        payload_status = payload.get("status", "missing")
        entry.update({
            "schema": payload.get("schema"),
            "summary_status": payload_status,
            "case_count": payload.get("case_count"),
            "pass_count": payload.get("pass_count"),
            "fail_count": payload.get("fail_count"),
            "fallback_path": payload.get("fallback_path"),
            "host_callback": payload.get("host_callback"),
            "stage1_driver_sha256": payload.get("stage1_driver_sha256"),
            "stage1_driver_deterministic": payload.get("stage1_driver_deterministic"),
            "gum_sse2_verified": payload.get("gum_sse2_verified"),
            "artifact_dir": payload.get("artifact_dir"),
            "results_tsv": payload.get("results_tsv"),
        })
        if payload_status != "pass":
            entry["status"] = "fail"
            fail_reasons.append(f"{gate}:summary_status={payload_status}")
        if payload.get("stage1_driver_deterministic") is False:
            entry["status"] = "fail"
            fail_reasons.append(f"{gate}:stage1_driver_not_deterministic")
        if payload.get("fallback_path") not in (None, "none"):
            entry["status"] = "fail"
            fail_reasons.append(f"{gate}:fallback_path={payload.get('fallback_path')}")
        if payload.get("host_callback") not in (None, "none"):
            entry["status"] = "fail"
            fail_reasons.append(f"{gate}:host_callback={payload.get('host_callback')}")
    else:
        text = read_text(log_path)
        md5_match = re.search(r"fixed-point md5=([0-9a-fA-F]+)", text)
        ep_match = re.search(r"epistemic stage1=(\S+) stage2=(\S+) stage3=(\S+)", text)
        stage1_ep = ep_match.group(1) if ep_match else None
        stage2_ep = ep_match.group(2) if ep_match else None
        stage3_ep = ep_match.group(3) if ep_match else None
        entry.update({
            "fixed_point_md5": md5_match.group(1) if md5_match else None,
            "stage1_epistemic": stage1_ep,
            "stage2_epistemic": stage2_ep,
            "stage3_epistemic": stage3_ep,
            "epistemic_fixed_point": bool(stage2_ep and stage2_ep == stage3_ep and stage2_ep != "absent"),
            "fallback_path": "none",
            "host_callback": "none",
            "policy_source": "native_v2_driver_self_compile_gate",
        })
        if rc == 0 and not entry["epistemic_fixed_point"]:
            entry["status"] = "fail"
            fail_reasons.append(f"{gate}:epistemic_fixed_point_unparsed")

    if rc != 0:
        fail_reasons.append(f"{gate}:rc={rc}")
    gates.append(entry)

status = "pass" if gates and all(g["status"] == "pass" for g in gates) else "fail"
payload = {
    "schema": "sounio.native_v2_cpu_compiler_umbrella.v1",
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "status": status,
    "compiler_resolved": souc_bin,
    "target": "x86_64-linux",
    "fallback_path": "none",
    "host_callback": "none",
    "gate_count": len(gates),
    "pass_count": sum(1 for g in gates if g["status"] == "pass"),
    "fail_count": sum(1 for g in gates if g["status"] != "pass"),
    "case_count": sum((g.get("case_count") or 0) for g in gates),
    "out_dir": str(out_dir),
    "results_tsv": str(results_path),
    "gates": gates,
    "fail_reasons": fail_reasons,
    "remaining_boundaries": [
        "CPU umbrella gate does not prove CUDA runtime execution",
        "CPU umbrella gate does not change compiler semantics or public support claims",
    ],
}

summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if status != "pass":
    raise SystemExit(1)
PY
}

echo "[native-v2-cpu-compiler] souc=$SOUC_BIN"
echo "[native-v2-cpu-compiler] out=$OUT_DIR"
echo "[native-v2-cpu-compiler] summary=$SUMMARY_JSON"

run_driver_self_compile
run_science_spine
run_f64_ladder
run_gum_primitives
run_semantic_hardening
run_lean_single_fixed_point

if emit_summary_json; then
  echo "[native-v2-cpu-compiler] PASS: driver self-compile, science spine, f64 ladder, GUM primitives, semantic hardening, lean_single fixed-point"
  echo "[native-v2-cpu-compiler] summary=$SUMMARY_JSON"
else
  echo "[native-v2-cpu-compiler] FAIL: see summary=$SUMMARY_JSON" >&2
  exit 1
fi
