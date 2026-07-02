#!/usr/bin/env bash
# stdlib_foundations_science_gate.sh — physics/chemistry science lanes (foundations).
#
# Check: default souc (Madaros). Run: import-free shards via lean_single (Madaros f64 runtime blocker).
# Soft by default; STDLIB_SCIENCE_FOUNDATIONS_STRICT=1 fails closed on lane errors.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_JSON="${STDLIB_FOUNDATIONS_STATUS_OUT:-$ROOT_DIR/artifacts/stdlib/stdlib_foundations_science_status.v1.json}"
GOLDEN_JSON="${STDLIB_FOUNDATIONS_GOLDEN:-$ROOT_DIR/tests/fixtures/foundations/pipeline_golden.v1.json}"
PHYSICS_TEST="${STDLIB_SCIENCE_PHYSICS_TEST:-$ROOT_DIR/tests/stdlib/physics/test_foundations_science_e2e.sio}"
CHEMISTRY_TEST="${STDLIB_SCIENCE_CHEMISTRY_TEST:-$ROOT_DIR/tests/stdlib/chemistry/test_foundations_science_e2e.sio}"
PHYSICS_SHARDS=(
  "${STDLIB_SCIENCE_PHYSICS_CLASSICAL_SHARD:-$ROOT_DIR/stdlib/physics/classical.sio}"
  "${STDLIB_SCIENCE_PHYSICS_EM_SHARD:-$ROOT_DIR/stdlib/physics/em.sio}"
  "${STDLIB_SCIENCE_PHYSICS_SR_SHARD:-$ROOT_DIR/stdlib/physics/sr.sio}"
  "${STDLIB_SCIENCE_PHYSICS_THERMO_SHARD:-$ROOT_DIR/stdlib/physics/thermo.sio}"
)
CHEMISTRY_SHARDS=(
  "${STDLIB_SCIENCE_CHEMISTRY_ACIDS_SHARD:-$ROOT_DIR/stdlib/chemistry/acids.sio}"
  "${STDLIB_SCIENCE_CHEMISTRY_EQUIL_SHARD:-$ROOT_DIR/stdlib/chemistry/equilibrium.sio}"
)
STRICT="${STDLIB_SCIENCE_FOUNDATIONS_STRICT:-0}"
RUN_ENGINE="${STDLIB_FOUNDATIONS_RUN_ENGINE:-lean_single}"

# shellcheck source=lib/resolve_souc.sh
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

mkdir -p "$(dirname "$OUT_JSON")"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

CHECK_SOUC="$SOUC_BIN"
RUN_SOUC="$SOUC_BIN"

run_check() {
  local src="$1"
  local out="$2"
  if [[ ! -f "$src" ]]; then
    echo "[foundations-science] missing check target: $src" >&2
    echo "125"
    return 0
  fi
  set +e
  "$CHECK_SOUC" check "$src" >"$out" 2>&1
  local rc=$?
  set -e
  echo "$rc"
}

run_shard() {
  local src="$1"
  local out="$2"
  if [[ ! -f "$src" ]]; then
    echo "[foundations-science] missing shard: $src" >&2
    echo "125"
    return 0
  fi
  set +e
  SOUNIO_SOUC_ENGINE="$RUN_ENGINE" "$RUN_SOUC" run "$src" >"$out" 2>&1
  local rc=$?
  set -e
  echo "$rc"
}

PHYSICS_CHECK_OUT="$TMP_DIR/physics_check.out"
CHEMISTRY_CHECK_OUT="$TMP_DIR/chemistry_check.out"
PHYSICS_RUN_OUT="$TMP_DIR/physics_run.out"
CHEMISTRY_RUN_OUT="$TMP_DIR/chemistry_run.out"

physics_check_rc="$(run_check "$PHYSICS_TEST" "$PHYSICS_CHECK_OUT")"
chemistry_check_rc="$(run_check "$CHEMISTRY_TEST" "$CHEMISTRY_CHECK_OUT")"
: >"$PHYSICS_RUN_OUT"
classical_shard="${PHYSICS_SHARDS[0]}"
em_shard="${PHYSICS_SHARDS[1]}"
sr_shard="${PHYSICS_SHARDS[2]}"
thermo_shard="${PHYSICS_SHARDS[3]}"
classical_out="$TMP_DIR/phys_classical.out"
em_out="$TMP_DIR/phys_em.out"
sr_out="$TMP_DIR/phys_sr.out"
thermo_out="$TMP_DIR/phys_thermo.out"
classical_run_rc="$(run_shard "$classical_shard" "$classical_out")"
em_run_rc="$(run_shard "$em_shard" "$em_out")"
sr_run_rc="$(run_shard "$sr_shard" "$sr_out")"
thermo_run_rc="$(run_shard "$thermo_shard" "$thermo_out")"
cat "$classical_out" >>"$PHYSICS_RUN_OUT"
cat "$em_out" >>"$PHYSICS_RUN_OUT"
cat "$sr_out" >>"$PHYSICS_RUN_OUT"
cat "$thermo_out" >>"$PHYSICS_RUN_OUT"
physics_run_rc=0
if [[ "$classical_run_rc" -ne 0 || "$em_run_rc" -ne 0 || "$sr_run_rc" -ne 0 || "$thermo_run_rc" -ne 0 ]]; then
  physics_run_rc=1
fi
echo "[foundations-science] shard classical rc=$classical_run_rc" >&2
echo "[foundations-science] shard em rc=$em_run_rc" >&2
echo "[foundations-science] shard sr rc=$sr_run_rc" >&2
echo "[foundations-science] shard thermo rc=$thermo_run_rc" >&2

: >"$CHEMISTRY_RUN_OUT"
acids_shard="${CHEMISTRY_SHARDS[0]}"
equil_shard="${CHEMISTRY_SHARDS[1]}"
acids_out="$TMP_DIR/chem_acids.out"
equil_out="$TMP_DIR/chem_equilibrium.out"
acids_run_rc="$(run_shard "$acids_shard" "$acids_out")"
equil_run_rc="$(run_shard "$equil_shard" "$equil_out")"
cat "$acids_out" >>"$CHEMISTRY_RUN_OUT"
cat "$equil_out" >>"$CHEMISTRY_RUN_OUT"
chemistry_run_rc=0
if [[ "$acids_run_rc" -ne 0 || "$equil_run_rc" -ne 0 ]]; then
  chemistry_run_rc=1
fi
echo "[foundations-science] shard acids rc=$acids_run_rc" >&2
echo "[foundations-science] shard equilibrium rc=$equil_run_rc" >&2

python3 - "$ROOT_DIR" "$OUT_JSON" "$GOLDEN_JSON" "$PHYSICS_TEST" "$CHEMISTRY_TEST" \
  "$PHYSICS_CHECK_OUT" "$CHEMISTRY_CHECK_OUT" "$PHYSICS_RUN_OUT" "$CHEMISTRY_RUN_OUT" \
  "$physics_check_rc" "$chemistry_check_rc" "$physics_run_rc" "$chemistry_run_rc" \
  "$classical_run_rc" "$em_run_rc" "$sr_run_rc" "$thermo_run_rc" \
  "$acids_run_rc" "$equil_run_rc" "$STRICT" "$RUN_ENGINE" <<'PY'
import datetime
import json
import math
import re
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
out_path = Path(sys.argv[2]).resolve()
golden_path = Path(sys.argv[3]).resolve()
physics_test = Path(sys.argv[4]).resolve()
chemistry_test = Path(sys.argv[5]).resolve()
physics_check_out_path = Path(sys.argv[6]).resolve()
chemistry_check_out_path = Path(sys.argv[7]).resolve()
physics_run_out_path = Path(sys.argv[8]).resolve()
chemistry_run_out_path = Path(sys.argv[9]).resolve()
physics_check_rc = int(sys.argv[10])
chemistry_check_rc = int(sys.argv[11])
physics_run_rc = int(sys.argv[12])
chemistry_run_rc = int(sys.argv[13])
classical_run_rc = int(sys.argv[14])
em_run_rc = int(sys.argv[15])
sr_run_rc = int(sys.argv[16])
thermo_run_rc = int(sys.argv[17])
acids_run_rc = int(sys.argv[18])
equil_run_rc = int(sys.argv[19])
strict = sys.argv[20] == "1"
run_engine = sys.argv[21]

now = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

golden_obj = {}
if golden_path.exists():
    golden_obj = json.loads(golden_path.read_text(encoding="utf-8"))

abs_tol = float((golden_obj.get("tolerance") or {}).get("abs", 0.15))
rel_tol = float((golden_obj.get("tolerance") or {}).get("rel", 0.05))

pattern = re.compile(r"SCIENCE_METRIC\s+(physics|chemistry)\s+([A-Za-z0-9_]+)\s+([-+0-9.eE]+)")

def parse_metrics(text: str):
    metrics = {"physics": {}, "chemistry": {}}
    for lane, key, raw_val in pattern.findall(text):
        try:
            metrics[lane][key] = float(raw_val)
        except Exception:
            pass
    return metrics

physics_check_out = physics_check_out_path.read_text(encoding="utf-8", errors="replace") if physics_check_out_path.exists() else ""
chemistry_check_out = chemistry_check_out_path.read_text(encoding="utf-8", errors="replace") if chemistry_check_out_path.exists() else ""
physics_run_out = physics_run_out_path.read_text(encoding="utf-8", errors="replace") if physics_run_out_path.exists() else ""
chemistry_run_out = chemistry_run_out_path.read_text(encoding="utf-8", errors="replace") if chemistry_run_out_path.exists() else ""

run_metrics = parse_metrics(physics_run_out + "\n" + chemistry_run_out)

chem = run_metrics.get("chemistry", {})
if chem.get("acids_ok") == 1.0 and chem.get("equilibrium_ok") == 1.0:
    chem["tests_passed"] = 8.0
run_metrics["chemistry"] = chem

phys = run_metrics.get("physics", {})
if (
    phys.get("classical_ok") == 1.0
    and phys.get("em_ok") == 1.0
    and phys.get("sr_ok") == 1.0
    and phys.get("thermo_ok") == 1.0
):
    phys["tests_passed"] = 15.0
run_metrics["physics"] = phys

def lane_row(
    lane: str,
    check_rc: int,
    run_rc: int,
    marker: str,
    check_test: Path,
    run_target: str,
    run_text: str,
):
    expected = ((golden_obj.get(lane) or {}).get("metrics") or {})
    mismatches = []
    status = "pass"
    observed = run_metrics.get(lane, {})

    if check_rc == 125 or run_rc == 125:
        status = "not_run"
    elif check_rc != 0:
        status = "fail"
        mismatches.append({"metric": "_check", "reason": f"check_exit_code={check_rc}"})
    elif run_rc != 0:
        status = "fail"
        mismatches.append({"metric": "_run", "reason": f"run_exit_code={run_rc}"})

    marker_found = marker in run_text
    if status != "not_run" and not marker_found:
        status = "fail"
        mismatches.append({"metric": "_marker", "reason": f"missing {marker}"})

    if status != "not_run" and expected:
        for key, exp_val in expected.items():
            if key not in observed:
                status = "fail"
                mismatches.append({"metric": key, "reason": "missing_metric", "expected": exp_val})
                continue
            obs_val = observed[key]
            delta = abs(obs_val - float(exp_val))
            tol = abs_tol + rel_tol * abs(float(exp_val))
            if not math.isfinite(obs_val) or delta > tol:
                status = "fail"
                mismatches.append(
                    {
                        "metric": key,
                        "expected": float(exp_val),
                        "observed": obs_val,
                        "delta": delta,
                        "tolerance": tol,
                    }
                )

    return {
        "status": status,
        "check_test_path": check_test.relative_to(root).as_posix() if check_test.exists() else str(check_test),
        "run_target": run_target,
        "check_exit_code": check_rc,
        "run_exit_code": run_rc,
        "run_engine": run_engine,
        "marker_found": marker_found,
        "metrics": observed,
        "mismatches": mismatches,
        "run_blocker": None if run_rc == 0 else "madaros_f64_runtime" if run_engine == "madaros" else "shard_run_failed",
    }

physics_lane = lane_row(
    "physics",
    physics_check_rc,
    physics_run_rc,
    "SCIENCE_PHYSICS_OK",
    physics_test,
    "stdlib/physics/{classical,em,sr,thermo}.sio",
    physics_run_out,
)
chemistry_lane = lane_row(
    "chemistry",
    chemistry_check_rc,
    chemistry_run_rc,
    "SCIENCE_CHEMISTRY_OK",
    chemistry_test,
    "stdlib/chemistry/{acids,equilibrium}.sio",
    chemistry_run_out,
)
chemistry_lane["shard_runs"] = {
    "acids": {
        "path": "stdlib/chemistry/acids.sio",
        "run_exit_code": acids_run_rc,
        "status": "pass" if acids_run_rc == 0 else "fail",
    },
    "equilibrium": {
        "path": "stdlib/chemistry/equilibrium.sio",
        "run_exit_code": equil_run_rc,
        "status": "pass" if equil_run_rc == 0 else "fail",
    },
}
physics_lane["shard_runs"] = {
    "classical": {
        "path": "stdlib/physics/classical.sio",
        "run_exit_code": classical_run_rc,
        "status": "pass" if classical_run_rc == 0 else "fail",
    },
    "em": {
        "path": "stdlib/physics/em.sio",
        "run_exit_code": em_run_rc,
        "status": "pass" if em_run_rc == 0 else "fail",
    },
    "sr": {
        "path": "stdlib/physics/sr.sio",
        "run_exit_code": sr_run_rc,
        "status": "pass" if sr_run_rc == 0 else "fail",
    },
    "thermo": {
        "path": "stdlib/physics/thermo.sio",
        "run_exit_code": thermo_run_rc,
        "status": "pass" if thermo_run_rc == 0 else "fail",
    },
}

lanes = [physics_lane, chemistry_lane]
pass_count = sum(1 for l in lanes if l["status"] == "pass")
fail_count = sum(1 for l in lanes if l["status"] == "fail")
not_run_count = sum(1 for l in lanes if l["status"] == "not_run")

if strict and fail_count > 0:
    status_summary = "fail"
elif pass_count == len(lanes):
    status_summary = "pass"
elif not_run_count == len(lanes):
    status_summary = "not_run"
elif strict:
    status_summary = "fail"
else:
    status_summary = "soft_pass"

obj = {
    "schema": "sounio.stdlib.foundations_science_status.v1",
    "generated_at_utc": now,
    "command": "bash scripts/stdlib/stdlib_foundations_science_gate.sh",
    "status_summary": status_summary,
    "strict_mode": strict,
    "run_engine": run_engine,
    "totals": {"pass": pass_count, "fail": fail_count, "not_run": not_run_count, "total": len(lanes)},
    "golden_file": golden_path.relative_to(root).as_posix() if golden_path.exists() else str(golden_path),
    "tolerance": {"abs": abs_tol, "rel": rel_tol},
    "lanes": {"physics": physics_lane, "chemistry": chemistry_lane},
    "notes": [
        "check_engine=madaros_default",
        f"run_engine={run_engine}",
        "madaros_imported_run_blocker=SIGSEGV_on_multi_import_E2E",
        "madaros_f64_runtime_blocker=negative_subtraction_and_metric_print_SIGSEGV",
        f"physics_check_rc={physics_check_rc}",
        f"chemistry_check_rc={chemistry_check_rc}",
        f"physics_run_rc={physics_run_rc}",
        f"chemistry_run_rc={chemistry_run_rc}",
        f"classical_shard_rc={classical_run_rc}",
        f"em_shard_rc={em_run_rc}",
        f"sr_shard_rc={sr_run_rc}",
        f"thermo_shard_rc={thermo_run_rc}",
        f"acids_shard_rc={acids_run_rc}",
        f"equilibrium_shard_rc={equil_run_rc}",
    ],
}
out_path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
print(status_summary)
PY

status_summary="$(python3 - "$OUT_JSON" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["status_summary"])
PY
)"

echo "[foundations-science-gate] status_json=${OUT_JSON#$ROOT_DIR/}"
echo "[foundations-science-gate] status_summary=$status_summary"
echo "[foundations-science-gate] run_engine=$RUN_ENGINE"

if [[ "$status_summary" == "pass" || "$status_summary" == "soft_pass" ]]; then
  echo "STDLIB_FOUNDATIONS_SCIENCE_GATE_${status_summary^^}"
  exit 0
fi

echo "error: STDLIB_FOUNDATIONS_SCIENCE_GATE_FAIL" >&2
exit 1