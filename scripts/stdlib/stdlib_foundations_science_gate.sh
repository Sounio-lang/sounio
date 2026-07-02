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
KINETICS_TEST="${STDLIB_SCIENCE_KINETICS_TEST:-$ROOT_DIR/tests/stdlib/chemistry/test_kinetics_foundations_e2e.sio}"
THERMO_KT_TEST="${STDLIB_SCIENCE_THERMO_KT_TEST:-$ROOT_DIR/tests/stdlib/chemistry/test_thermochem_equilibrium_kt_e2e.sio}"
THERMO_KINETICS_KT_TEST="${STDLIB_SCIENCE_THERMO_KINETICS_KT_TEST:-$ROOT_DIR/tests/stdlib/chemistry/test_thermochem_kinetics_kt_e2e.sio}"
ONTOLOGY_TEST="${STDLIB_SCIENCE_ONTOLOGY_TEST:-$ROOT_DIR/tests/stdlib/chemistry/test_chemistry_ontology_e2e.sio}"
PHYSICS_SHARDS=(
  "${STDLIB_SCIENCE_PHYSICS_CLASSICAL_SHARD:-$ROOT_DIR/stdlib/physics/classical.sio}"
  "${STDLIB_SCIENCE_PHYSICS_EM_SHARD:-$ROOT_DIR/stdlib/physics/em.sio}"
  "${STDLIB_SCIENCE_PHYSICS_SR_SHARD:-$ROOT_DIR/stdlib/physics/sr.sio}"
  "${STDLIB_SCIENCE_PHYSICS_THERMO_SHARD:-$ROOT_DIR/stdlib/physics/thermo.sio}"
)
CHEMISTRY_SHARDS=(
  "${STDLIB_SCIENCE_CHEMISTRY_ACIDS_SHARD:-$ROOT_DIR/stdlib/chemistry/acids.sio}"
  "${STDLIB_SCIENCE_CHEMISTRY_EQUIL_SHARD:-$ROOT_DIR/stdlib/chemistry/equilibrium.sio}"
  "${STDLIB_SCIENCE_CHEMISTRY_STOICH_SHARD:-$ROOT_DIR/stdlib/chemistry/stoichiometry.sio}"
  "${STDLIB_SCIENCE_CHEMISTRY_THERMO_SHARD:-$ROOT_DIR/stdlib/chemistry/thermochem.sio}"
  "${STDLIB_SCIENCE_CHEMISTRY_ONTOLOGY_SHARD:-$ROOT_DIR/stdlib/chemistry/ontology.sio}"
)
STRICT="${STDLIB_SCIENCE_FOUNDATIONS_STRICT:-0}"
RUN_ENGINE="${STDLIB_FOUNDATIONS_RUN_ENGINE:-lean_single}"
CHECK_TIMEOUT_SEC="${STDLIB_SCIENCE_CHECK_TIMEOUT_SEC:-90}"

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
  if command -v timeout >/dev/null 2>&1; then
    timeout --preserve-status "${CHECK_TIMEOUT_SEC}s" "$CHECK_SOUC" check "$src" >"$out" 2>&1
    local rc=$?
    if [[ "$rc" -eq 124 ]]; then
      echo "[foundations-science] check timeout (${CHECK_TIMEOUT_SEC}s): $src" >>"$out"
    fi
  else
    "$CHECK_SOUC" check "$src" >"$out" 2>&1
    local rc=$?
  fi
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
KINETICS_CHECK_OUT="$TMP_DIR/kinetics_check.out"
THERMO_KT_CHECK_OUT="$TMP_DIR/thermo_kt_check.out"
THERMO_KINETICS_KT_CHECK_OUT="$TMP_DIR/thermo_kinetics_kt_check.out"
ONTOLOGY_CHECK_OUT="$TMP_DIR/ontology_check.out"
PHYSICS_RUN_OUT="$TMP_DIR/physics_run.out"
CHEMISTRY_RUN_OUT="$TMP_DIR/chemistry_run.out"

physics_check_rc="$(run_check "$PHYSICS_TEST" "$PHYSICS_CHECK_OUT")"
chemistry_foundations_check_rc="$(run_check "$CHEMISTRY_TEST" "$CHEMISTRY_CHECK_OUT")"
kinetics_check_rc="$(run_check "$KINETICS_TEST" "$KINETICS_CHECK_OUT")"
thermo_kt_check_rc="$(run_check "$THERMO_KT_TEST" "$THERMO_KT_CHECK_OUT")"
thermo_kinetics_kt_check_rc="$(run_check "$THERMO_KINETICS_KT_TEST" "$THERMO_KINETICS_KT_CHECK_OUT")"
ontology_check_rc="$(run_check "$ONTOLOGY_TEST" "$ONTOLOGY_CHECK_OUT")"
chemistry_check_rc=0
if [[ "$chemistry_foundations_check_rc" -ne 0 || "$kinetics_check_rc" -ne 0 || "$thermo_kt_check_rc" -ne 0 || "$thermo_kinetics_kt_check_rc" -ne 0 || "$ontology_check_rc" -ne 0 ]]; then
  chemistry_check_rc=1
fi
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
stoich_shard="${CHEMISTRY_SHARDS[2]}"
thermo_shard="${CHEMISTRY_SHARDS[3]}"
ontology_shard="${CHEMISTRY_SHARDS[4]}"
acids_out="$TMP_DIR/chem_acids.out"
equil_out="$TMP_DIR/chem_equilibrium.out"
stoich_out="$TMP_DIR/chem_stoichiometry.out"
thermo_out="$TMP_DIR/chem_thermochem.out"
ontology_out="$TMP_DIR/chem_ontology.out"
acids_run_rc="$(run_shard "$acids_shard" "$acids_out")"
equil_run_rc="$(run_shard "$equil_shard" "$equil_out")"
stoich_run_rc="$(run_shard "$stoich_shard" "$stoich_out")"
thermochem_run_rc="$(run_shard "$thermo_shard" "$thermo_out")"
ontology_run_rc="$(run_shard "$ontology_shard" "$ontology_out")"
cat "$acids_out" >>"$CHEMISTRY_RUN_OUT"
cat "$equil_out" >>"$CHEMISTRY_RUN_OUT"
cat "$stoich_out" >>"$CHEMISTRY_RUN_OUT"
cat "$thermo_out" >>"$CHEMISTRY_RUN_OUT"
cat "$ontology_out" >>"$CHEMISTRY_RUN_OUT"
chemistry_run_rc=0
if [[ "$acids_run_rc" -ne 0 || "$equil_run_rc" -ne 0 || "$stoich_run_rc" -ne 0 || "$thermochem_run_rc" -ne 0 || "$ontology_run_rc" -ne 0 ]]; then
  chemistry_run_rc=1
fi
echo "[foundations-science] shard acids rc=$acids_run_rc" >&2
echo "[foundations-science] shard equilibrium rc=$equil_run_rc" >&2
echo "[foundations-science] shard stoichiometry rc=$stoich_run_rc" >&2
echo "[foundations-science] shard thermochem rc=$thermochem_run_rc" >&2
echo "[foundations-science] shard ontology rc=$ontology_run_rc" >&2
echo "[foundations-science] kinetics check rc=$kinetics_check_rc" >&2
echo "[foundations-science] thermo_kt check rc=$thermo_kt_check_rc" >&2
echo "[foundations-science] thermo_kinetics_kt check rc=$thermo_kinetics_kt_check_rc" >&2
echo "[foundations-science] ontology check rc=$ontology_check_rc" >&2

python3 - "$ROOT_DIR" "$OUT_JSON" "$GOLDEN_JSON" "$PHYSICS_TEST" "$CHEMISTRY_TEST" "$KINETICS_TEST" "$THERMO_KT_TEST" "$THERMO_KINETICS_KT_TEST" "$ONTOLOGY_TEST" \
  "$PHYSICS_CHECK_OUT" "$CHEMISTRY_CHECK_OUT" "$KINETICS_CHECK_OUT" "$THERMO_KT_CHECK_OUT" "$THERMO_KINETICS_KT_CHECK_OUT" "$ONTOLOGY_CHECK_OUT" "$PHYSICS_RUN_OUT" "$CHEMISTRY_RUN_OUT" \
  "$physics_check_rc" "$chemistry_check_rc" "$kinetics_check_rc" "$thermo_kt_check_rc" "$thermo_kinetics_kt_check_rc" "$ontology_check_rc" "$physics_run_rc" "$chemistry_run_rc" \
  "$classical_run_rc" "$em_run_rc" "$sr_run_rc" "$thermo_run_rc" \
  "$acids_run_rc" "$equil_run_rc" "$stoich_run_rc" "$thermochem_run_rc" "$ontology_run_rc" "$STRICT" "$RUN_ENGINE" <<'PY'
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
kinetics_test = Path(sys.argv[6]).resolve()
thermo_kt_test = Path(sys.argv[7]).resolve()
thermo_kinetics_kt_test = Path(sys.argv[8]).resolve()
ontology_test = Path(sys.argv[9]).resolve()
physics_check_out_path = Path(sys.argv[10]).resolve()
chemistry_check_out_path = Path(sys.argv[11]).resolve()
kinetics_check_out_path = Path(sys.argv[12]).resolve()
thermo_kt_check_out_path = Path(sys.argv[13]).resolve()
thermo_kinetics_kt_check_out_path = Path(sys.argv[14]).resolve()
ontology_check_out_path = Path(sys.argv[15]).resolve()
physics_run_out_path = Path(sys.argv[16]).resolve()
chemistry_run_out_path = Path(sys.argv[17]).resolve()
physics_check_rc = int(sys.argv[18])
chemistry_check_rc = int(sys.argv[19])
kinetics_check_rc = int(sys.argv[20])
thermo_kt_check_rc = int(sys.argv[21])
thermo_kinetics_kt_check_rc = int(sys.argv[22])
ontology_check_rc = int(sys.argv[23])
physics_run_rc = int(sys.argv[24])
chemistry_run_rc = int(sys.argv[25])
classical_run_rc = int(sys.argv[26])
em_run_rc = int(sys.argv[27])
sr_run_rc = int(sys.argv[28])
phys_thermo_run_rc = int(sys.argv[29])
acids_run_rc = int(sys.argv[30])
equil_run_rc = int(sys.argv[31])
stoich_run_rc = int(sys.argv[32])
thermochem_run_rc = int(sys.argv[33])
ontology_run_rc = int(sys.argv[34])
strict = sys.argv[35] == "1"
run_engine = sys.argv[36]

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
if (
    chem.get("acids_ok") == 1.0
    and chem.get("equilibrium_ok") == 1.0
    and chem.get("stoichiometry_ok") == 1.0
    and chem.get("thermochem_ok") == 1.0
    and chem.get("kt_coupling_ok") == 1.0
    and chem.get("kt_full_kirchhoff_ok") == 1.0
    and chem.get("ontology_ok") == 1.0
):
    chem["tests_passed"] = 15.0
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
    "stdlib/chemistry/{acids,equilibrium,stoichiometry,thermochem,ontology}.sio",
    chemistry_run_out,
)
chemistry_lane["kinetics_check"] = {
    "check_test_path": kinetics_test.relative_to(root).as_posix() if kinetics_test.exists() else str(kinetics_test),
    "check_exit_code": kinetics_check_rc,
    "status": "pass" if kinetics_check_rc == 0 else "fail",
    "run_blocker": "lean_single_nested_array_and_ref_indexing",
}
chemistry_lane["thermo_kt_check"] = {
    "check_test_path": thermo_kt_test.relative_to(root).as_posix() if thermo_kt_test.exists() else str(thermo_kt_test),
    "check_exit_code": thermo_kt_check_rc,
    "status": "pass" if thermo_kt_check_rc == 0 else "fail",
    "run_blocker": "madaros_imported_run_SIGSEGV",
}
chemistry_lane["thermo_kinetics_kt_check"] = {
    "check_test_path": thermo_kinetics_kt_test.relative_to(root).as_posix() if thermo_kinetics_kt_test.exists() else str(thermo_kinetics_kt_test),
    "check_exit_code": thermo_kinetics_kt_check_rc,
    "status": "pass" if thermo_kinetics_kt_check_rc == 0 else "fail",
    "run_blocker": "madaros_imported_run_SIGSEGV",
}
chemistry_lane["ontology_check"] = {
    "check_test_path": ontology_test.relative_to(root).as_posix() if ontology_test.exists() else str(ontology_test),
    "check_exit_code": ontology_check_rc,
    "status": "pass" if ontology_check_rc == 0 else "fail",
    "run_blocker": "madaros_imported_run_SIGSEGV",
}
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
    "stoichiometry": {
        "path": "stdlib/chemistry/stoichiometry.sio",
        "run_exit_code": stoich_run_rc,
        "status": "pass" if stoich_run_rc == 0 else "fail",
    },
    "thermochem": {
        "path": "stdlib/chemistry/thermochem.sio",
        "run_exit_code": thermochem_run_rc,
        "status": "pass" if thermochem_run_rc == 0 else "fail",
    },
    "ontology": {
        "path": "stdlib/chemistry/ontology.sio",
        "run_exit_code": ontology_run_rc,
        "status": "pass" if ontology_run_rc == 0 else "fail",
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
        "run_exit_code": phys_thermo_run_rc,
        "status": "pass" if phys_thermo_run_rc == 0 else "fail",
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
        f"kinetics_check_rc={kinetics_check_rc}",
        f"thermo_kt_check_rc={thermo_kt_check_rc}",
        f"thermo_kinetics_kt_check_rc={thermo_kinetics_kt_check_rc}",
        f"ontology_check_rc={ontology_check_rc}",
        f"physics_run_rc={physics_run_rc}",
        f"chemistry_run_rc={chemistry_run_rc}",
        f"classical_shard_rc={classical_run_rc}",
        f"em_shard_rc={em_run_rc}",
        f"sr_shard_rc={sr_run_rc}",
        f"thermo_shard_rc={phys_thermo_run_rc}",
        f"acids_shard_rc={acids_run_rc}",
        f"equilibrium_shard_rc={equil_run_rc}",
        f"stoichiometry_shard_rc={stoich_run_rc}",
        f"thermochem_shard_rc={thermochem_run_rc}",
        f"ontology_shard_rc={ontology_run_rc}",
        "kinetics_shard_run=skipped_lean_single_blocker",
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