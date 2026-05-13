#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_JSON="${STDLIB_RELIABILITY_STATUS_OUT:-$ROOT_DIR/artifacts/stdlib/stdlib_reliability_status.v1.json}"
E2E_JSON="${STDLIB_RELIABILITY_E2E_JSON:-$ROOT_DIR/artifacts/stdlib/stdlib_e2e_result.v1.json}"
INVENTORY_JSON="${STDLIB_RELIABILITY_INVENTORY_JSON:-$ROOT_DIR/artifacts/stdlib/stdlib_inventory.v1.json}"
SCIENCE_STATUS_JSON="${STDLIB_RELIABILITY_SCIENCE_STATUS_JSON:-$ROOT_DIR/artifacts/stdlib/stdlib_science_pipeline_status.v1.json}"
HYPER_STATUS_JSON="${STDLIB_RELIABILITY_HYPER_STATUS_JSON:-$ROOT_DIR/artifacts/stdlib/stdlib_hyper_execution_status.v1.json}"
BASELINE_PASS="${STDLIB_RELIABILITY_BASELINE_PASS:-52}"
BASELINE_FAIL="${STDLIB_RELIABILITY_BASELINE_FAIL:-13}"
BASELINE_SKIP="${STDLIB_RELIABILITY_BASELINE_SKIP:-5}"
BASELINE_TOTAL="${STDLIB_RELIABILITY_BASELINE_TOTAL:-70}"
RUNTIME_REGRESSION_STRICT="${STDLIB_RUNTIME_REGRESSION_STRICT:-0}"

usage() {
  cat <<'USAGE'
Usage: bash scripts/stdlib_reliability_gate.sh [--out-json PATH] [--e2e-json PATH] [--inventory-json PATH] [--science-json PATH] [--hyper-json PATH]

Runs fail-closed STDLIB reliability gate:
  1) Collect stdlib inventory snapshot.
  2) Run full stdlib E2E with structured JSON output.
  3) Emit reliability status artifact.
  4) Exit non-zero on fail/not_run.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-json)
      if [[ $# -lt 2 ]]; then
        echo "error: --out-json requires a path" >&2
        exit 2
      fi
      OUT_JSON="$2"
      shift 2
      ;;
    --e2e-json)
      if [[ $# -lt 2 ]]; then
        echo "error: --e2e-json requires a path" >&2
        exit 2
      fi
      E2E_JSON="$2"
      shift 2
      ;;
    --inventory-json)
      if [[ $# -lt 2 ]]; then
        echo "error: --inventory-json requires a path" >&2
        exit 2
      fi
      INVENTORY_JSON="$2"
      shift 2
      ;;
    --science-json)
      if [[ $# -lt 2 ]]; then
        echo "error: --science-json requires a path" >&2
        exit 2
      fi
      SCIENCE_STATUS_JSON="$2"
      shift 2
      ;;
    --hyper-json)
      if [[ $# -lt 2 ]]; then
        echo "error: --hyper-json requires a path" >&2
        exit 2
      fi
      HYPER_STATUS_JSON="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$(dirname "$OUT_JSON")" "$(dirname "$E2E_JSON")" "$(dirname "$INVENTORY_JSON")"

emit_not_run() {
  local reason="$1"
  python3 - "$OUT_JSON" "$reason" <<'PY'
import datetime
import json
from pathlib import Path
import sys

out_path = Path(sys.argv[1])
reason = sys.argv[2]
obj = {
    "schema": "sounio.stdlib.reliability_status.v1",
    "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    "command": "bash scripts/stdlib_reliability_gate.sh",
    "totals": {"pass": 0, "fail": 0, "skip": 0, "total": 0},
    "status_summary": "not_run",
    "failures": [],
    "ignored": [],
    "contract_adjustments": [],
    "inventory": {"sio_files": 0, "disabled_files": 0, "stub_mod_files": 0, "active_module_entrypoints": 0},
    "science_pipeline": {
        "status": "not_run",
        "status_artifact": "",
        "runtime_regression_enforcement": "soft",
        "runtime_regression_summary_status": "not_run",
        "runtime_souc_bin": "",
        "runtime_souc_version": "",
        "runtime_pinned_version_expected": "",
    },
    "hyper_pipeline": {
        "status": "not_run",
        "status_artifact": "",
        "totals": {"pass": 0, "fail": 0, "skip": 0, "total": 0},
    },
    "notes": [reason],
}
out_path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
PY
}

validate_status_json() {
  python3 - "$OUT_JSON" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit("status artifact missing")

obj = json.loads(path.read_text(encoding="utf-8"))
required = [
    "schema",
    "generated_at_utc",
    "command",
    "totals",
    "status_summary",
    "failures",
    "ignored",
    "contract_adjustments",
    "inventory",
    "science_pipeline",
    "hyper_pipeline",
    "notes",
]
for key in required:
    if key not in obj:
        raise SystemExit(f"missing required key: {key}")

if obj["schema"] != "sounio.stdlib.reliability_status.v1":
    raise SystemExit("unexpected schema value")

if obj["status_summary"] not in {"pass", "fail", "not_run"}:
    raise SystemExit("status_summary must be pass|fail|not_run")

for key in ("pass", "fail", "skip", "total"):
    if key not in obj["totals"]:
        raise SystemExit(f"totals missing key: {key}")

for key in ("sio_files", "disabled_files", "stub_mod_files"):
    if key not in obj["inventory"]:
        raise SystemExit(f"inventory missing key: {key}")

if "status" not in obj["science_pipeline"]:
    raise SystemExit("science_pipeline missing status")
if obj["science_pipeline"]["status"] not in {"pass", "fail", "not_run"}:
    raise SystemExit("science_pipeline.status must be pass|fail|not_run")
for key in ("runtime_regression_enforcement", "runtime_regression_summary_status"):
    if key not in obj["science_pipeline"]:
        raise SystemExit(f"science_pipeline missing {key}")
for key in ("runtime_souc_bin", "runtime_souc_version", "runtime_pinned_version_expected"):
    if key not in obj["science_pipeline"]:
        raise SystemExit(f"science_pipeline missing {key}")
if obj["science_pipeline"]["runtime_regression_enforcement"] not in {"soft", "strict"}:
    raise SystemExit("science_pipeline.runtime_regression_enforcement must be soft|strict")
if obj["science_pipeline"]["runtime_regression_summary_status"] not in {"pass", "fail", "not_run"}:
    raise SystemExit("science_pipeline.runtime_regression_summary_status must be pass|fail|not_run")
if "runtime_regression_fail_count" in obj["science_pipeline"] and not isinstance(obj["science_pipeline"]["runtime_regression_fail_count"], int):
    raise SystemExit("science_pipeline.runtime_regression_fail_count must be int")

if "status" not in obj["hyper_pipeline"]:
    raise SystemExit("hyper_pipeline missing status")
if obj["hyper_pipeline"]["status"] not in {"pass", "fail", "not_run"}:
    raise SystemExit("hyper_pipeline.status must be pass|fail|not_run")
if "totals" not in obj["hyper_pipeline"]:
    raise SystemExit("hyper_pipeline missing totals")
for key in ("pass", "fail", "skip", "total"):
    if key not in obj["hyper_pipeline"]["totals"]:
        raise SystemExit(f"hyper_pipeline.totals missing key: {key}")
PY
}

load_science_status() {
  if [[ ! -s "$SCIENCE_STATUS_JSON" ]]; then
    return 1
  fi
  python3 - "$SCIENCE_STATUS_JSON" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
obj = json.loads(path.read_text(encoding="utf-8"))
if obj.get("schema") != "sounio.stdlib.science_pipeline_status.v1":
    raise SystemExit("unexpected science schema")
status = obj.get("status_summary")
if status not in {"pass", "fail", "not_run"}:
    raise SystemExit("invalid science status_summary")
runtime_enforcement = obj.get("runtime_regression_enforcement")
if runtime_enforcement not in {"soft", "strict"}:
    raise SystemExit("invalid science runtime_regression_enforcement")
runtime_summary = obj.get("runtime_regression_summary")
if not isinstance(runtime_summary, dict):
    raise SystemExit("invalid science runtime_regression_summary")
runtime_status = runtime_summary.get("status")
if runtime_status not in {"pass", "fail", "not_run"}:
    raise SystemExit("invalid science runtime_regression_summary.status")
runtime_fail = runtime_summary.get("fail")
if not isinstance(runtime_fail, int):
    raise SystemExit("invalid science runtime_regression_summary.fail")
runtime_provenance = obj.get("runtime_provenance")
if not isinstance(runtime_provenance, dict):
    raise SystemExit("invalid science runtime_provenance")
runtime_souc_bin = runtime_provenance.get("souc_bin")
runtime_souc_version = runtime_provenance.get("souc_version")
runtime_pinned_expected = runtime_provenance.get("pinned_version_expected")
if not all(isinstance(v, str) for v in (runtime_souc_bin, runtime_souc_version, runtime_pinned_expected)):
    raise SystemExit("invalid science runtime_provenance fields")
print(f"{status}\t{runtime_enforcement}\t{runtime_status}\t{runtime_fail}\t{runtime_souc_bin}\t{runtime_souc_version}\t{runtime_pinned_expected}")
PY
}

load_hyper_status() {
  if [[ ! -s "$HYPER_STATUS_JSON" ]]; then
    return 1
  fi
  python3 - "$HYPER_STATUS_JSON" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
obj = json.loads(path.read_text(encoding="utf-8"))
if obj.get("schema") != "sounio.stdlib.hyper_execution_status.v1":
    raise SystemExit("unexpected hyper schema")
status = obj.get("status_summary")
if status not in {"pass", "fail", "not_run"}:
    raise SystemExit("invalid hyper status_summary")
totals = obj.get("totals")
if not isinstance(totals, dict):
    raise SystemExit("invalid hyper totals")
for key in ("pass", "fail", "skip", "total"):
    if key not in totals or not isinstance(totals[key], int):
        raise SystemExit(f"invalid hyper totals.{key}")
print(f"{status}\t{totals['pass']}\t{totals['fail']}\t{totals['skip']}\t{totals['total']}")
PY
}

echo "[stdlib-reliability-gate] collecting inventory snapshot"
set +e
bash "$ROOT_DIR/scripts/stdlib/scan_stdlib.sh" --json-out "$INVENTORY_JSON" --quiet
scan_rc=$?
set -e
if [[ $scan_rc -ne 0 || ! -s "$INVENTORY_JSON" ]]; then
  emit_not_run "inventory_scan_failed"
  validate_status_json || true
  echo "error: inventory scan failed (rc=$scan_rc)" >&2
  exit 1
fi

echo "[stdlib-reliability-gate] running hyper execution gate"
set +e
bash "$ROOT_DIR/scripts/stdlib/stdlib_hyper_execution_gate.sh" --out-json "$HYPER_STATUS_JSON"
hyper_gate_rc=$?
set -e
if [[ $hyper_gate_rc -ne 0 || ! -s "$HYPER_STATUS_JSON" ]]; then
  emit_not_run "hyper_gate_failed_or_missing"
  validate_status_json || true
  echo "error: hyper gate failed or status missing: $HYPER_STATUS_JSON (rc=$hyper_gate_rc)" >&2
  exit 1
fi

echo "[stdlib-reliability-gate] reading hyper gate status"
set +e
HYPER_STATUS_ROW="$(load_hyper_status)"
hyper_status_rc=$?
set -e
if [[ $hyper_status_rc -ne 0 ]]; then
  emit_not_run "hyper_status_missing_or_invalid"
  validate_status_json || true
  echo "error: hyper status missing/invalid: $HYPER_STATUS_JSON" >&2
  exit 1
fi
IFS=$'\t' read -r HYPER_STATUS HYPER_PASS HYPER_FAIL HYPER_SKIP HYPER_TOTAL <<<"$HYPER_STATUS_ROW"

echo "[stdlib-reliability-gate] reading science gate status"
set +e
SCIENCE_STATUS_ROW="$(load_science_status)"
science_rc=$?
set -e
if [[ $science_rc -ne 0 ]]; then
  emit_not_run "science_status_missing_or_invalid"
  validate_status_json || true
  echo "error: science status missing/invalid: $SCIENCE_STATUS_JSON" >&2
  exit 1
fi
IFS=$'\t' read -r SCIENCE_STATUS SCIENCE_RUNTIME_ENFORCEMENT SCIENCE_RUNTIME_STATUS SCIENCE_RUNTIME_FAIL SCIENCE_RUNTIME_SOUC_BIN SCIENCE_RUNTIME_SOUC_VERSION SCIENCE_RUNTIME_PINNED_EXPECTED <<<"$SCIENCE_STATUS_ROW"

echo "[stdlib-reliability-gate] running stdlib e2e suite"
set +e
bash "$ROOT_DIR/scripts/stdlib/run_stdlib_e2e.sh" --json-out "$E2E_JSON"
e2e_rc=$?
set -e
if [[ ! -s "$E2E_JSON" ]]; then
  emit_not_run "stdlib_e2e_json_missing"
  validate_status_json || true
  echo "error: stdlib e2e JSON artifact missing: $E2E_JSON" >&2
  exit 1
fi

set +e
python3 - "$ROOT_DIR" "$E2E_JSON" "$INVENTORY_JSON" "$OUT_JSON" "$e2e_rc" \
  "$BASELINE_PASS" "$BASELINE_FAIL" "$BASELINE_SKIP" "$BASELINE_TOTAL" \
  "$HYPER_STATUS" "$HYPER_STATUS_JSON" "$HYPER_PASS" "$HYPER_FAIL" "$HYPER_SKIP" "$HYPER_TOTAL" \
  "$SCIENCE_STATUS" "$SCIENCE_STATUS_JSON" "$SCIENCE_RUNTIME_ENFORCEMENT" "$SCIENCE_RUNTIME_STATUS" "$SCIENCE_RUNTIME_FAIL" "$RUNTIME_REGRESSION_STRICT" \
  "$SCIENCE_RUNTIME_SOUC_BIN" "$SCIENCE_RUNTIME_SOUC_VERSION" "$SCIENCE_RUNTIME_PINNED_EXPECTED" <<'PY'
import datetime
import json
import re
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
e2e_path = Path(sys.argv[2]).resolve()
inventory_path = Path(sys.argv[3]).resolve()
out_path = Path(sys.argv[4]).resolve()
e2e_rc = int(sys.argv[5])
baseline_pass = int(sys.argv[6])
baseline_fail = int(sys.argv[7])
baseline_skip = int(sys.argv[8])
baseline_total = int(sys.argv[9])
hyper_status = sys.argv[10]
hyper_status_json = Path(sys.argv[11]).resolve()
hyper_pass = int(sys.argv[12])
hyper_fail = int(sys.argv[13])
hyper_skip = int(sys.argv[14])
hyper_total = int(sys.argv[15])
science_status = sys.argv[16]
science_status_json = Path(sys.argv[17]).resolve()
science_runtime_enforcement = sys.argv[18]
science_runtime_status = sys.argv[19]
science_runtime_fail = int(sys.argv[20])
runtime_regression_strict = sys.argv[21] == "1"
science_runtime_souc_bin = sys.argv[22]
science_runtime_souc_version = sys.argv[23]
science_runtime_pinned_expected = sys.argv[24]

try:
    e2e = json.loads(e2e_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"failed to parse e2e/inventory JSON: {exc}")

totals = e2e.get("totals") or {}
pass_count = int(totals.get("pass", 0))
fail_count = int(totals.get("fail", 0))
skip_count = int(totals.get("skip", 0))
total_count = int(totals.get("total", pass_count + fail_count + skip_count))

inv_counts = inventory.get("counts") or {}
sio_files = int(inv_counts.get("sio_files", 0))
disabled_files = int(inv_counts.get("disabled_files", 0))
stub_mod_files = int(inv_counts.get("stub_mod_files", 0))
active_entrypoints = int(inv_counts.get("active_module_entrypoints", 0))

stub_paths = set(inventory.get("stub_mod_paths") or [])
stub_roots = {Path(p).parts[0] for p in stub_paths if Path(p).parts}

disabled_roots = set()
stdlib_root = Path(inventory.get("stdlib_path") or root / "stdlib")
if stdlib_root.exists():
    for path in stdlib_root.rglob("*.sio.disabled"):
        parts = path.relative_to(stdlib_root).parts
        if parts:
            disabled_roots.add(parts[0])

def test_module_root(test_path: str) -> str:
    parts = Path(test_path).parts
    if len(parts) >= 3 and parts[0] == "tests" and parts[1] == "stdlib":
        return parts[2]
    return ""

def categorize_failure(test_path: str, excerpt: str) -> str:
    text = (excerpt or "").lower()
    module_root = test_module_root(test_path)

    if any(
        token in text
        for token in (
            "could not import",
            "unresolved import",
            "cannot import",
            "module not found",
            "missing module",
            "cannot find module",
            "failed to read module",
            "no such file",
        )
    ):
        base = "missing_import_target"
    elif any(
        token in text
        for token in (
            "unresolved symbol",
            "unknown symbol",
            "cannot find value",
            "could not resolve",
            "undefined",
            "not found in module",
            "no member named",
            "not found in scope",
        )
    ):
        base = "missing_symbol"
    elif any(token in text for token in ("disabled", ".sio.disabled")):
        base = "disabled_module_surface"
    elif any(token in text for token in ("stub", "no exported api", "not implemented")):
        base = "stub_module_surface"
    else:
        base = "other"

    if base in {"missing_import_target", "missing_symbol", "other"}:
        if module_root and module_root in stub_roots:
            return "stub_module_surface"
        if module_root and module_root in disabled_roots:
            return "disabled_module_surface"
    return base

raw_failures = e2e.get("failures") or []
if not raw_failures:
    raw_failures = [row for row in (e2e.get("results") or []) if row.get("status") == "fail"]

failures = []
for row in raw_failures:
    test_path = row.get("test_path", "")
    excerpt = row.get("detail_excerpt", "") or row.get("reason", "")
    failures.append(
        {
            "test_path": test_path,
            "category": categorize_failure(test_path, excerpt),
            "error_excerpt": (excerpt or "")[:400],
        }
    )

ignored = []
for row in (e2e.get("ignored") or []):
    ignored.append(
        {
            "test_path": row.get("test_path", ""),
            "reason": row.get("reason", "") or "ignored",
            "owner": row.get("owner", "") or "unowned",
            "unblock_condition": row.get("unblock_condition", "") or "unspecified",
        }
    )
ignored.sort(key=lambda x: x["test_path"])

contract_adjustments = []
pattern = re.compile(r"^\s*//@\s*contract-adjustment:\s*(.+?)\s*$")
for path in sorted((root / "tests" / "stdlib").rglob("*.sio")):
    rel = path.relative_to(root).as_posix()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            contract_adjustments.append(
                {
                    "test_path": rel,
                    "adjustment": match.group(1),
                }
            )
            break

if e2e_rc > 1:
    status_summary = "not_run"
elif fail_count > 0:
    status_summary = "fail"
elif hyper_status != "pass":
    status_summary = "fail"
elif science_status != "pass":
    status_summary = "fail"
elif runtime_regression_strict and science_runtime_enforcement != "strict":
    status_summary = "fail"
elif runtime_regression_strict and science_runtime_status != "pass":
    status_summary = "fail"
else:
    status_summary = "pass"

notes = [
    f"e2e_exit_code={e2e_rc}",
    f"baseline_reference={{pass:{baseline_pass},fail:{baseline_fail},skip:{baseline_skip},total:{baseline_total}}}",
    (
        "baseline_delta={"
        f"pass:{pass_count - baseline_pass},"
        f"fail:{fail_count - baseline_fail},"
        f"skip:{skip_count - baseline_skip},"
        f"total:{total_count - baseline_total}"
        "}"
    ),
]

if e2e_rc == 1 and fail_count == 0:
    notes.append("warning: e2e exit code indicates failure but fail count is zero")
if e2e_rc == 0 and fail_count > 0:
    notes.append("warning: e2e exit code indicates success but fail count is non-zero")
notes.append(f"hyper_gate_required_pass_observed={hyper_status}")
notes.append(f"hyper_totals=pass:{hyper_pass},fail:{hyper_fail},skip:{hyper_skip},total:{hyper_total}")
if science_status != "pass":
    notes.append(f"science_gate_required_pass_observed={science_status}")
notes.append(f"science_runtime_regression_enforcement={science_runtime_enforcement}")
notes.append(f"science_runtime_regression_summary_status={science_runtime_status}")
notes.append(f"science_runtime_regression_fail_count={science_runtime_fail}")
notes.append(f"science_runtime_souc_bin={science_runtime_souc_bin}")
notes.append(f"science_runtime_souc_version={science_runtime_souc_version}")
notes.append(f"science_runtime_pinned_expected={science_runtime_pinned_expected}")
notes.append(f"runtime_regression_strict_requested={runtime_regression_strict}")
if runtime_regression_strict and science_runtime_enforcement != "strict":
    notes.append("runtime_regression_strict_mode_not_propagated")
if runtime_regression_strict and science_runtime_status != "pass":
    notes.append("runtime_regression_strict_mode_failed")

def rel_or_abs(path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return str(path)

obj = {
    "schema": "sounio.stdlib.reliability_status.v1",
    "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    "command": "bash scripts/stdlib_reliability_gate.sh",
    "totals": {
        "pass": pass_count,
        "fail": fail_count,
        "skip": skip_count,
        "total": total_count,
    },
    "status_summary": status_summary,
    "failures": failures,
    "ignored": ignored,
    "contract_adjustments": contract_adjustments,
    "inventory": {
        "sio_files": sio_files,
        "disabled_files": disabled_files,
        "stub_mod_files": stub_mod_files,
        "active_module_entrypoints": active_entrypoints,
    },
    "science_pipeline": {
        "status": science_status,
        "status_artifact": rel_or_abs(science_status_json),
        "runtime_regression_enforcement": science_runtime_enforcement,
        "runtime_regression_summary_status": science_runtime_status,
        "runtime_regression_fail_count": science_runtime_fail,
        "runtime_souc_bin": science_runtime_souc_bin,
        "runtime_souc_version": science_runtime_souc_version,
        "runtime_pinned_version_expected": science_runtime_pinned_expected,
    },
    "hyper_pipeline": {
        "status": hyper_status,
        "status_artifact": rel_or_abs(hyper_status_json),
        "totals": {
            "pass": hyper_pass,
            "fail": hyper_fail,
            "skip": hyper_skip,
            "total": hyper_total,
        },
    },
    "notes": notes,
}

out_path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
PY
aggregate_rc=$?
set -e

if [[ $aggregate_rc -ne 0 ]]; then
  emit_not_run "status_aggregation_failed"
  validate_status_json || true
  echo "error: failed to aggregate reliability status (rc=$aggregate_rc)" >&2
  exit 1
fi

if ! validate_status_json; then
  emit_not_run "status_json_validation_failed"
  validate_status_json || true
  echo "error: produced malformed reliability artifact" >&2
  exit 1
fi

status_summary="$(python3 - "$OUT_JSON" <<'PY'
import json
import pathlib
import sys

obj = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(obj.get("status_summary", "not_run"))
PY
)"

echo "[stdlib-reliability-gate] status_json=${OUT_JSON#$ROOT_DIR/}"
echo "[stdlib-reliability-gate] inventory_json=${INVENTORY_JSON#$ROOT_DIR/}"
echo "[stdlib-reliability-gate] e2e_json=${E2E_JSON#$ROOT_DIR/}"
echo "[stdlib-reliability-gate] hyper_json=${HYPER_STATUS_JSON#$ROOT_DIR/}"
echo "[stdlib-reliability-gate] hyper_status=${HYPER_STATUS}"
echo "[stdlib-reliability-gate] science_json=${SCIENCE_STATUS_JSON#$ROOT_DIR/}"
echo "[stdlib-reliability-gate] science_status=${SCIENCE_STATUS}"
echo "[stdlib-reliability-gate] science_runtime_enforcement=${SCIENCE_RUNTIME_ENFORCEMENT}"
echo "[stdlib-reliability-gate] science_runtime_status=${SCIENCE_RUNTIME_STATUS}"
echo "[stdlib-reliability-gate] status_summary=$status_summary"

if [[ "$status_summary" == "pass" ]]; then
  echo "STDLIB_RELIABILITY_GATE_PASS"
  exit 0
fi

echo "error: STDLIB_RELIABILITY_GATE_${status_summary^^}" >&2
exit 1
