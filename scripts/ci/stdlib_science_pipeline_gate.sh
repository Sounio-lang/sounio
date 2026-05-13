#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_JSON="${STDLIB_SCIENCE_STATUS_OUT:-$ROOT_DIR/artifacts/stdlib/stdlib_science_pipeline_status.v1.json}"
MANIFEST_JSON="${STDLIB_SCIENCE_FIXTURE_MANIFEST:-$ROOT_DIR/tests/fixtures/fmri/fixture_manifest.v1.json}"
GOLDEN_JSON="${STDLIB_SCIENCE_GOLDEN:-$ROOT_DIR/tests/fixtures/fmri/pipeline_golden.v1.json}"
FMRI_TEST="${STDLIB_SCIENCE_FMRI_TEST:-$ROOT_DIR/tests/stdlib/fmri/test_pipeline_real_e2e.sio}"
PBPK_TEST="${STDLIB_SCIENCE_PBPK_TEST:-$ROOT_DIR/tests/stdlib/darwin_pbpk/test_pipeline_real_e2e.sio}"
RUNTIME_LITERAL_SRC="${STDLIB_RUNTIME_PROBE_LITERAL:-$ROOT_DIR/tests/stdlib/runtime_regression/runtime_literal_as_bytes.sio}"
RUNTIME_TEXT_SRC="${STDLIB_RUNTIME_PROBE_TEXT:-$ROOT_DIR/tests/stdlib/runtime_regression/runtime_text_as_bytes.sio}"
RUNTIME_BINARY_SRC="${STDLIB_RUNTIME_PROBE_BINARY:-$ROOT_DIR/tests/stdlib/runtime_regression/runtime_binary_as_bytes.sio}"
RUNTIME_SLICE_SRC="${STDLIB_RUNTIME_PROBE_SLICE:-$ROOT_DIR/tests/stdlib/runtime_regression/runtime_dynamic_slice.sio}"
RUNTIME_REGRESSION_STRICT="${STDLIB_RUNTIME_REGRESSION_STRICT:-0}"
RUNTIME_SOUC_BIN=""
RUNTIME_SOUC_VERSION=""
RUNTIME_PINNED_VERSION_EXPECTED="${SOUNIO_SOUC_VERSION:-}"
RUNTIME_REGRESSION_ENFORCEMENT="soft"
if [[ "$RUNTIME_REGRESSION_STRICT" == "1" ]]; then
  RUNTIME_REGRESSION_ENFORCEMENT="strict"
fi

usage() {
  cat <<'USAGE'
Usage: bash scripts/stdlib_science_pipeline_gate.sh [--out-json PATH] [--manifest PATH] [--golden PATH]

Runs the fail-closed STDLIB science pipeline gate:
  1) Verify fixture manifest and SHA256 hashes.
  2) Enforce no //@ ignore annotations in fmri/darwin_pbpk stdlib tests.
  3) Execute real run-pass science tests for fmri and darwin_pbpk.
  4) Parse SCIENCE_METRIC lines and compare against pinned golden values.
  5) Emit artifacts/stdlib/stdlib_science_pipeline_status.v1.json.
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
    --manifest)
      if [[ $# -lt 2 ]]; then
        echo "error: --manifest requires a path" >&2
        exit 2
      fi
      MANIFEST_JSON="$2"
      shift 2
      ;;
    --golden)
      if [[ $# -lt 2 ]]; then
        echo "error: --golden requires a path" >&2
        exit 2
      fi
      GOLDEN_JSON="$2"
      shift 2
      ;;
    --help|-h)
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

mkdir -p "$(dirname "$OUT_JSON")"

emit_not_run() {
  local reason="$1"
  python3 - "$OUT_JSON" "$reason" "$RUNTIME_REGRESSION_ENFORCEMENT" "$RUNTIME_SOUC_BIN" "$RUNTIME_SOUC_VERSION" "$RUNTIME_PINNED_VERSION_EXPECTED" <<'PY'
import datetime
import json
from pathlib import Path
import sys

out_path = Path(sys.argv[1])
reason = sys.argv[2]
runtime_enforcement = sys.argv[3]
souc_bin = sys.argv[4]
souc_version = sys.argv[5]
pinned_version = sys.argv[6]
obj = {
    "schema": "sounio.stdlib.science_pipeline_status.v1",
    "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    "command": "bash scripts/stdlib_science_pipeline_gate.sh",
    "status_summary": "not_run",
    "totals": {"pass": 0, "fail": 0, "not_run": 2, "total": 2},
    "fixture_manifest": "",
    "golden_file": "",
    "tolerance": {"abs": 0.0, "rel": 0.0},
    "lanes": {
        "fmri": {"status": "not_run", "test_path": "", "exit_code": 125, "marker_found": False, "metrics": {}, "mismatches": []},
        "darwin_pbpk": {"status": "not_run", "test_path": "", "exit_code": 125, "marker_found": False, "metrics": {}, "mismatches": []},
    },
    "fixture_verification": {"verified": False, "entries": []},
    "ignored_policy_violation": {"detected": False, "hits": []},
    "runtime_regressions": {
        "literal_as_bytes": {"status": "not_run", "check_exit_code": 125, "run_exit_code": 125, "start_marker_found": False, "after_marker_found": False, "output_excerpt": ""},
        "text_as_bytes": {"status": "not_run", "check_exit_code": 125, "run_exit_code": 125, "start_marker_found": False, "after_marker_found": False, "output_excerpt": ""},
        "binary_as_bytes": {"status": "not_run", "check_exit_code": 125, "run_exit_code": 125, "start_marker_found": False, "after_marker_found": False, "output_excerpt": ""},
        "dynamic_slice": {"status": "not_run", "check_exit_code": 125, "run_exit_code": 125, "start_marker_found": False, "after_marker_found": False, "output_excerpt": ""},
    },
    "runtime_regression_enforcement": runtime_enforcement,
    "runtime_regression_summary": {"status": "not_run", "pass": 0, "fail": 0, "not_run": 4, "total": 4},
    "runtime_provenance": {
        "souc_bin": souc_bin,
        "souc_version": souc_version,
        "pinned_version_expected": pinned_version,
    },
    "failures": [{"lane": "gate", "kind": "not_run", "message": reason}],
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
    "status_summary",
    "totals",
    "fixture_manifest",
    "golden_file",
    "tolerance",
    "lanes",
    "fixture_verification",
    "ignored_policy_violation",
    "runtime_regressions",
    "runtime_regression_enforcement",
    "runtime_regression_summary",
    "runtime_provenance",
    "failures",
    "notes",
]
for key in required:
    if key not in obj:
        raise SystemExit(f"missing required key: {key}")

if obj["schema"] != "sounio.stdlib.science_pipeline_status.v1":
    raise SystemExit("unexpected schema value")

if obj["status_summary"] not in {"pass", "fail", "not_run"}:
    raise SystemExit("status_summary must be pass|fail|not_run")

if obj["runtime_regression_enforcement"] not in {"soft", "strict"}:
    raise SystemExit("runtime_regression_enforcement must be soft|strict")

rr = obj.get("runtime_regressions")
if not isinstance(rr, dict):
    raise SystemExit("runtime_regressions must be an object")
for key in ("literal_as_bytes", "text_as_bytes", "binary_as_bytes", "dynamic_slice"):
    if key not in rr:
        raise SystemExit(f"runtime_regressions missing key: {key}")

rr_summary = obj.get("runtime_regression_summary")
if not isinstance(rr_summary, dict):
    raise SystemExit("runtime_regression_summary must be an object")
if rr_summary.get("status") not in {"pass", "fail", "not_run"}:
    raise SystemExit("runtime_regression_summary.status must be pass|fail|not_run")

runtime_provenance = obj.get("runtime_provenance")
if not isinstance(runtime_provenance, dict):
    raise SystemExit("runtime_provenance must be an object")
for key in ("souc_bin", "souc_version", "pinned_version_expected"):
    if key not in runtime_provenance:
        raise SystemExit(f"runtime_provenance missing key: {key}")
PY
}

if [[ ! -f "$ROOT_DIR/scripts/lib/resolve_souc.sh" ]]; then
  emit_not_run "resolve_souc_script_missing"
  validate_status_json || true
  echo "error: missing scripts/lib/resolve_souc.sh" >&2
  exit 1
fi

# shellcheck source=lib/resolve_souc.sh
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

set +e
sounio_require_souc
souc_rc=$?
set -e
if [[ $souc_rc -ne 0 ]]; then
  emit_not_run "souc_unavailable"
  validate_status_json || true
  echo "error: souc binary resolution failed" >&2
  exit 1
fi

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
RUNTIME_SOUC_BIN="$SOUC_BIN"
RUNTIME_SOUC_VERSION="$("$SOUC_BIN" --version 2>/dev/null | awk 'NR==1 {print $2}')"
if [[ -z "$RUNTIME_SOUC_VERSION" ]]; then
  RUNTIME_SOUC_VERSION="unknown"
fi
if [[ -z "$RUNTIME_PINNED_VERSION_EXPECTED" ]]; then
  RUNTIME_PINNED_VERSION_EXPECTED="$(python3 - "$ROOT_DIR/scripts/omega/omega_resolve_souc_bin.sh" <<'PY'
import re
from pathlib import Path
import sys

path = Path(sys.argv[1])
src = path.read_text(encoding="utf-8", errors="replace")
match = re.search(r'SOUC_VERSION="\$\{SOUNIO_SOUC_VERSION:-([^}]+)\}"', src)
if match:
    print(match.group(1))
PY
)"
fi
if [[ -z "$RUNTIME_PINNED_VERSION_EXPECTED" ]]; then
  RUNTIME_PINNED_VERSION_EXPECTED="unknown"
fi

if [[ ! -f "$FMRI_TEST" || ! -f "$PBPK_TEST" ]]; then
  emit_not_run "science_tests_missing"
  validate_status_json || true
  echo "error: required science tests are missing" >&2
  exit 1
fi
for probe_src in "$RUNTIME_LITERAL_SRC" "$RUNTIME_TEXT_SRC" "$RUNTIME_BINARY_SRC" "$RUNTIME_SLICE_SRC"; do
  if [[ ! -f "$probe_src" ]]; then
    emit_not_run "runtime_probe_source_missing"
    validate_status_json || true
    echo "error: missing runtime probe source: $probe_src" >&2
    exit 1
  fi
done

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
FMRI_OUT="$TMP_DIR/fmri.out"
PBPK_OUT="$TMP_DIR/pbpk.out"
IGNORE_HITS_FILE="$TMP_DIR/ignore_hits.txt"
MANIFEST_VERIFY_LOG="$TMP_DIR/manifest_verify.log"
RUNTIME_LITERAL_OUT="$TMP_DIR/runtime_literal_as_bytes.out"
RUNTIME_TEXT_OUT="$TMP_DIR/runtime_text_as_bytes.out"
RUNTIME_BINARY_OUT="$TMP_DIR/runtime_binary_as_bytes.out"
RUNTIME_SLICE_OUT="$TMP_DIR/runtime_dynamic_slice.out"

run_runtime_probe() {
  local src="$1"
  local out="$2"
  local check_var="$3"
  local run_var="$4"
  local check_rc=125
  local run_rc=125
  set +e
  "$SOUC_BIN" check "$src" >"$out" 2>&1
  check_rc=$?
  if [[ $check_rc -eq 0 ]]; then
    "$SOUC_BIN" run "$src" >>"$out" 2>&1
    run_rc=$?
  fi
  set -e
  printf -v "$check_var" '%s' "$check_rc"
  printf -v "$run_var" '%s' "$run_rc"
}

literal_check_rc=125
literal_run_rc=125
text_check_rc=125
text_run_rc=125
binary_check_rc=125
binary_run_rc=125
slice_check_rc=125
slice_run_rc=125

run_runtime_probe "$RUNTIME_LITERAL_SRC" "$RUNTIME_LITERAL_OUT" literal_check_rc literal_run_rc
run_runtime_probe "$RUNTIME_TEXT_SRC" "$RUNTIME_TEXT_OUT" text_check_rc text_run_rc
run_runtime_probe "$RUNTIME_BINARY_SRC" "$RUNTIME_BINARY_OUT" binary_check_rc binary_run_rc
run_runtime_probe "$RUNTIME_SLICE_SRC" "$RUNTIME_SLICE_OUT" slice_check_rc slice_run_rc

# Hard policy: no ignore annotations in these lanes.
rg -n '^[[:space:]]*//@[[:space:]]*ignore([[:space:]].*)?$' tests/stdlib/fmri tests/stdlib/darwin_pbpk >"$IGNORE_HITS_FILE" || true
ignore_violation=0
if [[ -s "$IGNORE_HITS_FILE" ]]; then
  ignore_violation=1
fi

# Verify fixture manifest + hashes BEFORE execution.
manifest_ok=1
python3 - "$MANIFEST_JSON" "$ROOT_DIR" >"$MANIFEST_VERIFY_LOG" 2>&1 <<'PY'
import hashlib
import json
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()

if not manifest_path.exists():
    raise SystemExit(f"manifest_missing:{manifest_path}")

obj = json.loads(manifest_path.read_text(encoding="utf-8"))
fixtures = obj.get("fixtures")
if not isinstance(fixtures, list) or not fixtures:
    raise SystemExit("manifest_invalid:fixtures")

for fixture in fixtures:
    rel = fixture.get("path")
    expected = fixture.get("sha256")
    if not isinstance(rel, str) or not rel:
        raise SystemExit("manifest_invalid:path")
    if not isinstance(expected, str) or not expected:
        raise SystemExit(f"manifest_invalid:sha256:{rel}")
    file_path = (root / rel).resolve()
    if not file_path.exists():
        raise SystemExit(f"fixture_missing:{rel}")
    actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"fixture_hash_mismatch:{rel}:expected={expected}:actual={actual}")

print("manifest_ok")
PY
if [[ $? -ne 0 ]]; then
  manifest_ok=0
fi

fmri_rc=125
pbpk_rc=125

if [[ $manifest_ok -eq 1 && $ignore_violation -eq 0 ]]; then
  set +e
  "$SOUC_BIN" check "$FMRI_TEST" >"$FMRI_OUT" 2>&1
  fmri_rc=$?
  if [[ $fmri_rc -eq 0 ]]; then
    "$SOUC_BIN" run "$FMRI_TEST" >>"$FMRI_OUT" 2>&1
    fmri_rc=$?
  fi

  "$SOUC_BIN" check "$PBPK_TEST" >"$PBPK_OUT" 2>&1
  pbpk_rc=$?
  if [[ $pbpk_rc -eq 0 ]]; then
    "$SOUC_BIN" run "$PBPK_TEST" >>"$PBPK_OUT" 2>&1
    pbpk_rc=$?
  fi
  set -e
fi

python3 - "$ROOT_DIR" "$OUT_JSON" "$MANIFEST_JSON" "$GOLDEN_JSON" "$FMRI_TEST" "$PBPK_TEST" "$FMRI_OUT" "$PBPK_OUT" "$fmri_rc" "$pbpk_rc" "$manifest_ok" "$ignore_violation" "$IGNORE_HITS_FILE" "$MANIFEST_VERIFY_LOG" "$RUNTIME_REGRESSION_ENFORCEMENT" "$literal_check_rc" "$literal_run_rc" "$RUNTIME_LITERAL_OUT" "$text_check_rc" "$text_run_rc" "$RUNTIME_TEXT_OUT" "$binary_check_rc" "$binary_run_rc" "$RUNTIME_BINARY_OUT" "$slice_check_rc" "$slice_run_rc" "$RUNTIME_SLICE_OUT" "$RUNTIME_SOUC_BIN" "$RUNTIME_SOUC_VERSION" "$RUNTIME_PINNED_VERSION_EXPECTED" <<'PY'
import datetime
import hashlib
import json
import math
import re
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
out_path = Path(sys.argv[2]).resolve()
manifest_path = Path(sys.argv[3]).resolve()
golden_path = Path(sys.argv[4]).resolve()
fmri_test = Path(sys.argv[5]).resolve()
pbpk_test = Path(sys.argv[6]).resolve()
fmri_out_path = Path(sys.argv[7]).resolve()
pbpk_out_path = Path(sys.argv[8]).resolve()
fmri_rc = int(sys.argv[9])
pbpk_rc = int(sys.argv[10])
manifest_ok_flag = int(sys.argv[11])
ignore_violation = int(sys.argv[12])
ignore_hits_path = Path(sys.argv[13]).resolve()
manifest_verify_log = Path(sys.argv[14]).resolve()
runtime_regression_enforcement = sys.argv[15]
literal_check_rc = int(sys.argv[16])
literal_run_rc = int(sys.argv[17])
literal_out_path = Path(sys.argv[18]).resolve()
text_check_rc = int(sys.argv[19])
text_run_rc = int(sys.argv[20])
text_out_path = Path(sys.argv[21]).resolve()
binary_check_rc = int(sys.argv[22])
binary_run_rc = int(sys.argv[23])
binary_out_path = Path(sys.argv[24]).resolve()
slice_check_rc = int(sys.argv[25])
slice_run_rc = int(sys.argv[26])
slice_out_path = Path(sys.argv[27]).resolve()
runtime_souc_bin = sys.argv[28]
runtime_souc_version = sys.argv[29]
runtime_pinned_version_expected = sys.argv[30]

now = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

failures = []
notes = []

if ignore_hits_path.exists():
    ignore_hits = [line for line in ignore_hits_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
else:
    ignore_hits = []

if ignore_violation:
    failures.append({
        "lane": "policy",
        "kind": "ignore_annotation",
        "message": "ignore annotations are forbidden in tests/stdlib/fmri and tests/stdlib/darwin_pbpk",
    })

manifest_entries = []
manifest_verified = False
manifest_obj = {}
try:
    manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixtures = manifest_obj.get("fixtures")
    if isinstance(fixtures, list) and fixtures:
        manifest_verified = True
        for fixture in fixtures:
            rel = fixture.get("path")
            expected = fixture.get("sha256")
            entry = {
                "path": rel or "",
                "expected_sha256": expected or "",
                "actual_sha256": "",
                "status": "fail",
            }
            if not rel or not expected:
                entry["status"] = "fail"
                entry["error"] = "manifest_entry_missing_path_or_sha256"
                manifest_verified = False
            else:
                file_path = (root / rel).resolve()
                if not file_path.exists():
                    entry["status"] = "fail"
                    entry["error"] = "missing_file"
                    manifest_verified = False
                else:
                    actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
                    entry["actual_sha256"] = actual
                    if actual == expected:
                        entry["status"] = "pass"
                    else:
                        entry["status"] = "fail"
                        entry["error"] = "hash_mismatch"
                        manifest_verified = False
            manifest_entries.append(entry)
    else:
        failures.append({"lane": "fixtures", "kind": "manifest_invalid", "message": "fixtures list missing or empty"})
except Exception as exc:
    failures.append({"lane": "fixtures", "kind": "manifest_parse_error", "message": str(exc)})

if not manifest_verified:
    failures.append({"lane": "fixtures", "kind": "manifest_verification_failed", "message": "fixture manifest/hash verification failed"})

if manifest_ok_flag == 0:
    log_excerpt = ""
    if manifest_verify_log.exists():
        lines = manifest_verify_log.read_text(encoding="utf-8", errors="replace").splitlines()
        log_excerpt = " | ".join(lines[:3])
    if log_excerpt:
        notes.append(f"manifest_precheck={log_excerpt}")

try:
    golden_obj = json.loads(golden_path.read_text(encoding="utf-8"))
except Exception as exc:
    golden_obj = {}
    failures.append({"lane": "gate", "kind": "golden_parse_error", "message": str(exc)})

abs_tol = float((golden_obj.get("tolerance") or {}).get("abs", 1e-6))
rel_tol = float((golden_obj.get("tolerance") or {}).get("rel", 1e-6))

def parse_metrics(text: str):
    metrics = {"fmri": {}, "darwin_pbpk": {}}
    pattern = re.compile(r"SCIENCE_METRIC\s+(fmri|darwin_pbpk)\s+([A-Za-z0-9_]+)\s+([-+0-9.eE]+)")
    for lane, key, raw_val in pattern.findall(text):
        try:
            metrics[lane][key] = float(raw_val)
        except Exception:
            pass
    return metrics

def runtime_probe_status(check_rc: int, run_rc: int, output_path: Path, start_marker: str, after_marker: str):
    text = output_path.read_text(encoding="utf-8", errors="replace") if output_path.exists() else ""
    start_found = start_marker in text
    after_found = after_marker in text

    if check_rc == 125 and run_rc == 125:
        status = "not_run"
    elif check_rc != 0 or run_rc != 0 or not start_found or not after_found:
        status = "fail"
    else:
        status = "pass"

    lines = [line for line in text.splitlines() if line.strip()]
    excerpt = " | ".join(lines[:4])
    return {
        "status": status,
        "check_exit_code": check_rc,
        "run_exit_code": run_rc,
        "start_marker_found": start_found,
        "after_marker_found": after_found,
        "output_excerpt": excerpt,
    }

runtime_regressions = {
    "literal_as_bytes": runtime_probe_status(
        literal_check_rc,
        literal_run_rc,
        literal_out_path,
        "RUNTIME_PROBE_LITERAL_START",
        "RUNTIME_PROBE_LITERAL_AFTER",
    ),
    "text_as_bytes": runtime_probe_status(
        text_check_rc,
        text_run_rc,
        text_out_path,
        "RUNTIME_PROBE_TEXT_START",
        "RUNTIME_PROBE_TEXT_AFTER",
    ),
    "binary_as_bytes": runtime_probe_status(
        binary_check_rc,
        binary_run_rc,
        binary_out_path,
        "RUNTIME_PROBE_BINARY_START",
        "RUNTIME_PROBE_BINARY_AFTER",
    ),
    "dynamic_slice": runtime_probe_status(
        slice_check_rc,
        slice_run_rc,
        slice_out_path,
        "RUNTIME_PROBE_SLICE_START",
        "RUNTIME_PROBE_SLICE_AFTER",
    ),
}

runtime_pass = 0
runtime_fail = 0
runtime_not_run = 0
for row in runtime_regressions.values():
    if row["status"] == "pass":
        runtime_pass += 1
    elif row["status"] == "not_run":
        runtime_not_run += 1
    else:
        runtime_fail += 1

if runtime_fail > 0:
    runtime_status = "fail"
elif runtime_not_run > 0 and runtime_pass == 0:
    runtime_status = "not_run"
else:
    runtime_status = "pass"

runtime_regression_summary = {
    "status": runtime_status,
    "pass": runtime_pass,
    "fail": runtime_fail,
    "not_run": runtime_not_run,
    "total": len(runtime_regressions),
}

fmri_out = fmri_out_path.read_text(encoding="utf-8", errors="replace") if fmri_out_path.exists() else ""
pbpk_out = pbpk_out_path.read_text(encoding="utf-8", errors="replace") if pbpk_out_path.exists() else ""
all_metrics = parse_metrics(fmri_out + "\n" + pbpk_out)

fmri_expected = ((golden_obj.get("fmri") or {}).get("metrics") or {})
pbpk_expected = ((golden_obj.get("darwin_pbpk") or {}).get("metrics") or {})

def lane_status(lane: str, rc: int, marker: str, test_path: Path, observed: dict, expected: dict):
    mismatches = []
    status = "pass"

    if rc == 125:
        status = "not_run"
    elif rc != 0:
        status = "fail"
        mismatches.append({"metric": "_execution", "reason": f"exit_code={rc}"})

    marker_found = marker in (fmri_out if lane == "fmri" else pbpk_out)
    if not marker_found:
        if status != "not_run":
            status = "fail"
        mismatches.append({"metric": "_marker", "reason": f"missing marker {marker}"})

    if not isinstance(expected, dict) or not expected:
        if status != "not_run":
            status = "fail"
        mismatches.append({"metric": "_golden", "reason": "missing expected metrics in golden file"})
    else:
        for key, exp_val in expected.items():
            if key not in observed:
                if status != "not_run":
                    status = "fail"
                mismatches.append({"metric": key, "reason": "missing_metric", "expected": exp_val})
                continue
            obs_val = observed[key]
            delta = abs(obs_val - float(exp_val))
            tol = abs_tol + rel_tol * abs(float(exp_val))
            if not math.isfinite(obs_val):
                if status != "not_run":
                    status = "fail"
                mismatches.append({"metric": key, "reason": "non_finite", "observed": obs_val})
                continue
            if delta > tol:
                if status != "not_run":
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
        "test_path": test_path.relative_to(root).as_posix() if test_path.exists() else str(test_path),
        "exit_code": rc,
        "marker_found": marker_found,
        "metrics": observed,
        "mismatches": mismatches,
    }

fmri_lane = lane_status("fmri", fmri_rc, "SCIENCE_FMRI_OK", fmri_test, all_metrics.get("fmri", {}), fmri_expected)
pbpk_lane = lane_status("darwin_pbpk", pbpk_rc, "SCIENCE_PBPK_OK", pbpk_test, all_metrics.get("darwin_pbpk", {}), pbpk_expected)

if fmri_lane["status"] != "pass":
    failures.append({
        "lane": "fmri",
        "kind": "lane_failed",
        "message": "fmri lane did not satisfy science gate",
    })
if pbpk_lane["status"] != "pass":
    failures.append({
        "lane": "darwin_pbpk",
        "kind": "lane_failed",
        "message": "darwin_pbpk lane did not satisfy science gate",
    })

if runtime_regression_enforcement == "strict" and runtime_fail > 0:
    failures.append({
        "lane": "runtime",
        "kind": "runtime_regression",
        "message": "runtime regression probes failed under strict enforcement",
    })

pass_count = 0
fail_count = 0
not_run_count = 0
for lane in (fmri_lane, pbpk_lane):
    if lane["status"] == "pass":
        pass_count += 1
    elif lane["status"] == "not_run":
        not_run_count += 1
    else:
        fail_count += 1

if pass_count == 2 and not failures and manifest_verified and ignore_violation == 0:
    status_summary = "pass"
elif not_run_count > 0 and pass_count == 0:
    status_summary = "not_run"
else:
    status_summary = "fail"

notes.append(f"manifest_verified={manifest_verified}")
notes.append(f"ignore_policy_violation={bool(ignore_violation)}")
notes.append(f"fmri_exit_code={fmri_rc}")
notes.append(f"darwin_pbpk_exit_code={pbpk_rc}")
notes.append(f"runtime_regression_enforcement={runtime_regression_enforcement}")
notes.append(f"runtime_regression_summary={runtime_status}:{runtime_fail}fail/{runtime_pass}pass/{runtime_not_run}not_run")
notes.append(f"runtime_souc_bin={runtime_souc_bin}")
notes.append(f"runtime_souc_version={runtime_souc_version}")
notes.append(f"runtime_pinned_version_expected={runtime_pinned_version_expected}")

obj = {
    "schema": "sounio.stdlib.science_pipeline_status.v1",
    "generated_at_utc": now,
    "command": "bash scripts/stdlib_science_pipeline_gate.sh",
    "status_summary": status_summary,
    "totals": {
        "pass": pass_count,
        "fail": fail_count,
        "not_run": not_run_count,
        "total": 2,
    },
    "fixture_manifest": manifest_path.relative_to(root).as_posix() if manifest_path.exists() else str(manifest_path),
    "golden_file": golden_path.relative_to(root).as_posix() if golden_path.exists() else str(golden_path),
    "tolerance": {
        "abs": abs_tol,
        "rel": rel_tol,
    },
    "lanes": {
        "fmri": fmri_lane,
        "darwin_pbpk": pbpk_lane,
    },
    "fixture_verification": {
        "verified": manifest_verified,
        "entries": manifest_entries,
    },
    "ignored_policy_violation": {
        "detected": bool(ignore_violation),
        "hits": ignore_hits,
    },
    "runtime_regressions": runtime_regressions,
    "runtime_regression_enforcement": runtime_regression_enforcement,
    "runtime_regression_summary": runtime_regression_summary,
    "runtime_provenance": {
        "souc_bin": runtime_souc_bin,
        "souc_version": runtime_souc_version,
        "pinned_version_expected": runtime_pinned_version_expected,
    },
    "failures": failures,
    "notes": notes,
}

out_path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
PY

if ! validate_status_json; then
  emit_not_run "status_json_validation_failed"
  validate_status_json || true
  echo "error: produced malformed science status artifact" >&2
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

echo "[stdlib-science-gate] status_json=${OUT_JSON#$ROOT_DIR/}"
echo "[stdlib-science-gate] manifest_json=${MANIFEST_JSON#$ROOT_DIR/}"
echo "[stdlib-science-gate] golden_json=${GOLDEN_JSON#$ROOT_DIR/}"
echo "[stdlib-science-gate] status_summary=$status_summary"

if [[ "$status_summary" == "pass" ]]; then
  echo "STDLIB_SCIENCE_PIPELINE_GATE_PASS"
  exit 0
fi

echo "error: STDLIB_SCIENCE_PIPELINE_GATE_${status_summary^^}" >&2
exit 1
