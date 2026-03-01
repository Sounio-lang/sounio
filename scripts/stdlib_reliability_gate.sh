#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_JSON="${STDLIB_RELIABILITY_STATUS_OUT:-$ROOT_DIR/artifacts/stdlib/stdlib_reliability_status.v1.json}"
E2E_JSON="${STDLIB_RELIABILITY_E2E_JSON:-$ROOT_DIR/artifacts/stdlib/stdlib_e2e_result.v1.json}"
INVENTORY_JSON="${STDLIB_RELIABILITY_INVENTORY_JSON:-$ROOT_DIR/artifacts/stdlib/stdlib_inventory.v1.json}"
BASELINE_PASS="${STDLIB_RELIABILITY_BASELINE_PASS:-52}"
BASELINE_FAIL="${STDLIB_RELIABILITY_BASELINE_FAIL:-13}"
BASELINE_SKIP="${STDLIB_RELIABILITY_BASELINE_SKIP:-5}"
BASELINE_TOTAL="${STDLIB_RELIABILITY_BASELINE_TOTAL:-70}"

usage() {
  cat <<'USAGE'
Usage: bash scripts/stdlib_reliability_gate.sh [--out-json PATH] [--e2e-json PATH] [--inventory-json PATH]

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
PY
}

echo "[stdlib-reliability-gate] collecting inventory snapshot"
set +e
bash "$ROOT_DIR/scripts/scan_stdlib.sh" --json-out "$INVENTORY_JSON" --quiet
scan_rc=$?
set -e
if [[ $scan_rc -ne 0 || ! -s "$INVENTORY_JSON" ]]; then
  emit_not_run "inventory_scan_failed"
  validate_status_json || true
  echo "error: inventory scan failed (rc=$scan_rc)" >&2
  exit 1
fi

echo "[stdlib-reliability-gate] running stdlib e2e suite"
set +e
bash "$ROOT_DIR/scripts/run_stdlib_e2e.sh" --json-out "$E2E_JSON"
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
  "$BASELINE_PASS" "$BASELINE_FAIL" "$BASELINE_SKIP" "$BASELINE_TOTAL" <<'PY'
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
echo "[stdlib-reliability-gate] status_summary=$status_summary"

if [[ "$status_summary" == "pass" ]]; then
  echo "STDLIB_RELIABILITY_GATE_PASS"
  exit 0
fi

echo "error: STDLIB_RELIABILITY_GATE_${status_summary^^}" >&2
exit 1
