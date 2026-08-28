#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ARTIFACT_ROOT="${SOUNIO_STDLIB_SURFACE_ARTIFACT_ROOT:-/tmp/sounio-stdlib-surface-support-$(date -u +%Y%m%dT%H%M%SZ)}"
INVENTORY_JSON="$ARTIFACT_ROOT/stdlib_inventory.v1.json"
SUMMARY_JSON="$ARTIFACT_ROOT/stdlib_surface_support_status.v1.json"
PACKAGE_LOG="$ARTIFACT_ROOT/package_pbpk_gum_gate.log"

mkdir -p "$ARTIFACT_ROOT"

echo "== Sounio Stdlib Surface Support Gate =="
echo "repo=$ROOT_DIR"
echo "head=$(git rev-parse HEAD 2>/dev/null || true)"
echo "artifacts=$ARTIFACT_ROOT"

run_step() {
  local label="$1"
  shift
  local log="$ARTIFACT_ROOT/$label.log"
  echo "[stdlib-surface] >>> $label"
  if "$@" >"$log" 2>&1; then
    echo "[stdlib-surface] <<< $label PASS log=$log"
  else
    local rc=$?
    echo "[stdlib-surface] !!! $label FAIL rc=$rc log=$log" >&2
    tail -n 80 "$log" >&2 || true
    return "$rc"
  fi
}

run_step inventory bash "$ROOT_DIR/scripts/stdlib/scan_stdlib.sh" \
  --json-out "$INVENTORY_JSON" --quiet

echo "[stdlib-surface] >>> package-pbpk-gum"
if bash "$ROOT_DIR/scripts/ci/package_pbpk_gum_gate.sh" >"$PACKAGE_LOG" 2>&1; then
  echo "[stdlib-surface] <<< package-pbpk-gum PASS log=$PACKAGE_LOG"
else
  rc=$?
  echo "[stdlib-surface] !!! package-pbpk-gum FAIL rc=$rc log=$PACKAGE_LOG" >&2
  tail -n 100 "$PACKAGE_LOG" >&2 || true
  exit "$rc"
fi

python3 - "$ROOT_DIR" "$INVENTORY_JSON" "$PACKAGE_LOG" "$SUMMARY_JSON" <<'PY'
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
inventory_path = Path(sys.argv[2]).resolve()
package_log = Path(sys.argv[3]).resolve()
summary_path = Path(sys.argv[4]).resolve()

registry_path = root / "docs/serious-language/public-claim-registry.v1.tsv"
spec_matrix_path = root / "docs/serious-language/spec-evidence-matrix.v1.tsv"
known_limitations_path = root / "docs/compiler/KNOWN_LIMITATIONS.md"

gate_ref = "scripts/ci/sounio_stdlib_surface_support_gate.sh"

failures: list[dict[str, str]] = []

def add_failure(kind: str, message: str) -> None:
    failures.append({"kind": kind, "message": message})

try:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
except Exception as exc:  # pragma: no cover - gate diagnostics
    inventory = {}
    add_failure("inventory", f"inventory JSON unreadable: {exc}")

counts = inventory.get("counts", {})
sio_files = int(counts.get("sio_files", 0) or 0)
active_entrypoints = int(counts.get("active_module_entrypoints", 0) or 0)
disabled_files = int(counts.get("disabled_files", 0) or 0)
stub_mod_files = int(counts.get("stub_mod_files", 0) or 0)

if sio_files < 1000:
    add_failure("inventory", f"expected at least 1000 stdlib .sio files, got {sio_files}")
if active_entrypoints < 150:
    add_failure("inventory", f"expected at least 150 active module entrypoints, got {active_entrypoints}")
if disabled_files != 0:
    add_failure("inventory", f"expected disabled_files=0 in active inventory, got {disabled_files}")
if stub_mod_files != 0:
    add_failure("inventory", f"expected stub_mod_files=0 in active inventory, got {stub_mod_files}")

package_text = package_log.read_text(encoding="utf-8", errors="replace") if package_log.exists() else ""
required_markers = [
    "[package-import-science] PASS: multi-package import contract is gated",
    "[package-pbpk-gum] run canonical observed PETAB baseline",
    "PACKAGE_PBPK_GUM_OK",
    "[package-pbpk-gum] PASS",
]
missing_markers = [marker for marker in required_markers if marker not in package_text]
if missing_markers:
    add_failure("package_pbpk_gum", "missing package gate markers: " + ",".join(missing_markers))

registry_rows: dict[str, dict[str, str]] = {}
with registry_path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    for row in reader:
        registry_rows[row["claim_id"]] = {key: (value or "").strip() for key, value in row.items()}

stdlib_row = registry_rows.get("stdlib.surface")
if not stdlib_row:
    add_failure("claim_registry", "missing stdlib.surface row")
else:
    if stdlib_row.get("claim_level") != "validated_research":
        add_failure("claim_registry", f"stdlib.surface claim_level={stdlib_row.get('claim_level')}")
    if stdlib_row.get("closure_status") != "closed":
        add_failure("claim_registry", f"stdlib.surface closure_status={stdlib_row.get('closure_status')}")
    if stdlib_row.get("evidence_ref") != gate_ref:
        add_failure("claim_registry", f"stdlib.surface evidence_ref={stdlib_row.get('evidence_ref')}")
    wording = stdlib_row.get("public_wording", "")
    for token in ["bounded stdlib support", "not broad", "not hyper/science pipeline"]:
        if token not in wording:
            add_failure("claim_registry", f"stdlib.surface wording missing token: {token}")

spec_rows: dict[str, dict[str, str]] = {}
with spec_matrix_path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    for row in reader:
        spec_rows[row["spec_id"]] = {key: (value or "").strip() for key, value in row.items()}

spec_row = spec_rows.get("stdlib.surface")
if not spec_row:
    add_failure("spec_matrix", "missing stdlib.surface spec row")
else:
    if spec_row.get("status") != "partially_executable":
        add_failure("spec_matrix", f"stdlib.surface status={spec_row.get('status')}")
    if spec_row.get("evidence_ref") != gate_ref:
        add_failure("spec_matrix", f"stdlib.surface evidence_ref={spec_row.get('evidence_ref')}")

limitations = known_limitations_path.read_text(encoding="utf-8", errors="replace")
for token in [
    "stdlib.surface = validated_research",
    "1252 `.sio` files",
    "155 active module entrypoints",
    "NOT PROVED",
]:
    if token not in limitations:
        add_failure("known_limitations", f"missing token: {token}")

not_proved = [
    {
        "surface": "broad_stdlib_callability",
        "evidence": "scripts/ci/stdlib_evolution_gate.sh currently remains outside this support contract",
    },
    {
        "surface": "hypercomplex_native_lanes",
        "evidence": "scripts/stdlib/stdlib_hyper_execution_gate.sh remains a separate required lane",
    },
    {
        "surface": "fmri_and_darwin_pbpk_science_pipeline",
        "evidence": "scripts/ci/stdlib_science_pipeline_gate.sh remains a separate required lane",
    },
    {
        "surface": "external_runtime_dependencies",
        "evidence": "CUDA, libz, libzstd, real network/async/concurrency load, crypto security, and clinical/regulatory validity are not proved here",
    },
]

summary = {
    "schema": "sounio.stdlib.surface_support_status.v1",
    "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    "status_summary": "pass" if not failures else "fail",
    "claim": "stdlib.surface=validated_research",
    "claim_shape": "bounded stdlib support contract; not broad stdlib callability",
    "artifacts": {
        "inventory_json": str(inventory_path),
        "package_pbpk_gum_log": str(package_log),
    },
    "proved": {
        "inventory": {
            "sio_files": sio_files,
            "active_module_entrypoints": active_entrypoints,
            "disabled_files": disabled_files,
            "stub_mod_files": stub_mod_files,
        },
        "package_backed_slices": [
            "epistemic-core",
            "sounio-units",
            "sounio-formats",
            "sounio-io-primitives",
            "package-backed PBPK/GUM workflow",
            "canonical observed PETAB baseline",
        ],
        "claim_docs": [
            str(registry_path.relative_to(root)),
            str(spec_matrix_path.relative_to(root)),
            str(known_limitations_path.relative_to(root)),
        ],
    },
    "not_proved": not_proved,
    "failures": failures,
}

summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"status_summary={summary['status_summary']}")
print(f"sio_files={sio_files}")
print(f"active_module_entrypoints={active_entrypoints}")
print(f"summary_json={summary_path}")
if failures:
    for failure in failures:
        print(f"failure {failure['kind']}: {failure['message']}")
    raise SystemExit(1)
PY

echo "[stdlib-surface] PASS: bounded stdlib support contract is checked"
