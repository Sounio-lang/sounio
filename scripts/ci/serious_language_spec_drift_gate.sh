#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MATRIX="${SOUNIO_SPEC_EVIDENCE_MATRIX:-$ROOT_DIR/docs/serious-language/spec-evidence-matrix.v1.tsv}"
CONFORMANCE_MANIFEST="${SOUNIO_SERIOUS_CONFORMANCE_MANIFEST:-$ROOT_DIR/tests/conformance/manifest.v1.tsv}"
timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
ARTIFACT_ROOT="${SOUNIO_SPEC_DRIFT_ARTIFACT_ROOT:-/tmp/sounio-serious-spec-drift-$timestamp}"
CONFORMANCE_ARTIFACT_ROOT="$ARTIFACT_ROOT/conformance"
CONFORMANCE_SUMMARY="${SOUNIO_SPEC_DRIFT_CONFORMANCE_SUMMARY:-$CONFORMANCE_ARTIFACT_ROOT/summary.v1.tsv}"
SUMMARY_JSON="$ARTIFACT_ROOT/summary.v1.json"
RESULTS="$ARTIFACT_ROOT/RESULTS.md"

mkdir -p "$ARTIFACT_ROOT"

if [[ ! -f "$MATRIX" ]]; then
  echo "Spec evidence matrix not found: $MATRIX" >&2
  exit 2
fi

if [[ ! -f "$CONFORMANCE_MANIFEST" ]]; then
  echo "Conformance manifest not found: $CONFORMANCE_MANIFEST" >&2
  exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required for serious_language_spec_drift_gate.sh" >&2
  exit 2
fi

if [[ -z "${SOUNIO_SPEC_DRIFT_CONFORMANCE_SUMMARY:-}" ]]; then
  set +e
  env \
    SOUNIO_SERIOUS_CONFORMANCE_ARTIFACT_ROOT="$CONFORMANCE_ARTIFACT_ROOT" \
    SOUNIO_SERIOUS_CONFORMANCE_MANIFEST="$CONFORMANCE_MANIFEST" \
    bash scripts/ci/serious_language_conformance_gate.sh \
    >"$ARTIFACT_ROOT/conformance.log" 2>&1
  conformance_rc=$?
  set -e
  if [[ "$conformance_rc" -ne 0 ]]; then
    echo "Conformance gate failed while preparing spec drift evidence: rc=$conformance_rc" >&2
    echo "Conformance log: $ARTIFACT_ROOT/conformance.log" >&2
    exit 1
  fi
fi

if [[ ! -f "$CONFORMANCE_SUMMARY" ]]; then
  echo "Conformance summary not found: $CONFORMANCE_SUMMARY" >&2
  exit 2
fi

python3 - "$ROOT_DIR" "$MATRIX" "$CONFORMANCE_MANIFEST" "$CONFORMANCE_SUMMARY" "$SUMMARY_JSON" "$RESULTS" "$timestamp" <<'PY'
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
matrix_path = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
conformance_summary_path = Path(sys.argv[4])
summary_json = Path(sys.argv[5])
results_md = Path(sys.argv[6])
timestamp = sys.argv[7]

required_header = [
    "spec_id",
    "spec_anchor",
    "status",
    "evidence_kind",
    "evidence_ref",
    "claim_id",
    "notes",
]
evidence_required = {"executable", "partially_executable", "validated_research"}
valid_statuses = evidence_required | {"prototype", "specified_only", "stale_conflicting"}


def rel_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


with matrix_path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    fieldnames = [name.strip() for name in (reader.fieldnames or [])]
    if fieldnames != required_header:
        print("Spec evidence matrix header mismatch.", file=sys.stderr)
        print(f"expected: {required_header}", file=sys.stderr)
        print(f"actual:   {reader.fieldnames}", file=sys.stderr)
        raise SystemExit(2)
    rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]

with manifest_path.open(newline="", encoding="utf-8") as handle:
    manifest_reader = csv.DictReader(handle, delimiter="\t")
    required_manifest_fields = {"id", "claim_id"}
    manifest_fields = {name.strip() for name in (manifest_reader.fieldnames or [])}
    if not required_manifest_fields.issubset(manifest_fields):
        print("Conformance manifest header mismatch.", file=sys.stderr)
        print(f"required fields: {sorted(required_manifest_fields)}", file=sys.stderr)
        print(f"actual:          {manifest_reader.fieldnames}", file=sys.stderr)
        raise SystemExit(2)
    manifest = [{key: (value or "").strip() for key, value in row.items()} for row in manifest_reader]

with conformance_summary_path.open(newline="", encoding="utf-8") as handle:
    summary_reader = csv.DictReader(handle, delimiter="\t")
    required_summary_fields = {"id", "claim_id", "status"}
    summary_fields = {name.strip() for name in (summary_reader.fieldnames or [])}
    if not required_summary_fields.issubset(summary_fields):
        print("Conformance summary header mismatch.", file=sys.stderr)
        print(f"required fields: {sorted(required_summary_fields)}", file=sys.stderr)
        print(f"actual:          {summary_reader.fieldnames}", file=sys.stderr)
        raise SystemExit(2)
    conformance_rows = [{key: (value or "").strip() for key, value in row.items()} for row in summary_reader]

case_ids = {row["id"] for row in manifest}
case_status = {row["id"]: row["status"] for row in conformance_rows}
claim_ids = {row["claim_id"] for row in manifest}
claim_statuses: dict[str, list[str]] = {}
for row in conformance_rows:
    claim_statuses.setdefault(row["claim_id"], []).append(row["status"])

failures: list[dict[str, str]] = []


def repo_path_exists(rel: str) -> bool:
    clean = rel.split("#", 1)[0]
    if not clean:
        return False
    return (root / clean).exists()


for row in rows:
    spec_id = row["spec_id"]
    status = row["status"]
    kind = row["evidence_kind"]
    ref = row["evidence_ref"]
    claim_id = row["claim_id"]

    if status not in valid_statuses:
        failures.append({"spec_id": spec_id, "reason": f"unknown status: {status}"})
        continue

    if status in evidence_required and (not ref or ref == "-" or kind == "-"):
        failures.append({"spec_id": spec_id, "reason": "executable status without evidence"})
        continue

    if claim_id and claim_id != "-" and status in evidence_required and kind in {"conformance_case", "conformance_claim"} and claim_id not in claim_ids:
        failures.append({"spec_id": spec_id, "reason": f"claim_id not in conformance manifest: {claim_id}"})

    if not ref or ref == "-":
        continue

    if kind == "conformance_case":
        if ref not in case_ids:
            failures.append({"spec_id": spec_id, "reason": f"missing conformance case: {ref}"})
        elif case_status.get(ref) != "pass":
            failures.append({"spec_id": spec_id, "reason": f"conformance case did not pass: {ref}"})
    elif kind == "conformance_claim":
        if ref not in claim_ids:
            failures.append({"spec_id": spec_id, "reason": f"missing conformance claim: {ref}"})
        elif not claim_statuses.get(ref):
            failures.append({"spec_id": spec_id, "reason": f"conformance claim has no executed cases: {ref}"})
        elif any(status != "pass" for status in claim_statuses[ref]):
            failures.append({"spec_id": spec_id, "reason": f"one or more conformance cases failed for claim: {ref}"})
    elif kind == "test":
        if not repo_path_exists(ref):
            failures.append({"spec_id": spec_id, "reason": f"missing test evidence path: {ref}"})
    elif kind == "gate":
        path = root / ref
        if not repo_path_exists(ref):
            failures.append({"spec_id": spec_id, "reason": f"missing gate evidence path: {ref}"})
        elif path.suffix == ".sh":
            if not os.access(path, os.X_OK):
                failures.append({"spec_id": spec_id, "reason": f"gate is not executable: {ref}"})
            syntax = subprocess.run(["bash", "-n", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if syntax.returncode != 0:
                failures.append({"spec_id": spec_id, "reason": f"gate shell syntax failed: {ref}: {syntax.stderr.strip()}"})
    elif kind == "doc":
        if not repo_path_exists(ref):
            failures.append({"spec_id": spec_id, "reason": f"missing doc evidence path: {ref}"})
    else:
        failures.append({"spec_id": spec_id, "reason": f"unknown evidence kind: {kind}"})

payload = {
    "schema": "sounio.serious_language_spec_drift.v1",
    "generated_at": timestamp,
    "matrix": rel_to_root(matrix_path),
    "conformance_manifest": rel_to_root(manifest_path),
    "conformance_summary": rel_to_root(conformance_summary_path),
    "total_rows": len(rows),
    "failures": failures,
    "status": "pass" if not failures else "fail",
}
summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = [
    "# Sounio Serious-Language Spec Drift Gate",
    "",
    f"Generated at: {timestamp}",
    "",
    "| Field | Value |",
    "|---|---|",
    f"| matrix | `{rel_to_root(matrix_path)}` |",
    f"| conformance_manifest | `{rel_to_root(manifest_path)}` |",
    f"| conformance_summary | `{rel_to_root(conformance_summary_path)}` |",
    f"| rows | `{len(rows)}` |",
    f"| failures | `{len(failures)}` |",
    "",
]

if failures:
    lines.extend(["## Failures", ""])
    for item in failures:
        lines.append(f"- `{item['spec_id']}`: {item['reason']}")
    lines.append("")

lines.extend([
    "## Rule",
    "",
    "Rows marked `executable`, `partially_executable`, or `validated_research` must cite live repo evidence.",
    "Conformance evidence must exist in the manifest and pass in the conformance summary.",
    "Rows marked `prototype`, `specified_only`, or `stale_conflicting` may remain documented without executable conformance coverage, but public claims must keep that lower status.",
    "",
])
results_md.write_text("\n".join(lines), encoding="utf-8")

if failures:
    print(f"Spec drift gate failed with {len(failures)} failure(s). See {results_md}", file=sys.stderr)
    raise SystemExit(1)

print(f"Spec drift gate passed. See {results_md}")
PY
