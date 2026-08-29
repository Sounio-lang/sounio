#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CLAIM_REGISTRY="${SOUNIO_PUBLIC_CLAIM_REGISTRY:-$ROOT_DIR/docs/serious-language/public-claim-registry.v1.tsv}"
DOC_SURFACE="${SOUNIO_DOC_CLAIM_SURFACE:-$ROOT_DIR/docs/serious-language/doc-claim-surface.v1.tsv}"
CLAIM_LINES="${SOUNIO_CLAIM_LINE_ANNOTATIONS:-$ROOT_DIR/docs/serious-language/claim-line-annotations.v1.tsv}"
MATRIX="${SOUNIO_SPEC_EVIDENCE_MATRIX:-$ROOT_DIR/docs/serious-language/spec-evidence-matrix.v1.tsv}"
CONFORMANCE_MANIFEST="${SOUNIO_SERIOUS_CONFORMANCE_MANIFEST:-$ROOT_DIR/tests/conformance/manifest.v1.tsv}"
timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
ARTIFACT_ROOT="${SOUNIO_CLAIM_CLOSURE_ARTIFACT_ROOT:-/tmp/sounio-serious-claim-closure-$timestamp}"
SPEC_DRIFT_ROOT="$ARTIFACT_ROOT/spec-drift"
CONFORMANCE_SUMMARY="${SOUNIO_CLAIM_CLOSURE_CONFORMANCE_SUMMARY:-$SPEC_DRIFT_ROOT/conformance/summary.v1.tsv}"
SUMMARY_JSON="$ARTIFACT_ROOT/summary.v1.json"
RESULTS="$ARTIFACT_ROOT/RESULTS.md"

mkdir -p "$ARTIFACT_ROOT"

for path in "$CLAIM_REGISTRY" "$DOC_SURFACE" "$CLAIM_LINES" "$MATRIX" "$CONFORMANCE_MANIFEST"; do
  if [[ ! -f "$path" ]]; then
    echo "Required claim-closure input not found: $path" >&2
    exit 2
  fi
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required for serious_language_claim_closure_gate.sh" >&2
  exit 2
fi

if [[ -z "${SOUNIO_CLAIM_CLOSURE_CONFORMANCE_SUMMARY:-}" ]]; then
  set +e
  env \
    SOUNIO_SPEC_DRIFT_ARTIFACT_ROOT="$SPEC_DRIFT_ROOT" \
    SOUNIO_SPEC_EVIDENCE_MATRIX="$MATRIX" \
    SOUNIO_SERIOUS_CONFORMANCE_MANIFEST="$CONFORMANCE_MANIFEST" \
    bash scripts/ci/serious_language_spec_drift_gate.sh \
    >"$ARTIFACT_ROOT/spec-drift.log" 2>&1
  spec_drift_rc=$?
  set -e
  if [[ "$spec_drift_rc" -ne 0 ]]; then
    echo "Spec drift gate failed while preparing claim-closure evidence: rc=$spec_drift_rc" >&2
    echo "Spec drift log: $ARTIFACT_ROOT/spec-drift.log" >&2
    exit 1
  fi
fi

if [[ ! -f "$CONFORMANCE_SUMMARY" ]]; then
  echo "Conformance summary not found for claim closure: $CONFORMANCE_SUMMARY" >&2
  exit 2
fi

python3 - "$ROOT_DIR" "$CLAIM_REGISTRY" "$DOC_SURFACE" "$CLAIM_LINES" "$MATRIX" "$CONFORMANCE_MANIFEST" "$CONFORMANCE_SUMMARY" "$SUMMARY_JSON" "$RESULTS" "$timestamp" <<'PY'
from __future__ import annotations

import csv
import fnmatch
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
claim_registry_path = Path(sys.argv[2])
doc_surface_path = Path(sys.argv[3])
claim_lines_path = Path(sys.argv[4])
matrix_path = Path(sys.argv[5])
manifest_path = Path(sys.argv[6])
conformance_summary_path = Path(sys.argv[7])
summary_json = Path(sys.argv[8])
results_md = Path(sys.argv[9])
timestamp = sys.argv[10]

claim_header = [
    "claim_id",
    "claim_level",
    "closure_status",
    "evidence_kind",
    "evidence_ref",
    "spec_refs",
    "public_wording",
]
surface_header = ["pattern", "match_kind", "surface_policy", "claim_ids", "notes"]
claim_line_header = ["annotation_id", "doc_path", "line", "claim_id", "anchor_text", "notes"]
matrix_header = ["spec_id", "spec_anchor", "status", "evidence_kind", "evidence_ref", "claim_id", "notes"]
valid_levels = {"stable", "validated_research", "prototype", "scaffold", "stale_conflicting"}
valid_closure = {"closed", "downgraded", "internal_only", "historical"}
valid_surface = {"public", "downgraded", "internal", "historical"}
valid_match = {"exact", "prefix", "glob"}
passing_status = {"pass"}
failures: list[dict[str, str]] = []


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def read_tsv(path: Path, expected: list[str], label: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = [name.strip() for name in (reader.fieldnames or [])]
        if fields != expected:
            failures.append({
                "surface": label,
                "reason": f"header mismatch: expected {expected}, actual {reader.fieldnames}",
            })
            return []
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


claims = read_tsv(claim_registry_path, claim_header, "claim-registry")
surfaces = read_tsv(doc_surface_path, surface_header, "doc-surface")
claim_lines = read_tsv(claim_lines_path, claim_line_header, "claim-line-annotations")
matrix_rows = read_tsv(matrix_path, matrix_header, "spec-matrix")

with manifest_path.open(newline="", encoding="utf-8") as handle:
    manifest_reader = csv.DictReader(handle, delimiter="\t")
    manifest_rows = [{key: (value or "").strip() for key, value in row.items()} for row in manifest_reader]

with conformance_summary_path.open(newline="", encoding="utf-8") as handle:
    summary_reader = csv.DictReader(handle, delimiter="\t")
    conformance_rows = [{key: (value or "").strip() for key, value in row.items()} for row in summary_reader]

claim_by_id: dict[str, dict[str, str]] = {}
for claim in claims:
    claim_id = claim["claim_id"]
    if not claim_id:
        failures.append({"surface": "claim-registry", "reason": "empty claim_id"})
        continue
    if claim_id in claim_by_id:
        failures.append({"surface": claim_id, "reason": "duplicate claim_id"})
        continue
    claim_by_id[claim_id] = claim

    if claim["claim_level"] not in valid_levels:
        failures.append({"surface": claim_id, "reason": f"unknown claim_level: {claim['claim_level']}"})
    if claim["closure_status"] not in valid_closure:
        failures.append({"surface": claim_id, "reason": f"unknown closure_status: {claim['closure_status']}"})
    if claim["claim_level"] in {"stable", "validated_research"} and claim["closure_status"] != "closed":
        failures.append({"surface": claim_id, "reason": "stable or validated_research claim must be closed"})
    if claim["claim_level"] in {"prototype", "scaffold", "stale_conflicting"} and claim["closure_status"] == "closed":
        failures.append({"surface": claim_id, "reason": "downgraded claim level must not be marked closed"})
    if not claim["public_wording"]:
        failures.append({"surface": claim_id, "reason": "missing public_wording guardrail"})

case_ids = {row.get("id", "") for row in manifest_rows}
manifest_claim_ids = {row.get("claim_id", "") for row in manifest_rows if row.get("claim_id", "")}
case_status = {row.get("id", ""): row.get("status", "") for row in conformance_rows}
claim_statuses: dict[str, list[str]] = {}
for row in conformance_rows:
    claim_statuses.setdefault(row.get("claim_id", ""), []).append(row.get("status", ""))


def repo_path_exists(ref: str) -> bool:
    clean = ref.split("#", 1)[0]
    if not clean:
        return False
    return (root / clean).exists()


def split_refs(value: str) -> list[str]:
    if not value or value == "-":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_evidence(owner: str, kind: str, refs: list[str]) -> None:
    if kind in {"-", ""}:
        failures.append({"surface": owner, "reason": "missing evidence_kind"})
        return
    if not refs:
        failures.append({"surface": owner, "reason": "missing evidence_ref"})
        return
    if kind == "conformance_case":
        for ref_id in refs:
            if ref_id not in case_ids:
                failures.append({"surface": owner, "reason": f"missing conformance case: {ref_id}"})
            elif case_status.get(ref_id) not in passing_status:
                failures.append({"surface": owner, "reason": f"conformance case did not pass: {ref_id}"})
    elif kind == "conformance_claim":
        for ref_id in refs:
            if ref_id not in manifest_claim_ids:
                failures.append({"surface": owner, "reason": f"missing conformance claim: {ref_id}"})
            elif not claim_statuses.get(ref_id):
                failures.append({"surface": owner, "reason": f"conformance claim has no executed cases: {ref_id}"})
            elif any(status not in passing_status for status in claim_statuses[ref_id]):
                failures.append({"surface": owner, "reason": f"one or more conformance cases failed for claim: {ref_id}"})
    elif kind in {"test", "doc"}:
        for ref_path in refs:
            if not repo_path_exists(ref_path):
                failures.append({"surface": owner, "reason": f"missing {kind} evidence path: {ref_path}"})
    elif kind == "gate":
        for ref_path in refs:
            path = root / ref_path
            if not repo_path_exists(ref_path):
                failures.append({"surface": owner, "reason": f"missing gate evidence path: {ref_path}"})
            elif path.suffix == ".sh":
                if not os.access(path, os.X_OK):
                    failures.append({"surface": owner, "reason": f"gate is not executable: {ref_path}"})
                syntax = subprocess.run(["bash", "-n", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if syntax.returncode != 0:
                    failures.append({"surface": owner, "reason": f"gate shell syntax failed: {ref_path}: {syntax.stderr.strip()}"})
    else:
        failures.append({"surface": owner, "reason": f"unknown evidence_kind: {kind}"})


for claim in claims:
    validate_evidence(claim["claim_id"], claim["evidence_kind"], split_refs(claim["evidence_ref"]))

matrix_claim_ids = {row["claim_id"] for row in matrix_rows if row.get("claim_id") and row["claim_id"] != "-"}
for claim_id in sorted(matrix_claim_ids | manifest_claim_ids):
    if claim_id not in claim_by_id:
        failures.append({"surface": claim_id, "reason": "claim_id referenced by matrix or conformance manifest is not registered"})

matrix_spec_ids = {row["spec_id"] for row in matrix_rows}
for claim in claims:
    for spec_id in split_refs(claim["spec_refs"]):
        if spec_id not in matrix_spec_ids:
            failures.append({"surface": claim["claim_id"], "reason": f"spec_ref not in matrix: {spec_id}"})


def matches(row: dict[str, str], path: str) -> bool:
    pattern = row["pattern"]
    kind = row["match_kind"]
    if kind == "exact":
        return path == pattern
    if kind == "prefix":
        prefix = pattern[:-3] if pattern.endswith("/**") else pattern
        return path == prefix or path.startswith(prefix.rstrip("/") + "/")
    if kind == "glob":
        return fnmatch.fnmatchcase(path, pattern)
    return False


for surface in surfaces:
    if surface["match_kind"] not in valid_match:
        failures.append({"surface": surface["pattern"], "reason": f"unknown match_kind: {surface['match_kind']}"})
    if surface["surface_policy"] not in valid_surface:
        failures.append({"surface": surface["pattern"], "reason": f"unknown surface_policy: {surface['surface_policy']}"})
    for claim_id in split_refs(surface["claim_ids"]):
        if claim_id not in claim_by_id:
            failures.append({"surface": surface["pattern"], "reason": f"surface references unknown claim_id: {claim_id}"})
        elif surface["surface_policy"] == "public" and claim_by_id[claim_id]["closure_status"] not in {"closed", "downgraded"}:
            failures.append({"surface": surface["pattern"], "reason": f"public surface references non-public claim closure: {claim_id}"})

annotation_ids: set[str] = set()
for annotation in claim_lines:
    annotation_id = annotation["annotation_id"]
    doc_path = annotation["doc_path"]
    claim_id = annotation["claim_id"]
    anchor_text = annotation["anchor_text"]
    line_value = annotation["line"]

    if not annotation_id:
        failures.append({"surface": "claim-line-annotations", "reason": "empty annotation_id"})
        continue
    if annotation_id in annotation_ids:
        failures.append({"surface": annotation_id, "reason": "duplicate annotation_id"})
        continue
    annotation_ids.add(annotation_id)

    if claim_id not in claim_by_id:
        failures.append({"surface": annotation_id, "reason": f"unknown claim_id: {claim_id}"})

    if not doc_path or doc_path.startswith("/") or ".." in Path(doc_path).parts:
        failures.append({"surface": annotation_id, "reason": f"unsafe doc_path: {doc_path}"})
        continue

    doc_file = root / doc_path
    if not doc_file.is_file():
        failures.append({"surface": annotation_id, "reason": f"missing annotated doc path: {doc_path}"})
        continue

    try:
        line_number = int(line_value)
    except ValueError:
        failures.append({"surface": annotation_id, "reason": f"line is not an integer: {line_value}"})
        continue

    if line_number < 1:
        failures.append({"surface": annotation_id, "reason": f"line must be >= 1: {line_value}"})
        continue

    lines_for_doc = doc_file.read_text(encoding="utf-8").splitlines()
    if line_number > len(lines_for_doc):
        failures.append({"surface": annotation_id, "reason": f"line exceeds doc length: {doc_path}:{line_number}"})
        continue

    if not anchor_text or anchor_text == "-":
        failures.append({"surface": annotation_id, "reason": "missing anchor_text"})
    elif anchor_text not in lines_for_doc[line_number - 1]:
        failures.append({"surface": annotation_id, "reason": f"anchor_text not found at {doc_path}:{line_number}"})

    if not any(matches(row, doc_path) for row in surfaces):
        failures.append({"surface": annotation_id, "reason": f"annotated doc has no doc-surface rule: {doc_path}"})

git = subprocess.run(
    ["git", "ls-files", "README.md", "INSTALL.md", "docs"],
    cwd=root,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    check=False,
)
if git.returncode != 0:
    failures.append({"surface": "doc-surface", "reason": f"git ls-files failed: {git.stderr.strip()}"})
    doc_paths: list[str] = []
else:
    doc_paths = [
        line.strip()
        for line in git.stdout.splitlines()
        if line.strip()
        and (line.strip() in {"README.md", "INSTALL.md"} or line.strip().startswith("docs/"))
        and (line.strip().endswith(".md") or line.strip().endswith(".txt"))
    ]

coverage: dict[str, str] = {}
for path in doc_paths:
    matched = [row for row in surfaces if matches(row, path)]
    if not matched:
        failures.append({"surface": path, "reason": "repo doc is not covered by doc-claim-surface manifest"})
        continue
    # Most specific match wins for reporting, but any match is enough for coverage.
    matched.sort(key=lambda row: len(row["pattern"]), reverse=True)
    coverage[path] = matched[0]["pattern"]

payload = {
    "schema": "sounio.serious_language_claim_closure.v1",
    "generated_at": timestamp,
    "claim_registry": rel(claim_registry_path),
    "doc_surface": rel(doc_surface_path),
    "claim_line_annotations": rel(claim_lines_path),
    "spec_matrix": rel(matrix_path),
    "conformance_manifest": rel(manifest_path),
    "conformance_summary": rel(conformance_summary_path),
    "claim_count": len(claims),
    "doc_surface_rule_count": len(surfaces),
    "claim_line_annotation_count": len(claim_lines),
    "repo_doc_count": len(doc_paths),
    "covered_repo_doc_count": len(coverage),
    "failures": failures,
    "status": "pass" if not failures else "fail",
}
summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = [
    "# Sounio Serious-Language Claim Closure Gate",
    "",
    f"Generated at: {timestamp}",
    "",
    "| Field | Value |",
    "|---|---|",
    f"| claim_registry | `{rel(claim_registry_path)}` |",
    f"| doc_surface | `{rel(doc_surface_path)}` |",
    f"| claim_line_annotations | `{rel(claim_lines_path)}` |",
    f"| spec_matrix | `{rel(matrix_path)}` |",
    f"| conformance_manifest | `{rel(manifest_path)}` |",
    f"| conformance_summary | `{rel(conformance_summary_path)}` |",
    f"| claims | `{len(claims)}` |",
    f"| doc_surface_rules | `{len(surfaces)}` |",
    f"| claim_line_annotations | `{len(claim_lines)}` |",
    f"| repo_docs | `{len(doc_paths)}` |",
    f"| covered_repo_docs | `{len(coverage)}` |",
    f"| failures | `{len(failures)}` |",
    "",
]
if failures:
    lines.extend(["## Failures", ""])
    for item in failures:
        lines.append(f"- `{item['surface']}`: {item['reason']}")
    lines.append("")

lines.extend([
    "## Rule",
    "",
    "Every claim ID cited by the conformance manifest or spec/evidence matrix must exist in the public claim registry.",
    "Every stable or validated-research registry claim must have passing evidence.",
    "Every prototype, scaffold, or stale/conflicting claim must be explicitly downgraded rather than silently promoted.",
    "Every governed repo doc under `README.md`, `INSTALL.md`, or `docs/` must be covered by the doc-surface manifest.",
    "Every high-value claim-line annotation must point to a registered claim and an exact doc line containing the recorded anchor text.",
    "",
])
results_md.write_text("\n".join(lines), encoding="utf-8")

if failures:
    print(f"Claim closure gate failed with {len(failures)} failure(s). See {results_md}", file=sys.stderr)
    raise SystemExit(1)

print(f"Claim closure gate passed. See {results_md}")
PY
