#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DOC="${SOUNIO_MADAROS_BLOCKER_DOC:-docs/audit/MADAROS_PRODUCTION_READINESS_PLAN_2026-06-21.md}"

if [[ ! -f "$DOC" ]]; then
  echo "error: missing Madaros blocker doc: $DOC" >&2
  exit 1
fi

python3 - "$DOC" <<'PY'
import re
import sys

doc_path = sys.argv[1]
with open(doc_path, "r", encoding="utf-8") as fh:
    text = fh.read()

required_ids = [
    "BLK-20260621-codex-source-elf-normal-bss",
    "BLK-20260621-codex-madaros-build-segfault",
]

required_fields = [
    "Blocker-ID",
    "Status",
    "Severity",
    "Class",
    "Owner",
    "Lane",
    "Worktree",
    "Branch",
    "Files-Owned",
    "Do-Not-Touch",
    "Repro",
    "Observed",
    "Expected",
    "Acceptance-Gate",
    "Evidence-Level",
    "Evidence",
    "Fallback-Path",
    "Legacy-Kept",
    "LLM-Offload",
    "Next-Action",
]

optional_but_expected = ["Files-Read-Only"]
allowed_status = {
    "proposed",
    "reproduced",
    "classified",
    "owned",
    "fixing",
    "review-ready",
    "merge-ready",
    "closed",
    "waived",
}
allowed_severity = {"B0", "B1", "B2", "B3", "B4"}
allowed_evidence = {"E0", "E1", "E2", "E3", "E4"}

blocks = {}
for match in re.finditer(r"```text\n(.*?)\n```", text, re.S):
    block = match.group(1)
    id_match = re.search(r"^Blocker-ID:\s*(\S+)\s*$", block, re.M)
    if id_match:
        blocks[id_match.group(1)] = block

errors = []
for blocker_id in required_ids:
    block = blocks.get(blocker_id)
    if block is None:
        errors.append(f"{blocker_id}: missing fenced blocker record")
        continue

    values = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()

    for field in required_fields:
        if not values.get(field):
            errors.append(f"{blocker_id}: missing or empty field {field}")

    for field in optional_but_expected:
        if field not in values:
            errors.append(f"{blocker_id}: missing expected field {field}")

    if values.get("Status") and values["Status"] not in allowed_status:
        errors.append(f"{blocker_id}: invalid Status {values['Status']}")
    if values.get("Severity") and values["Severity"] not in allowed_severity:
        errors.append(f"{blocker_id}: invalid Severity {values['Severity']}")
    if values.get("Evidence-Level") and values["Evidence-Level"] not in allowed_evidence:
        errors.append(f"{blocker_id}: invalid Evidence-Level {values['Evidence-Level']}")
    if values.get("Repro", "").lower() in {"none", "n/a"}:
        errors.append(f"{blocker_id}: Repro must be an executable command or explicit missing resource")
    if "https://github.com/Sounio-lang/sounio/issues/356" not in values.get("Evidence", ""):
        errors.append(f"{blocker_id}: Evidence must cite canonical issue #356 evidence")

if errors:
    print("MADAROS_BLOCKER_CONTRACT_FAIL", file=sys.stderr)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    sys.exit(1)

print(f"MADAROS_BLOCKER_CONTRACT_PASS doc={doc_path} blockers={len(required_ids)}")
PY
