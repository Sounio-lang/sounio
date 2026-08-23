#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERIFIER="$ROOT_DIR/scripts/dev/sounio_fleet_trace_verify.py"
MODEL="$ROOT_DIR/formal/tla/SounioFleet.tla"

fail() {
  printf 'sounio-fleet-trace-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

python3 -m py_compile "$VERIFIER"
if rg -q '(^|[[:space:]])(from|import)[[:space:]]+.*sounio_coord_fleetd' "$VERIFIER"; then
  fail 'refinement verifier imports the producer it must independently check'
fi

python3 - "$VERIFIER" "$MODEL" <<'PY'
import ast
import re
import sys

verifier_path, model_path = sys.argv[1:]
tree = ast.parse(open(verifier_path, encoding="utf-8").read())
labels = set()
for node in ast.walk(tree):
    if not isinstance(node, ast.Assign):
        continue
    if not any(isinstance(target, ast.Name) and target.id == "label" for target in node.targets):
        continue
    values = [node.value]
    if isinstance(node.value, ast.IfExp):
        values = [node.value.body, node.value.orelse]
    for value in values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            labels.add(value.value)

model = open(model_path, encoding="utf-8").read()
actions = set(re.findall(r"^([A-Z][A-Za-z0-9_]*)\s*==", model, re.MULTILINE))
required = {
    "IssueStartCapability",
    "StartWithLinearCapability",
    "CreateCheckpoint",
    "VerifyCheckpoint",
    "PrepareHandoff",
    "AnchorVerifiedPrefix",
    "AcceptAnchoredHandoff",
    "Stutter",
}
assert labels == required, (labels, required)
assert labels - {"Stutter"} <= actions, (labels, actions)
assert {"Crash", "Recover", "PersistentStep"} <= actions, actions
PY

echo 'sounio-fleet-trace-selftest: PASS independence=producer-import-absent refinement_actions=7 crash_steps=2 stutter=explicit'
