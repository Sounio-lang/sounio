#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLASSIFIER="$ROOT_DIR/scripts/ci/classify_ci_impact.sh"
DECISION="$ROOT_DIR/scripts/ci/evaluate_ci_decision.py"

expect() {
  local output="$1" key="$2" value="$3"
  grep -Fxq "$key=$value" <<<"$output" || {
    echo "impact-ci-selftest: expected $key=$value" >&2
    echo "$output" >&2
    exit 1
  }
}

classify_pr() {
  CI_EVENT_NAME=pull_request "$CLASSIFIER" "$@"
}

docs="$(classify_pr docs/internal/concepts/README.md)"
expect "$docs" docs true
expect "$docs" compiler false
expect "$docs" lean false

lean="$(classify_pr formal/lean4/SounioGradedModal.lean)"
expect "$lean" lean true
expect "$lean" math true
expect "$lean" compiler false

compiler="$(classify_pr self-hosted/compiler/main.sio)"
expect "$compiler" compiler true
expect "$compiler" sio true

unknown="$(classify_pr newly-introduced/build.graph)"
expect "$unknown" full true
expect "$unknown" compiler true
expect "$unknown" lean true

root_build="$(classify_pr Makefile)"
expect "$root_build" full true

workflow="$(classify_pr .github/workflows/ci.yml)"
for key in docs website compiler runtime stdlib tests lean math ontology clinical sio full; do
  expect "$workflow" "$key" true
done

non_pr="$(CI_EVENT_NAME=push $CLASSIFIER)"
expect "$non_pr" full true
expect "$non_pr" compiler true

# A failed `git diff` used to be invisible through process substitution:
# paths=(), every output false, exit 0 -- a run downstream reads identically
# to "no jobs needed" (silent skip of the whole matrix). Same for an empty
# PR diff. Both must now refuse. Regression guards for the diff-status
# capture in classify_ci_impact.sh.
if (cd "$ROOT_DIR" && CI_EVENT_NAME=pull_request \
      CI_BASE_SHA=0000000000000000000000000000000000000000 \
      CI_HEAD_SHA=HEAD "$CLASSIFIER" >/dev/null 2>&1); then
  echo "impact-ci-selftest: classifier accepted a failing git diff (all-false matrix, exit 0)" >&2
  exit 1
fi
if (cd "$ROOT_DIR" && CI_EVENT_NAME=pull_request \
      CI_BASE_SHA=HEAD CI_HEAD_SHA=HEAD "$CLASSIFIER" >/dev/null 2>&1); then
  echo "impact-ci-selftest: classifier accepted an empty PR diff (every job would silently skip)" >&2
  exit 1
fi
# Positive control: the guards must not over-fire -- a real, non-empty diff
# still classifies through the guarded path.
root_commit="$(cd "$ROOT_DIR" && git rev-list --max-parents=0 HEAD | tail -1)"
live="$(cd "$ROOT_DIR" && CI_EVENT_NAME=pull_request \
      CI_BASE_SHA="$root_commit" CI_HEAD_SHA=HEAD "$CLASSIFIER")"
expect "$live" full true

good_needs='{"impact":{"outputs":{"compiler":"false","runtime":"false","stdlib":"false","tests":"false","sio":"false","lean":"false","website":"false","full":"false"}},"contracts":{"result":"success"},"native-selfhost-linux-x86_64":{"result":"skipped"},"source-bootstrap-selfhost-linux-x86_64":{"result":"skipped"},"madaros-current-source-deref-f64":{"result":"skipped"},"native-selfhost-macos-arm64":{"result":"skipped"},"full-test-suite":{"result":"skipped"},"madaros-witness-gate":{"result":"skipped"},"sounio-lint":{"result":"skipped"},"lean-proofs":{"result":"skipped"},"website":{"result":"skipped"}}'
NEEDS_JSON="$good_needs" python3 "$DECISION" | grep -Fq CI_DECISION_PASS

bad_needs="${good_needs/\"contracts\":{\"result\":\"success\"}/\"contracts\":{\"result\":\"failure\"}}"
if NEEDS_JSON="$bad_needs" python3 "$DECISION" >/dev/null 2>&1; then
  echo "impact-ci-selftest: decision accepted failed contracts" >&2
  exit 1
fi

compiler_needs='{"impact":{"outputs":{"compiler":"true","runtime":"false","stdlib":"false","tests":"false","sio":"true","lean":"false","website":"false","full":"false"}},"contracts":{"result":"success"},"native-selfhost-linux-x86_64":{"result":"success"},"source-bootstrap-selfhost-linux-x86_64":{"result":"success"},"madaros-current-source-deref-f64":{"result":"failure"},"native-selfhost-macos-arm64":{"result":"success"},"full-test-suite":{"result":"success"},"madaros-witness-gate":{"result":"success"},"sounio-lint":{"result":"success"},"lean-proofs":{"result":"skipped"},"website":{"result":"skipped"}}'
if NEEDS_JSON="$compiler_needs" python3 "$DECISION" >/dev/null 2>&1; then
  echo "impact-ci-selftest: decision accepted failed current-source Madaros gate" >&2
  exit 1
fi
compiler_green_needs="${compiler_needs/\"failure\"/\"success\"}"
NEEDS_JSON="$compiler_green_needs" python3 "$DECISION" | grep -Fq CI_DECISION_PASS
witness_failed_needs="${compiler_green_needs/\"madaros-witness-gate\":{\"result\":\"success\"}/\"madaros-witness-gate\":{\"result\":\"failure\"}}"
if NEEDS_JSON="$witness_failed_needs" python3 "$DECISION" >/dev/null 2>&1; then
  echo "impact-ci-selftest: decision accepted failed selected Madaros witness gate" >&2
  exit 1
fi

python3 - "$ROOT_DIR" <<'PY'
import ast
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
workflow = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
decision = (root / "scripts/ci/evaluate_ci_decision.py").read_text(encoding="utf-8")

lines = workflow.splitlines()
workflow_needs = set()
in_ci_decision = False
in_needs = False
for line in lines:
    if line == "  ci-decision:":
        in_ci_decision = True
        continue
    if in_ci_decision and line.startswith("  ") and not line.startswith("    ") and line != "  ci-decision:":
        break
    if not in_ci_decision:
        continue
    if line == "    needs:":
        in_needs = True
        continue
    if in_needs:
        if line.startswith("      - "):
            workflow_needs.add(line.split("-", 1)[1].strip())
            continue
        break
if not workflow_needs:
    raise SystemExit("impact-ci-selftest: could not parse ci-decision needs")
workflow_needs.discard("impact")

module = ast.parse(decision)
required_keys = None
for node in ast.walk(module):
    if isinstance(node, ast.Assign):
        if any(isinstance(target, ast.Name) and target.id == "required" for target in node.targets):
            if isinstance(node.value, ast.Dict):
                required_keys = {
                    key.value
                    for key in node.value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
                break
if required_keys is None:
    raise SystemExit("impact-ci-selftest: could not parse evaluator required map")

if workflow_needs != required_keys:
    missing = sorted(workflow_needs - required_keys)
    extra = sorted(required_keys - workflow_needs)
    raise SystemExit(
        "impact-ci-selftest: ci-decision needs/evaluator mismatch "
        f"missing_in_evaluator={missing} extra_in_evaluator={extra}"
    )
PY

stdlib_needs='{"impact":{"outputs":{"compiler":"false","runtime":"false","stdlib":"true","tests":"false","sio":"true","lean":"false","website":"false","full":"false"}},"contracts":{"result":"success"},"native-selfhost-linux-x86_64":{"result":"skipped"},"source-bootstrap-selfhost-linux-x86_64":{"result":"skipped"},"madaros-current-source-deref-f64":{"result":"skipped"},"native-selfhost-macos-arm64":{"result":"skipped"},"full-test-suite":{"result":"skipped"},"madaros-witness-gate":{"result":"success"},"sounio-lint":{"result":"success"},"lean-proofs":{"result":"skipped"},"website":{"result":"skipped"}}'
if NEEDS_JSON="$stdlib_needs" python3 "$DECISION" >/dev/null 2>&1; then
  echo "impact-ci-selftest: decision accepted stdlib suite without native compiler/full suite" >&2
  exit 1
fi

echo 'IMPACT_CI_SELFTEST_PASS'
