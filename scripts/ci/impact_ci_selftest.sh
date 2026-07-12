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

good_needs='{"impact":{"outputs":{"compiler":"false","runtime":"false","stdlib":"false","tests":"false","sio":"false","lean":"false","website":"false","full":"false"}},"contracts":{"result":"success"},"native-selfhost-linux-x86_64":{"result":"skipped"},"source-bootstrap-selfhost-linux-x86_64":{"result":"skipped"},"native-selfhost-macos-arm64":{"result":"skipped"},"full-test-suite":{"result":"skipped"},"sounio-lint":{"result":"skipped"},"lean-proofs":{"result":"skipped"},"website":{"result":"skipped"}}'
NEEDS_JSON="$good_needs" python3 "$DECISION" | grep -Fq CI_DECISION_PASS

bad_needs="${good_needs/\"contracts\":{\"result\":\"success\"}/\"contracts\":{\"result\":\"failure\"}}"
if NEEDS_JSON="$bad_needs" python3 "$DECISION" >/dev/null 2>&1; then
  echo "impact-ci-selftest: decision accepted failed contracts" >&2
  exit 1
fi

stdlib_needs='{"impact":{"outputs":{"compiler":"false","runtime":"false","stdlib":"true","tests":"false","sio":"true","lean":"false","website":"false","full":"false"}},"contracts":{"result":"success"},"native-selfhost-linux-x86_64":{"result":"skipped"},"source-bootstrap-selfhost-linux-x86_64":{"result":"skipped"},"native-selfhost-macos-arm64":{"result":"skipped"},"full-test-suite":{"result":"skipped"},"sounio-lint":{"result":"success"},"lean-proofs":{"result":"skipped"},"website":{"result":"skipped"}}'
if NEEDS_JSON="$stdlib_needs" python3 "$DECISION" >/dev/null 2>&1; then
  echo "impact-ci-selftest: decision accepted stdlib suite without native compiler/full suite" >&2
  exit 1
fi

echo 'IMPACT_CI_SELFTEST_PASS'
