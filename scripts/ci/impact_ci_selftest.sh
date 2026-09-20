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
expect "$docs" chemistry false

# Every path family previously covered by the standalone PR trigger remains
# selected, including nested files and the bootstrap receipt suffix.
for path in examples/chemistry/nested/probe.sio stdlib/chemistry/model.sio benchmarks/chemistry/golden/probe.lean_single.txt scripts/ci/chemistry_probe_golden_gate.sh bin/souc bin/souc-lean-single-x86_64.SeedReceipt.json; do
  expect "$(classify_pr "$path")" chemistry true
done
expect "$(classify_pr stdlib/collections/list.sio)" chemistry false
expect "$(classify_pr self-hosted/compiler/main.sio)" chemistry false

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
for key in docs website compiler runtime stdlib tests lean math ontology clinical chemistry sio full; do
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

# Decision fixtures include every job and exercise missing/skipped/cancelled
# evidence separately, so JSON parser crashes cannot impersonate rejection.
python3 "$ROOT_DIR/scripts/ci/test_ci_layers.py"
echo 'IMPACT_CI_SELFTEST_PASS'
