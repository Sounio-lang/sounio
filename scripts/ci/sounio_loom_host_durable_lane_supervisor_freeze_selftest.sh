#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/host_durable_lane_supervisor.freeze.v1"

fail() {
  printf 'sounio-loom-host-durable-lane-supervisor-freeze-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

manifest_value() {
  local key="$1" count line
  count="$(grep -c "^${key}=" "$MANIFEST" || true)"
  [[ "$count" == 1 ]] || fail "manifest field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$MANIFEST")"
  printf '%s' "${line#*=}"
}

expect_value() {
  local key="$1" expected="$2" actual
  actual="$(manifest_value "$key")"
  [[ "$actual" == "$expected" ]] || fail "$key expected $expected but found $actual"
}

expect_hash() {
  local path="$1" expected="$2" actual
  [[ -f "$ROOT_DIR/$path" && ! -L "$ROOT_DIR/$path" ]] ||
    fail "$path is absent or linked"
  actual="$(sha256sum "$ROOT_DIR/$path" | cut -d ' ' -f 1)"
  [[ "$actual" == "$expected" ]] || fail "$path hash drifted"
}

[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail 'freeze manifest is absent or linked'
expect_value schema loom-host-durable-lane-supervisor-freeze-v1
expect_value stage SEMANTICS_FROZEN
expect_value producing_language Sounio
expect_value language_role SEMANTIC_AUTHORITY
expect_value action 9032
expect_value causal_sabotage PASS
expect_value load_bearing_rule guardian_start_equal
expect_value expected_results_encoded_in_material_layer false
expect_value material_execution false
expect_value host_guardian_attached false
expect_value transport_pod_deleted false
expect_value same_physical_reattach_measured false
expect_value production_activation false
expect_value parity_open false
expect_value claim_ready false

for key in garden source entrypoint build_script selftest; do
  expect_hash "$(manifest_value "${key}_path")" "$(manifest_value "${key}_sha256")"
done
expect_hash "$(manifest_value evidence_path)" "$(manifest_value evidence_sha256)"
expect_hash "$(manifest_value toolchain_wrapper_path)" "$(manifest_value toolchain_wrapper_sha256)"
expect_hash "$(manifest_value toolchain_compiler_path)" "$(manifest_value toolchain_compiler_sha256)"

source_commit="$(manifest_value sounio_executable_commit)"
git -C "$ROOT_DIR" cat-file -e "${source_commit}^{commit}" || fail 'Sounio executable commit is absent'
for key in garden source entrypoint build_script selftest; do
  path="$(manifest_value "${key}_path")"
  at_commit="$(git -C "$ROOT_DIR" show "${source_commit}:$path" | sha256sum | cut -d ' ' -f 1)"
  [[ "$at_commit" == "$(manifest_value "${key}_sha256")" ]] ||
    fail "$key is not bound to the Sounio executable commit"
done

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-host-durable-freeze.XXXXXX")"
trap 'rm -rf "$work"' EXIT
for ordinal in one two; do
  SOUNIO_LOOM_HOST_DURABLE_LANE_OUTPUT="$work/runtime-$ordinal" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_host_durable_lane_supervisor_fixture.sh" >/dev/null
done
cmp "$work/runtime-one" "$work/runtime-two" || fail 'Sounio rebuild is nondeterministic'
[[ "$(sha256sum "$work/runtime-one" | cut -d ' ' -f 1)" == \
   "$(manifest_value executable_sha256)" ]] || fail 'Sounio executable hash drifted'

same='9032 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 1 1 1'
lineage='9032 3 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1'
[[ "$(printf '%s\n' "$same" | "$work/runtime-one")" == \
   "$(manifest_value same_physical_decision)" ]] || fail 'same-physical decision drifted'
[[ "$(printf '%s\n' "$lineage" | "$work/runtime-one")" == \
   "$(manifest_value lineage_decision)" ]] || fail 'lineage decision drifted'

result="$(bash "$ROOT_DIR/scripts/ci/sounio_loom_host_durable_lane_supervisor_fixture_selftest.sh")"
[[ "$result" == "$(manifest_value result)" ]] || fail 'Sounio fixture result drifted'
[[ "$(printf '%s' "$result" | sha256sum | cut -d ' ' -f 1)" == \
   "$(manifest_value result_sha256)" ]] || fail 'Sounio fixture result hash drifted'

printf 'sounio-loom-host-durable-lane-supervisor-freeze-selftest: PASS semantic_authority=Sounio action=9032 stage=SEMANTICS_FROZEN executable_sha256=%s same_physical=SAME_PHYSICAL_REATTACH lineage=LINEAGE_RESURRECTION guardian_start_mismatch=DENY526 guardian_loss_without_lineage=DENY529 causal_sabotage=PASS python_executed=false rust_executed=false material_execution=false host_guardian_attached=false transport_pod_deleted=false production_activation=false parity_open=false claim_ready=false\n' \
  "$(manifest_value executable_sha256)"
