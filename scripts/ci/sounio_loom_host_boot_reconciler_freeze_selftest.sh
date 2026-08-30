#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/host_boot_reconciler.freeze.v1"

fail() {
  printf 'sounio-loom-host-boot-reconciler-freeze-selftest: FAIL: %s\n' "$*" >&2
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
expect_value schema loom-host-boot-reconciler-freeze-v1
expect_value stage SEMANTICS_FROZEN
expect_value producing_language Sounio
expect_value language_role SEMANTIC_AUTHORITY
expect_value action 9041
expect_value observation_field_count 30
expect_value case_count 14
expect_value causal_sabotage PASS
expect_value load_bearing_rule guardian_start_verified
expect_value automatic_lineage_resurrection false
expect_value same_pty_after_guardian_loss false
expect_value expected_results_encoded_in_operational_layer false
expect_value operational_parity false
expect_value systemd_installed false
expect_value service_enabled false
expect_value production_activation false
expect_value parity_open false
expect_value claim_ready false

for key in garden source entrypoint build_script selftest evidence toolchain_wrapper toolchain_compiler; do
  expect_hash "$(manifest_value "${key}_path")" "$(manifest_value "${key}_sha256")"
done

source_commit="$(manifest_value sounio_executable_commit)"
git -C "$ROOT_DIR" cat-file -e "${source_commit}^{commit}" || fail 'Sounio executable commit is absent'
for key in garden source entrypoint build_script selftest; do
  path="$(manifest_value "${key}_path")"
  at_commit="$(git -C "$ROOT_DIR" show "${source_commit}:$path" | sha256sum | cut -d ' ' -f 1)"
  [[ "$at_commit" == "$(manifest_value "${key}_sha256")" ]] ||
    fail "$key is not bound to the Sounio executable commit"
done

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-host-boot-freeze.XXXXXX")"
trap 'rm -rf "$work"' EXIT
for ordinal in one two; do
  SOUNIO_LOOM_HOST_BOOT_OUTPUT="$work/runtime-$ordinal" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_host_boot_reconciler.sh" >/dev/null
done
cmp "$work/runtime-one" "$work/runtime-two" || fail 'Sounio rebuild is nondeterministic'
[[ "$(sha256sum "$work/runtime-one" | cut -d ' ' -f 1)" == \
   "$(manifest_value executable_sha256)" ]] || fail 'Sounio executable hash drifted'

active='9041 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1'
recover='9041 3 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1'
lost='9041 3 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 0 1 1 1 1 1 1'
sabotage='9041 3 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1'
[[ "$(printf '%s\n' "$active" | "$work/runtime-one")" == \
   "$(manifest_value active_decision)" ]] || fail 'active decision drifted'
[[ "$(printf '%s\n' "$recover" | "$work/runtime-one")" == \
   "$(manifest_value recover_decision)" ]] || fail 'recover decision drifted'
[[ "$(printf '%s\n' "$lost" | "$work/runtime-one")" == \
   "$(manifest_value guardian_loss_decision)" ]] || fail 'Guardian-loss decision drifted'
set +e
sabotage_output="$(printf '%s\n' "$sabotage" | "$work/runtime-one" 2>&1)"
sabotage_code=$?
set -e
[[ $sabotage_code -eq 42 && "$sabotage_output" == \
   "$(manifest_value guardian_start_mismatch_decision)" ]] ||
  fail 'Guardian start-tick sabotage drifted'

result="$(bash "$ROOT_DIR/scripts/ci/sounio_loom_host_boot_reconciler_selftest.sh" | tail -1)"
[[ "$result" == "$(manifest_value result)" ]] || fail 'Sounio selftest result drifted'
[[ "$(printf '%s' "$result" | sha256sum | cut -d ' ' -f 1)" == \
   "$(manifest_value result_sha256)" ]] || fail 'Sounio selftest result hash drifted'

printf 'sounio-loom-host-boot-reconciler-freeze-selftest: PASS semantic_authority=Sounio action=9041 stage=SEMANTICS_FROZEN executable_sha256=%s active=NOOP_ACTIVE recover=RECOVER_SAME_PHYSICAL guardian_loss=HOLD_LINEAGE_REQUIRED guardian_start_mismatch=DENY545 causal_sabotage=PASS automatic_lineage_resurrection=false python_executed=false rust_executed=false operational_parity=false production_activation=false claim_ready=false\n' \
  "$(manifest_value executable_sha256)"
