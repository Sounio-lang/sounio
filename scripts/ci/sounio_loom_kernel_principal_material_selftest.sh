#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-kernel-principal-material.XXXXXX")"
AUTHORITY="$TEST_ROOT/kernel-principal-authority"
PROBE_ONE="$TEST_ROOT/kernel-principal-probe-one"
PROBE_TWO="$TEST_ROOT/kernel-principal-probe-two"
FROZEN_MANIFEST="$ROOT_DIR/tools/loom/kernel_principal_authority.freeze.v1"
FROZEN_MANIFEST_SHA256='4cbf80364b1d266ab1417103642cc66917acdc4f5ec68d8cdba92ece34db07dc'

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-kernel-principal-material-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

line_hash() {
  local sum
  sum="$(printf '%s\n' "$1" | sha256sum)"
  printf '%s' "${sum%% *}"
}

field() {
  local line="$1" key="$2" token
  for token in $line; do
    if [[ "$token" == "$key="* ]]; then
      printf '%s' "${token#*=}"
      return 0
    fi
  done
  fail "receipt omitted $key"
}

[[ "$(sha256sum "$FROZEN_MANIFEST" | cut -d' ' -f1)" == "$FROZEN_MANIFEST_SHA256" ]] ||
  fail 'frozen action 9026 manifest drifted'
bash "$ROOT_DIR/scripts/ci/sounio_loom_kernel_principal_authority_freeze_selftest.sh" >/dev/null

SOUNIO_LOOM_KERNEL_PRINCIPAL_OUTPUT="$AUTHORITY" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_principal_authority.sh" >/dev/null
SOUNIO_LOOM_KERNEL_PRINCIPAL_PROBE_OUTPUT="$PROBE_ONE" \
  bash "$ROOT_DIR/scripts/dev/build_loom_kernel_principal_probe.sh" >/dev/null
SOUNIO_LOOM_KERNEL_PRINCIPAL_PROBE_OUTPUT="$PROBE_TWO" \
  bash "$ROOT_DIR/scripts/dev/build_loom_kernel_principal_probe.sh" >/dev/null
cmp "$PROBE_ONE" "$PROBE_TWO" || fail 'two C++ material-probe builds differ'

run_probe() {
  local mode="$1" output receipt digest frame calculated decision
  if [[ "$mode" == normal ]]; then
    output="$($PROBE_ONE)"
  else
    output="$($PROBE_ONE --simulate-helper-exit-only)"
  fi
  receipt="$(printf '%s\n' "$output" | sed -n '1p')"
  digest="$(printf '%s\n' "$output" | sed -n '2s/^LOOM_KERNEL_PRINCIPAL_RECEIPT_SHA256 //p')"
  frame="$(printf '%s\n' "$output" | sed -n '3s/^SOUNIO_FRAME //p')"
  [[ "$receipt" == 'LOOM_KERNEL_PRINCIPAL_MATERIAL '* ]] || fail "$mode receipt is missing"
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || fail "$mode receipt digest is malformed"
  calculated="$(line_hash "$receipt")"
  [[ "$calculated" == "$digest" ]] || fail "$mode receipt digest differs"
  [[ -n "$frame" ]] || fail "$mode Sounio frame is missing"
  decision="$(printf '%s\n' "$frame" | "$AUTHORITY" || true)"
  printf '%s\n%s\n%s\n' "$receipt" "$digest" "$decision"
}

normal="$(run_probe normal)"
normal_receipt="$(printf '%s\n' "$normal" | sed -n '1p')"
normal_digest="$(printf '%s\n' "$normal" | sed -n '2p')"
normal_decision="$(printf '%s\n' "$normal" | sed -n '3p')"
[[ "$(field "$normal_receipt" ranges_disjoint)" == 1 ]] || fail 'subordinate ranges overlap the outer principal'
[[ "$(field "$normal_receipt" helpers_setuid_root)" == 1 ]] || fail 'uidmap helpers are not installed as setuid root'
[[ "$(field "$normal_receipt" user_namespace)" == 1 ]] || fail 'basic user namespace is unavailable'
[[ "$(field "$normal_receipt" current_host_uid)" == 1000 && \
   "$(field "$normal_receipt" current_principal_distinct)" == 0 ]] ||
  fail 'map-current-user was laundered as a distinct principal'
[[ "$(field "$normal_receipt" subordinate_map_exit)" != 0 && \
   "$(field "$normal_receipt" uid_map_exact)" == 0 && \
   "$(field "$normal_receipt" gid_map_exact)" == 0 && \
   "$(field "$normal_receipt" mapping_materialized)" == 0 ]] ||
  fail 'current pod unexpectedly materialized a subordinate mapping'
[[ "$(field "$normal_receipt" pid_namespace)" == 1 && \
   "$(field "$normal_receipt" mount_namespace)" == 1 && \
   "$(field "$normal_receipt" cgroup_v2)" == 1 ]] ||
  fail 'current namespace substrate receipt is incomplete'
[[ "$(field "$normal_receipt" outer_privilege_regain)" == 1 && \
   "$(field "$normal_receipt" material_isolation)" == 0 ]] ||
  fail 'passwordless privilege regain was not treated as a blocker'
[[ "$normal_decision" == \
  'SOUNIO_KERNEL_PRINCIPAL_DENY code=455 reason=subordinate-allocation-incomplete stage=SEMANTICS_FROZEN' ]] ||
  fail "Sounio did not refuse the current material probe: $normal_decision"

sabotaged="$(run_probe sabotage)"
sabotaged_receipt="$(printf '%s\n' "$sabotaged" | sed -n '1p')"
sabotaged_digest="$(printf '%s\n' "$sabotaged" | sed -n '2p')"
sabotaged_decision="$(printf '%s\n' "$sabotaged" | sed -n '3p')"
[[ "$(field "$sabotaged_receipt" sabotage)" == helper-exit-only && \
   "$(field "$sabotaged_receipt" subordinate_map_exit)" == 0 && \
   "$(field "$sabotaged_receipt" mapping_materialized)" == 0 ]] ||
  fail 'helper-exit sabotage did not preserve the absent map witness'
[[ "$sabotaged_decision" == "$normal_decision" ]] ||
  fail 'zero helper exit laundered the absent mapping'
[[ "$normal_digest" != "$sabotaged_digest" ]] || fail 'sabotage did not change the material receipt'

printf '%s\n' \
  "sounio-loom-kernel-principal-material-selftest: PASS semantic_authority=Sounio operational_realization=C++20+Linux action=9026 frozen_manifest_sha256=$FROZEN_MANIFEST_SHA256 current_material=DENY455 helpers=setuid-root mapping_attempt=EPERM map_current_user=outer-uid-not-distinct helper_exit_sabotage=zero-exit+DENY455 receipt=sha256-bound pid_namespace=present mount_namespace=present cgroup_v2=present outer_privilege_regain=present material_isolation=false exec_attached=false commit_attached=false ci_attached=false"
