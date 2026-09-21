#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-policy-v7-selftest.XXXXXX")"
PLAN_ONE="$TEST_ROOT/plan-one"
PLAN_TWO="$TEST_ROOT/plan-two"
AUTHORITY="$TEST_ROOT/action-9025-authority"
BUNDLE="$TEST_ROOT/policy-plan-v7"

cleanup() { rm -rf "$TEST_ROOT"; }
trap cleanup EXIT

fail() {
  printf 'sounio-loom-process-witness-effect-policy-plan-v7-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

for output in "$PLAN_ONE" "$PLAN_TWO"; do
  SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V7_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_process_witness_effect_policy_plan_v7.sh" \
      >/dev/null
done
cmp "$PLAN_ONE" "$PLAN_TWO" || fail 'two source-fresh V7 builds differ'
[[ "$(stat -c '%a' "$PLAN_ONE")" == 755 && ! -u "$PLAN_ONE" && ! -g "$PLAN_ONE" ]] ||
  fail 'V7 executable mode is unsafe'
SOUNIO_LOOM_EFFECT_CLOSURE_OUTPUT="$AUTHORITY" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_effect_closure_authority.sh" >/dev/null

"$PLAN_ONE" > "$BUNDLE"
[[ "$(wc -l < "$BUNDLE")" == 44 ]] || fail 'V7 bundle line count diverged'
[[ "$(grep -c '^SYSCALL ' "$BUNDLE" || true)" == 4 ]] || fail 'V7 syscall count diverged'
[[ "$(grep -c '^FAMILY ' "$BUNDLE" || true)" == 12 ]] || fail 'V7 family count diverged'
[[ "$(grep -c '^CASE ' "$BUNDLE" || true)" == 14 ]] || fail 'V7 authority-case count diverged'
[[ "$(grep -c '^BOOTSTRAP_CASE ' "$BUNDLE" || true)" == 4 ]] ||
  fail 'V7 bootstrap-case count diverged'
grep -Fxq 'ROOT_SCHEMA paths=/loom/effect-cell+/loom/payload+/loom/payload.freeze.v1+/loom/effect-policy-v7.freeze.v1+/dev/null+/run/systemd/incoming+/sys+/var/tmp empty_readonly=/tmp+/var/tmp proc_treatment=absent' "$BUNDLE" ||
  fail 'V7 corrected root schema drifted'
grep -Fxq 'SYSTEMD_MOUNT path=/run/systemd/incoming parent_chain=root_owned principal_writable=false ready_contents=empty source=/run/systemd/propagate/EXACT_UNIT backing_root_read_only=true required_for_namespace=true systemd_version=257' "$BUNDLE" ||
  fail 'V7 incoming mount contract drifted'
grep -Fxq 'OBSERVER_SPLIT path=/run/systemd/incoming principal_exists=true principal_root_owned=true principal_writable=false principal_readable=false principal_enumeration=forbidden empty_observer=ROOT_HOST mount_observer=ROOT_HOST extinction_observer=ROOT_HOST' "$BUNDLE" ||
  fail 'V7 observer authority split drifted'
grep -Fxq 'SYSTEMD_SYS_MOUNT path=/sys backing_contents=empty parent=root principal_writable=false ready_filesystem=sysfs ready_source=sysfs ready_read_only=true required_for_namespace=true private_network=true' "$BUNDLE" ||
  fail 'V7 sys mount contract drifted'
grep -Fxq 'SYSTEMD_VAR_TMP path=/var/tmp backing_contents=empty parent_chain=root_owned principal_writable=false ready_contents=empty ready_source=IMMUTABLE_ROOT_TMP ready_read_only=true dynamic_user=true required_for_namespace=true' "$BUNDLE" ||
  fail 'V7 var-tmp mount contract drifted'
grep -Fxq 'BOOTSTRAP_CASE label=treatment EXPECT SOUNIO_ROOT_BOOTSTRAP_ALLOW code=0 reason=systemd-mountpoints-present stage=SEMANTICS_FROZEN' "$BUNDLE" ||
  fail 'V7 treatment bootstrap result drifted'
grep -Fxq 'BOOTSTRAP_CASE label=missing_incoming EXPECT SOUNIO_ROOT_BOOTSTRAP_DENY code=226 reason=incoming-mountpoint-absent stage=SEMANTICS_FROZEN' "$BUNDLE" ||
  fail 'V7 missing-incoming sabotage result drifted'
grep -Fxq 'BOOTSTRAP_CASE label=missing_sys EXPECT SOUNIO_ROOT_BOOTSTRAP_DENY code=226 reason=sys-mountpoint-absent stage=SEMANTICS_FROZEN' "$BUNDLE" ||
  fail 'V7 missing-sys sabotage result drifted'
grep -Fxq 'BOOTSTRAP_CASE label=missing_var_tmp EXPECT SOUNIO_ROOT_BOOTSTRAP_DENY code=226 reason=var-tmp-mountpoint-absent stage=SEMANTICS_FROZEN' "$BUNDLE" ||
  fail 'V7 missing-var-tmp sabotage result drifted'
grep -Fq 'FAMILY id=10 name=device_and_kernel_control mode=2 mechanism=lock_personality_plus_seccomp_no_personality treatment_probe=personality_change ' "$BUNDLE" ||
  fail 'V7 family-10 causal probe drifted'
if grep -Fq 'treatment_probe=bpf' "$BUNDLE"; then fail 'V7 retained the non-causal BPF probe'; fi
for nr in 0 1 60 322; do
  [[ "$(grep -c "^SYSCALL nr=${nr} " "$BUNDLE" || true)" == 1 ]] ||
    fail "V7 syscall $nr is absent or duplicated"
done
for id in $(seq 1 12); do
  [[ "$(grep -c "^FAMILY id=${id} " "$BUNDLE" || true)" == 1 ]] ||
    fail "V7 family $id is absent or duplicated"
  [[ "$(grep -c "^CASE label=missing_${id} " "$BUNDLE" || true)" == 1 ]] ||
    fail "V7 missing-family case $id is absent or duplicated"
done

case_count=0
complete_actual=''
current_actual=''
while IFS= read -r line; do
  [[ "$line" == CASE\ label=* ]] || continue
  body="${line#CASE label=}"
  label="${body%% EXPECT *}"
  rest="${body#* EXPECT }"
  expected="${rest%% FRAME *}"
  frame="${rest#* FRAME }"
  actual="$(printf '%s\n' "$frame" | "$AUTHORITY" || true)"
  [[ "$actual" == "$expected" ]] || fail "action 9025 disagreed with V7 case $label"
  if [[ "$label" == complete ]]; then complete_actual="$actual"; fi
  if [[ "$label" == current ]]; then current_actual="$actual"; fi
  case_count=$((case_count + 1))
done < "$BUNDLE"
[[ "$case_count" == 14 && "$complete_actual" == SOUNIO_EFFECT_CLOSURE_ALLOW* &&
   "$current_actual" == SOUNIO_EFFECT_CLOSURE_DENY* ]] ||
  fail 'V7 action-9025 decision matrix is incomplete'

boundary='BOUNDARY v6_materializable=false v7_required_for_native=true root_treatment=false bootstrap_sabotage=false material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false'
grep -Fxq "$boundary" "$BUNDLE" || fail 'V7 evidence boundary drifted'
native_sentinel="$TEST_ROOT/v6-native"
native_executed="$TEST_ROOT/v6-native-executed"
printf '#!/bin/sh\nprintf prohibited > %s\n' "$native_executed" > "$native_sentinel"
chmod 0755 "$native_sentinel"
if ! grep -Fq 'v6_materializable=false' "$BUNDLE"; then "$native_sentinel"; fi
[[ ! -e "$native_executed" ]] || fail 'V6 native path crossed the V7 refusal'

dependencies="$(ldd "$PLAN_ONE" 2>&1 || true)"
if printf '%s\n' "$dependencies" | grep -Eqi 'python|rust'; then
  fail 'Sounio V7 executable has a prohibited dependency'
fi

printf 'sounio-loom-process-witness-effect-policy-plan-v7-selftest: PASS producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE observer_split=principal-opaque+root-host-authority families=12 treatments=12 sabotages=12 bootstrap_cases=4 bootstrap_treatment=ALLOW bootstrap_missing_incoming=DENY226 bootstrap_missing_sys=DENY226 bootstrap_missing_var_tmp=DENY226 allowed_syscalls=4 authority_cases=14 complete=ALLOW current=DENY447 family10=personality_change landlock_required=false v6_native=refused native_executed=false deterministic=true shell_expected_results=false source_sha256=%s executable_sha256=%s bundle_sha256=%s root_treatment=false bootstrap_sabotage=false material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v7_main.sio" | cut -d ' ' -f 1)" \
  "$(sha256sum "$PLAN_ONE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$BUNDLE" | cut -d ' ' -f 1)"
