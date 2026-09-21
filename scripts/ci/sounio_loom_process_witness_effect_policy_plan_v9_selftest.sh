#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-policy-v9-selftest.XXXXXX")"
PLAN_ONE="$TEST_ROOT/plan-one"
PLAN_TWO="$TEST_ROOT/plan-two"
AUTHORITY="$TEST_ROOT/action-9025-authority"
BUNDLE="$TEST_ROOT/policy-plan-v9"

cleanup() { rm -rf "$TEST_ROOT"; }
trap cleanup EXIT

fail() {
  printf 'sounio-loom-process-witness-effect-policy-plan-v9-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

for output in "$PLAN_ONE" "$PLAN_TWO"; do
  SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V9_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_process_witness_effect_policy_plan_v9.sh" \
      >/dev/null
done
cmp "$PLAN_ONE" "$PLAN_TWO" || fail 'two source-fresh V9 builds differ'
[[ "$(stat -c '%a' "$PLAN_ONE")" == 755 && ! -u "$PLAN_ONE" && ! -g "$PLAN_ONE" ]] ||
  fail 'V9 executable mode is unsafe'
SOUNIO_LOOM_EFFECT_CLOSURE_OUTPUT="$AUTHORITY" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_effect_closure_authority.sh" >/dev/null

"$PLAN_ONE" > "$BUNDLE"
[[ "$(wc -l < "$BUNDLE")" == 46 ]] || fail 'V9 bundle line count diverged'
[[ "$(grep -c '^SYSCALL ' "$BUNDLE" || true)" == 4 ]] || fail 'V9 syscall count diverged'
[[ "$(grep -c '^FAMILY ' "$BUNDLE" || true)" == 12 ]] || fail 'V9 family count diverged'
[[ "$(grep -c '^CASE ' "$BUNDLE" || true)" == 14 ]] || fail 'V9 authority-case count diverged'
[[ "$(grep -c '^BOOTSTRAP_CASE ' "$BUNDLE" || true)" == 4 ]] ||
  fail 'V9 bootstrap-case count diverged'
grep -Fxq 'ROOT_SCHEMA paths=/loom/effect-cell+/loom/payload+/loom/payload.freeze.v1+/loom/effect-policy-v9.freeze.v1+/dev/null+/run/systemd/incoming+/sys+/var/tmp empty_readonly=/tmp+/var/tmp proc_treatment=absent' "$BUNDLE" ||
  fail 'V9 corrected root schema drifted'
grep -Fxq 'FILE_BOUNDS effect_cell_min=1 effect_cell_max=16777216 payload_min=1 payload_max=1048576 policy_manifest_min=1 policy_manifest_max=65536 payload_manifest_min=1 payload_manifest_max=65536 units=bytes identity=size+root_owned+single_link+non_writable+sha256 diagnostic=object+observed_size+configured_max' "$BUNDLE" ||
  fail 'V9 typed immutable-root file bounds drifted'
grep -Fxq 'SYSTEMD_MOUNT path=/run/systemd/incoming parent_chain=root_owned principal_writable=false ready_contents=empty source=/run/systemd/propagate/EXACT_UNIT backing_root_read_only=true required_for_namespace=true systemd_version=257' "$BUNDLE" ||
  fail 'V9 incoming mount contract drifted'
grep -Fxq 'OBSERVER_SPLIT path=/run/systemd/incoming principal_exists=true principal_root_owned=true principal_writable=false principal_readable=false principal_enumeration=forbidden empty_observer=ROOT_HOST mount_observer=ROOT_HOST extinction_observer=ROOT_HOST' "$BUNDLE" ||
  fail 'V9 observer authority split drifted'
grep -Fxq 'SYSTEMD_SYS_MOUNT path=/sys backing_contents=empty parent=root principal_writable=false ready_filesystem=sysfs ready_source=sysfs ready_read_only=true required_for_namespace=true private_network=true' "$BUNDLE" ||
  fail 'V9 sys mount contract drifted'
grep -Fxq 'SYSTEMD_VAR_TMP path=/var/tmp backing_contents=empty parent_chain=root_owned principal_writable=false ready_contents=empty ready_source=IMMUTABLE_ROOT_TMP ready_read_only=true dynamic_user=true required_for_namespace=true' "$BUNDLE" ||
  fail 'V9 var-tmp mount contract drifted'
grep -Fxq 'SYSTEMD_EFFECTIVE dynamic_user=true private_tmp=disconnected property_private_tmp_observed=yes protect_system=strict protect_home=read-only property_authority=CONFIGURATION_ONLY filesystem_authority=ROOT_HOST_MOUNTINFO temporary_mounts=/tmp+/var/tmp temporary_sources=SAME_IMMUTABLE_ROOT_TMP temporary_read_only=true temporary_empty=true forbidden_mounts=/proc+/home+/root+/run+/var+/etc' "$BUNDLE" ||
  fail 'V9 effective systemd mount authority drifted'
grep -Fxq 'BOOTSTRAP_CASE label=treatment EXPECT SOUNIO_ROOT_BOOTSTRAP_ALLOW code=0 reason=systemd-mountpoints-present stage=SEMANTICS_FROZEN' "$BUNDLE" ||
  fail 'V9 treatment bootstrap result drifted'
grep -Fxq 'BOOTSTRAP_CASE label=missing_incoming EXPECT SOUNIO_ROOT_BOOTSTRAP_DENY code=226 reason=incoming-mountpoint-absent stage=SEMANTICS_FROZEN' "$BUNDLE" ||
  fail 'V9 missing-incoming sabotage result drifted'
grep -Fxq 'BOOTSTRAP_CASE label=missing_sys EXPECT SOUNIO_ROOT_BOOTSTRAP_DENY code=226 reason=sys-mountpoint-absent stage=SEMANTICS_FROZEN' "$BUNDLE" ||
  fail 'V9 missing-sys sabotage result drifted'
grep -Fxq 'BOOTSTRAP_CASE label=missing_var_tmp EXPECT SOUNIO_ROOT_BOOTSTRAP_DENY code=226 reason=var-tmp-mountpoint-absent stage=SEMANTICS_FROZEN' "$BUNDLE" ||
  fail 'V9 missing-var-tmp sabotage result drifted'
grep -Fq 'FAMILY id=10 name=device_and_kernel_control mode=2 mechanism=lock_personality_plus_seccomp_no_personality treatment_probe=personality_change ' "$BUNDLE" ||
  fail 'V9 family-10 causal probe drifted'
if grep -Fq 'treatment_probe=bpf' "$BUNDLE"; then fail 'V9 retained the non-causal BPF probe'; fi
for nr in 0 1 60 322; do
  [[ "$(grep -c "^SYSCALL nr=${nr} " "$BUNDLE" || true)" == 1 ]] ||
    fail "V9 syscall $nr is absent or duplicated"
done
for id in $(seq 1 12); do
  [[ "$(grep -c "^FAMILY id=${id} " "$BUNDLE" || true)" == 1 ]] ||
    fail "V9 family $id is absent or duplicated"
  [[ "$(grep -c "^CASE label=missing_${id} " "$BUNDLE" || true)" == 1 ]] ||
    fail "V9 missing-family case $id is absent or duplicated"
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
  [[ "$actual" == "$expected" ]] || fail "action 9025 disagreed with V9 case $label"
  if [[ "$label" == complete ]]; then complete_actual="$actual"; fi
  if [[ "$label" == current ]]; then current_actual="$actual"; fi
  case_count=$((case_count + 1))
done < "$BUNDLE"
[[ "$case_count" == 14 && "$complete_actual" == SOUNIO_EFFECT_CLOSURE_ALLOW* &&
   "$current_actual" == SOUNIO_EFFECT_CLOSURE_DENY* ]] ||
  fail 'V9 action-9025 decision matrix is incomplete'

boundary='BOUNDARY v8_materializable=false v9_required_for_native=true root_treatment=false bootstrap_sabotage=false material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false'
grep -Fxq "$boundary" "$BUNDLE" || fail 'V9 evidence boundary drifted'
native_sentinel="$TEST_ROOT/v8-native"
native_executed="$TEST_ROOT/v8-native-executed"
printf '#!/bin/sh\nprintf prohibited > %s\n' "$native_executed" > "$native_sentinel"
chmod 0755 "$native_sentinel"
if ! grep -Fq 'v8_materializable=false' "$BUNDLE"; then "$native_sentinel"; fi
[[ ! -e "$native_executed" ]] || fail 'V8 native path crossed the V9 refusal'

dependencies="$(ldd "$PLAN_ONE" 2>&1 || true)"
if printf '%s\n' "$dependencies" | grep -Eqi 'python|rust'; then
  fail 'Sounio V9 executable has a prohibited dependency'
fi

printf 'sounio-loom-process-witness-effect-policy-plan-v9-selftest: PASS producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE observer_split=principal-opaque+root-host-authority typed_file_bounds=effect-cell-16MiB+payload-1MiB+manifests-64KiB effective_mount_truth=DynamicUser+disconnected+strict+read-only families=12 treatments=12 sabotages=12 bootstrap_cases=4 bootstrap_treatment=ALLOW bootstrap_missing_incoming=DENY226 bootstrap_missing_sys=DENY226 bootstrap_missing_var_tmp=DENY226 allowed_syscalls=4 authority_cases=14 complete=ALLOW current=DENY447 family10=personality_change landlock_required=false v8_native=refused native_executed=false deterministic=true shell_expected_results=false source_sha256=%s executable_sha256=%s bundle_sha256=%s root_treatment=false bootstrap_sabotage=false material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v9_main.sio" | cut -d ' ' -f 1)" \
  "$(sha256sum "$PLAN_ONE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$BUNDLE" | cut -d ' ' -f 1)"
