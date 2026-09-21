#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-policy-v2-selftest.XXXXXX")"
PLAN_ONE="$TEST_ROOT/plan-one"
PLAN_TWO="$TEST_ROOT/plan-two"
AUTHORITY="$TEST_ROOT/action-9025-authority"
BUNDLE="$TEST_ROOT/policy-plan-v2"
V1_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan.freeze.v1"
PROCESS_MANIFEST="$ROOT_DIR/tools/loom/process_witness_host.runtime.v1"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-process-witness-effect-policy-plan-v2-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

for output in "$PLAN_ONE" "$PLAN_TWO"; do
  SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V2_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_process_witness_effect_policy_plan_v2.sh" \
      >/dev/null
done
cmp "$PLAN_ONE" "$PLAN_TWO" || fail 'two source-fresh Sounio V2 builds differ'
[[ "$(stat -c '%a' "$PLAN_ONE")" == 755 && ! -u "$PLAN_ONE" &&
   ! -g "$PLAN_ONE" ]] || fail 'V2 policy executable mode is unsafe'

SOUNIO_LOOM_EFFECT_CLOSURE_OUTPUT="$AUTHORITY" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_effect_closure_authority.sh" \
    >/dev/null

"$PLAN_ONE" > "$BUNDLE"
[[ "$(wc -l < "$BUNDLE")" == 36 ]] || fail 'V2 policy line count diverged'
[[ "$(sed -n '1p' "$BUNDLE")" == \
  'LOOM_PROCESS_WITNESS_EFFECT_POLICY_PLAN_V2 producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 schema=2 families=12 treatments=12 sabotages=12 allowed_syscalls=4' ]] ||
  fail 'V2 policy metadata diverged'
[[ "$(sed -n '2p' "$BUNDLE")" == \
  'PARENTS effect_closure_manifest_sha256=c1f0cf93f8427acdf794246a11c3551e265a09be12a3cd000bad25b707e8ca91 process_witness_manifest_sha256=eda00fee106a9f4090d381194b9f1bcd3838f3dcc0bafb0c7769a0877e05aa00 v1_policy_manifest_sha256=14ee27eee71f04d1aa5462426379b37bb9c775215e94e17a864dbea308e43f21 garden_commit=13ccfc7896' ]] ||
  fail 'V2 parent binding diverged'

[[ "$(grep -c '^SYSCALL ' "$BUNDLE" || true)" == 4 ]] ||
  fail 'exact positive syscall surface is not four rows'
for nr in 0 1 60 322; do
  [[ "$(grep -c "^SYSCALL nr=${nr} " "$BUNDLE" || true)" == 1 ]] ||
    fail "syscall $nr is absent or duplicated"
done
grep -Fxq 'SYSCALL nr=0 name=read seccomp_args=fd0 object=close_channel effect_family=4' "$BUNDLE" ||
  fail 'read constraint drifted'
grep -Fxq 'SYSCALL nr=1 name=write seccomp_args=fd1_or_fd2 object=bounded_receipt_streams effect_family=4' "$BUNDLE" ||
  fail 'write constraint drifted'
grep -Fxq 'SYSCALL nr=60 name=exit seccomp_args=any_status object=terminal_wait_status effect_family=2' "$BUNDLE" ||
  fail 'exit constraint drifted'
grep -Fxq 'SYSCALL nr=322 name=execveat seccomp_args=fd3_and_AT_EMPTY_PATH cell_args=empty_path+frozen_argv+empty_env fd_cloexec=true object=preopened_hashed_Sounio_payload effect_family=1' "$BUNDLE" ||
  fail 'execveat constraint drifted'
grep -Fxq 'SECCOMP architecture=AUDIT_ARCH_X86_64 architecture_mismatch=KILL_PROCESS default_action=ERRNO_EP1 allowlist_kind=positive argument_constraints=required blacklist_fallback=false' "$BUNDLE" ||
  fail 'seccomp closed-world policy drifted'
grep -Fxq 'LANDLOCK required=true rules=preopened_object_read_only installed_before_seccomp=true pathname_authority=false fallback=false' "$BUNDLE" ||
  fail 'Landlock object policy drifted'

[[ "$(grep -c '^FAMILY ' "$BUNDLE" || true)" == 12 ]] ||
  fail 'V2 family count diverged'
[[ "$(grep -c '^CASE ' "$BUNDLE" || true)" == 14 ]] ||
  fail 'V2 action-9025 case count diverged'
for id in $(seq 1 12); do
  [[ "$(grep -c "^FAMILY id=${id} " "$BUNDLE" || true)" == 1 ]] ||
    fail "family $id is absent or duplicated"
  [[ "$(grep -c "^CASE label=missing_${id} " "$BUNDLE" || true)" == 1 ]] ||
    fail "missing-family case $id is absent or duplicated"
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
  [[ "$frame" == '9025 3 '* && "$frame" != *$'\n'* && ${#frame} -le 65535 ]] ||
    fail "case $label has an invalid action-9025 frame"
  actual="$(printf '%s\n' "$frame" | "$AUTHORITY" || true)"
  [[ "$actual" == "$expected" ]] ||
    fail "frozen action 9025 disagreed with Sounio V2 case $label: $actual"
  if [[ "$label" == complete ]]; then complete_actual="$actual"; fi
  if [[ "$label" == current ]]; then current_actual="$actual"; fi
  case_count=$((case_count + 1))
done < "$BUNDLE"
[[ "$case_count" == 14 ]] || fail 'not every V2 authority case was judged'
[[ "$complete_actual" == SOUNIO_EFFECT_CLOSURE_ALLOW* ]] ||
  fail 'complete hypothetical V2 frame did not reach Sounio ALLOW'
[[ "$current_actual" == SOUNIO_EFFECT_CLOSURE_DENY* ]] ||
  fail 'current material frame did not remain closed'

native_sentinel="$TEST_ROOT/native-consumer"
native_executed="$TEST_ROOT/native-executed"
printf '#!/bin/sh\nprintf prohibited > %s\n' "$native_executed" > "$native_sentinel"
chmod 0755 "$native_sentinel"
v1_decision="$(grep '^NATIVE_CONSUMPTION binding=v1_only EXPECT ' "$BUNDLE" || true)"
[[ "$v1_decision" == \
  'NATIVE_CONSUMPTION binding=v1_only EXPECT REFUSED reason=exact-syscall-surface-absent' ]] ||
  fail 'Sounio V2 did not refuse V1-only native consumption'
if [[ "$v1_decision" == *' EXPECT ALLOW '* ]]; then "$native_sentinel"; fi
[[ ! -e "$native_executed" ]] || fail 'V1-only native consumer was launched'

boundary='BOUNDARY v1_sufficient_for_native=false v2_required_for_native=true material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false'
grep -Fxq "$boundary" "$BUNDLE" || fail 'V2 evidence boundary drifted'
grep -Fxq 'material_coverage=false' "$V1_MANIFEST" || fail 'V1 boundary drifted'
grep -Fxq 'complete_effects=false' "$PROCESS_MANIFEST" ||
  fail 'ProcessWitness parent boundary drifted'

python_sentinel="$TEST_ROOT/python3"
python_executed="$TEST_ROOT/python-executed"
printf '#!/bin/sh\nprintf prohibited > %s\n' "$python_executed" > "$python_sentinel"
chmod 0755 "$python_sentinel"
if [[ "$current_actual" == SOUNIO_EFFECT_CLOSURE_ALLOW* ]]; then
  "$python_sentinel"
fi
[[ ! -e "$python_executed" ]] || fail 'Python oracle crossed the Sounio refusal'

dependencies="$(ldd "$PLAN_ONE" 2>&1 || true)"
if printf '%s\n' "$dependencies" | grep -Eqi 'python|rust'; then
  fail 'Sounio V2 executable has a prohibited runtime dependency'
fi

printf 'sounio-loom-process-witness-effect-policy-plan-v2-selftest: PASS producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 families=12 treatments=12 sabotages=12 allowed_syscalls=4 syscall_surface=0+1+60+322 authority_cases=14 complete=ALLOW current=DENY447 missing_known=DENY447x11 missing_unknown=DENY452 v1_native=refused native_executed=false python_control=refused python_executed=false deterministic=true shell_expected_results=false source_sha256=%s executable_sha256=%s bundle_sha256=%s material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v2_main.sio" | cut -d ' ' -f 1)" \
  "$(sha256sum "$PLAN_ONE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$BUNDLE" | cut -d ' ' -f 1)"
