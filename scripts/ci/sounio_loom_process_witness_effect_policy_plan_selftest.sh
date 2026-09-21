#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-policy-selftest.XXXXXX")"
PLAN_ONE="$TEST_ROOT/plan-one"
PLAN_TWO="$TEST_ROOT/plan-two"
AUTHORITY="$TEST_ROOT/action-9025-authority"
BUNDLE="$TEST_ROOT/policy-plan.v1"
EFFECT_MANIFEST="$ROOT_DIR/tools/loom/effect_closure_authority.freeze.v1"
PROCESS_MANIFEST="$ROOT_DIR/tools/loom/process_witness_host.runtime.v1"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-process-witness-effect-policy-plan-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

for output in "$PLAN_ONE" "$PLAN_TWO"; do
  SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_process_witness_effect_policy_plan.sh" \
      >/dev/null
done
cmp "$PLAN_ONE" "$PLAN_TWO" || fail 'two source-fresh Sounio plan builds differ'
[[ "$(stat -c '%a' "$PLAN_ONE")" == 755 && ! -u "$PLAN_ONE" &&
   ! -g "$PLAN_ONE" ]] || fail 'policy-plan executable mode is unsafe'

SOUNIO_LOOM_EFFECT_CLOSURE_OUTPUT="$AUTHORITY" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_effect_closure_authority.sh" \
    >/dev/null

"$PLAN_ONE" > "$BUNDLE"
[[ "$(wc -l < "$BUNDLE")" == 29 ]] || fail 'policy-plan line count diverged'
[[ "$(sed -n '1p' "$BUNDLE")" == \
  'LOOM_PROCESS_WITNESS_EFFECT_POLICY_PLAN_V1 producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 schema=1 families=12 treatments=12 sabotages=12' ]] ||
  fail 'policy-plan metadata diverged'
[[ "$(sed -n '2p' "$BUNDLE")" == \
  'PARENTS effect_closure_manifest_sha256=c1f0cf93f8427acdf794246a11c3551e265a09be12a3cd000bad25b707e8ca91 process_witness_manifest_sha256=eda00fee106a9f4090d381194b9f1bcd3838f3dcc0bafb0c7769a0877e05aa00 garden_commit=e2fe391d6c' ]] ||
  fail 'policy-plan parent binding diverged'
[[ "$(grep -c '^FAMILY ' "$BUNDLE" || true)" == 12 ]] ||
  fail 'policy-plan family count diverged'
[[ "$(grep -c '^CASE ' "$BUNDLE" || true)" == 14 ]] ||
  fail 'policy-plan authority-case count diverged'
[[ "$(grep -c ' treatment_kernel=REFUSED sabotage_kernel=CROSSED ' "$BUNDLE" || true)" == 12 ]] ||
  fail 'treatment/sabotage plan is incomplete'

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
    fail "Sounio action 9025 disagreed with Sounio plan case $label: $actual"
  if [[ "$label" == complete ]]; then complete_actual="$actual"; fi
  if [[ "$label" == current ]]; then current_actual="$actual"; fi
  case_count=$((case_count + 1))
done < "$BUNDLE"
[[ "$case_count" == 14 ]] || fail 'not every Sounio policy case was judged'
[[ "$complete_actual" == SOUNIO_EFFECT_CLOSURE_ALLOW* ]] ||
  fail 'complete hypothetical frame did not reach the frozen Sounio allow'
[[ "$current_actual" == SOUNIO_EFFECT_CLOSURE_DENY* ]] ||
  fail 'current material frame did not remain closed'

boundary='BOUNDARY complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false'
grep -Fxq "$boundary" "$BUNDLE" || fail 'policy-plan evidence boundary drifted'
grep -Fxq 'complete_effects=false' "$PROCESS_MANIFEST" ||
  fail 'frozen ProcessWitness parent no longer records closed effects'
grep -Fxq 'material_coverage=false' "$EFFECT_MANIFEST" ||
  fail 'frozen action-9025 parent no longer records closed material coverage'

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
  fail 'Sounio policy-plan executable has a prohibited runtime dependency'
fi

printf 'sounio-loom-process-witness-effect-policy-plan-selftest: PASS producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 families=12 treatments=12 sabotages=12 authority_cases=14 complete=ALLOW current=DENY447 missing_known=DENY447x11 missing_unknown=DENY452 python_control=refused python_executed=false deterministic=true shell_expected_results=false source_sha256=%s executable_sha256=%s bundle_sha256=%s complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_main.sio" | cut -d ' ' -f 1)" \
  "$(sha256sum "$PLAN_ONE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$BUNDLE" | cut -d ' ' -f 1)"
