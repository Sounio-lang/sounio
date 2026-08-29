#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-policy-v11-selftest.XXXXXX")"
PLAN_ONE="$TEST_ROOT/plan-one"
PLAN_TWO="$TEST_ROOT/plan-two"
AUTHORITY="$TEST_ROOT/action-9025-authority"
BUNDLE="$TEST_ROOT/policy-plan-v11"

cleanup() { rm -rf "$TEST_ROOT"; }
trap cleanup EXIT

fail() {
  printf 'sounio-loom-process-witness-effect-policy-plan-v11-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

for output in "$PLAN_ONE" "$PLAN_TWO"; do
  SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V11_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_process_witness_effect_policy_plan_v11.sh" \
      >/dev/null
done
cmp "$PLAN_ONE" "$PLAN_TWO" || fail 'two source-fresh V11 builds differ'
[[ "$(stat -c '%a' "$PLAN_ONE")" == 755 && ! -u "$PLAN_ONE" && ! -g "$PLAN_ONE" ]] ||
  fail 'V11 executable mode is unsafe'
SOUNIO_LOOM_EFFECT_CLOSURE_OUTPUT="$AUTHORITY" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_effect_closure_authority.sh" >/dev/null

"$PLAN_ONE" > "$BUNDLE"
[[ "$(wc -l < "$BUNDLE")" == 113 ]] || fail 'V11 bundle line count diverged'
[[ "$(grep -c '^SYSCALL ' "$BUNDLE" || true)" == 4 ]] || fail 'V11 syscall count diverged'
[[ "$(grep -c '^MECHANISMS ' "$BUNDLE" || true)" == 12 ]] ||
  fail 'V11 mechanism-family count diverged'
[[ "$(grep -c '^PROBE ' "$BUNDLE" || true)" == 13 ]] || fail 'V11 probe count diverged'
[[ "$(grep -c '^VERTEX ' "$BUNDLE" || true)" == 40 ]] || fail 'V11 vertex count diverged'
[[ "$(grep -c '^MINCUT ' "$BUNDLE" || true)" == 13 ]] || fail 'V11 mincut count diverged'
[[ "$(grep -c '^CASE ' "$BUNDLE" || true)" == 14 ]] || fail 'V11 action-case count diverged'
[[ "$(grep -c '^BOOTSTRAP_CASE ' "$BUNDLE" || true)" == 8 ]] ||
  fail 'V11 bootstrap-case count diverged'

grep -Fxq 'ROOT_SCHEMA paths=/loom/effect-cell+/loom/payload+/loom/payload.freeze.v1+/loom/effect-policy-v11.freeze.v1+/dev/null+/run/systemd/incoming+/sys+/var/tmp empty_readonly=/tmp+/var/tmp proc_treatment=CAPSULE_EMPTY_BIND' "$BUNDLE" ||
  fail 'V11 typed root schema drifted'
if grep -Fq 'proc_treatment=absent' "$BUNDLE"; then
  fail 'V11 retained the falsified path-only proc statement'
fi
grep -Fxq 'OBSERVATION_TYPES values=REFUSED_BEFORE_EFFECT+CROSSED_NAMED_RULE+EFFECT_COMPLETED+EXPERIMENT_UNAVAILABLE crossed_named_rule_counts_as_completion=false unavailable_counts_as_coverage=false' "$BUNDLE" ||
  fail 'V11 typed observation lattice drifted'
grep -Fxq 'HASH_BINDING fields=invariant_sha256+delta_sha256+witness_sha256 invariant_scope=probe_cube delta_scope=mechanism_bits witness_scope=typed_observation peer_rule=identical_invariant+permitted_delta' "$BUNDLE" ||
  fail 'V11 causal triple-hash binding drifted'
grep -Fxq 'MONOTONICITY refusal_superset=true completion_subset=true violation=DENY457_nonmonotone-material-effect mincut_rule=refusing_vertex+every_single_active_bit_removal_completes' "$BUNDLE" ||
  fail 'V11 monotonicity contract drifted'

for id in $(seq 1 12); do
  [[ "$(grep -c "^MECHANISMS family=${id} " "$BUNDLE" || true)" == 1 ]] ||
    fail "V11 family $id is absent or duplicated"
  [[ "$(grep -c "^CASE label=missing_${id} " "$BUNDLE" || true)" == 1 ]] ||
    fail "V11 missing-family action case $id is absent or duplicated"
done

vertex_keys="$TEST_ROOT/vertex-keys"
sed -n 's/^VERTEX family=\([^ ]*\) probe=\([^ ]*\) bits=\([^ ]*\) .*/\1 \2 \3/p' \
  "$BUNDLE" > "$vertex_keys"
[[ "$(sort -u "$vertex_keys" | wc -l)" == 40 ]] ||
  fail 'V11 contains a duplicate family/probe/bit vertex'

while IFS= read -r probe_line; do
  body="${probe_line#PROBE family=}"
  family="${body%% probe=*}"
  body="${body#* probe=}"
  probe="${body%% dimensions=*}"
  body="${body#* dimensions=}"
  dimensions="${body%% vertices=*}"
  vertices="${body#* vertices=}"
  family="${family% }"
  probe="${probe% }"
  dimensions="${dimensions% }"
  [[ "$(grep -c "^VERTEX family=${family} probe=${probe} " "$BUNDLE" || true)" == "$vertices" ]] ||
    fail "V11 probe ${family}/${probe} vertex count diverged"
  [[ "$(grep -c "^MINCUT family=${family} probe=${probe} " "$BUNDLE" || true)" == 1 ]] ||
    fail "V11 probe ${family}/${probe} mincut is absent or duplicated"
  if [[ "$dimensions" == 1 ]]; then
    grep -Fq "VERTEX family=${family} probe=${probe} bits=1 expected=REFUSED_BEFORE_EFFECT " "$BUNDLE" ||
      fail "V11 probe ${family}/${probe} lacks its treatment refusal"
    grep -Fq "VERTEX family=${family} probe=${probe} bits=0 expected=EFFECT_COMPLETED " "$BUNDLE" ||
      fail "V11 probe ${family}/${probe} lacks its open completion"
  else
    [[ "$dimensions" == 2 ]] || fail "V11 probe ${family}/${probe} has an unsupported dimension"
    grep -Fq "VERTEX family=${family} probe=${probe} bits=11 expected=REFUSED_BEFORE_EFFECT " "$BUNDLE" ||
      fail "V11 probe ${family}/${probe} lacks its full-treatment refusal"
    grep -Fq "VERTEX family=${family} probe=${probe} bits=00 expected=EFFECT_COMPLETED " "$BUNDLE" ||
      fail "V11 probe ${family}/${probe} lacks its open completion"
  fi
done < <(grep '^PROBE ' "$BUNDLE")

if grep '^VERTEX .* expected=EFFECT_COMPLETED ' "$BUNDLE" | grep -Fq 'witness_kind=NONE'; then
  fail 'V11 permits an effect completion without a positive witness kind'
fi
if grep '^VERTEX .* expected=REFUSED_BEFORE_EFFECT ' "$BUNDLE" | grep -Fvq 'witness_kind=NONE'; then
  fail 'V11 refusal vertex falsely carries an effect-completion witness kind'
fi
[[ "$(grep -c '^VERTEX .* invariant_sha256=REQUIRED .* delta_sha256=REQUIRED witness_sha256=REQUIRED$' "$BUNDLE" || true)" == 40 ]] ||
  fail 'V11 does not triple-bind every vertex'

grep -Fxq 'MINCUT family=1 probe=repeat_exact_exec cuts=10 derivation=ORDERED_VERTEX_TABLE monotone_required=true' "$BUNDLE" ||
  fail 'V11 repeat-exec CLOEXEC cut drifted'
grep -Fxq 'MINCUT family=1 probe=first_wrong_flags_exec cuts=01 derivation=ORDERED_VERTEX_TABLE monotone_required=true' "$BUNDLE" ||
  fail 'V11 wrong-flags argument-filter cut drifted'
for family in 3 7 8 10 11; do
  grep -q "^MINCUT family=${family} .* cuts=10+01 derivation=ORDERED_VERTEX_TABLE monotone_required=true$" "$BUNDLE" ||
    fail "V11 redundant-defense cuts drifted for family $family"
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
  [[ "$actual" == "$expected" ]] || fail "action 9025 disagreed with V11 case $label"
  if [[ "$label" == complete ]]; then complete_actual="$actual"; fi
  if [[ "$label" == current ]]; then current_actual="$actual"; fi
  case_count=$((case_count + 1))
done < "$BUNDLE"
[[ "$case_count" == 14 && "$complete_actual" == SOUNIO_EFFECT_CLOSURE_ALLOW* &&
   "$current_actual" == SOUNIO_EFFECT_CLOSURE_DENY* ]] ||
  fail 'V11 action-9025 decision matrix is incomplete'

grep -Fxq 'PRESERVED_V10 root_treatment=true bootstrap_sabotage=true bootstrap_negative_controls=7 typed_proc_sabotages=4 material_coverage=false' "$BUNDLE" ||
  fail 'V11 failed to preserve the bounded V10 positive result'
boundary='BOUNDARY v10_frozen=true v11_required_for_native=true semantics_frozen=false parity_open=false material_hypercube=false material_coverage=false complete_effects=false material_execution=false action_9025_judged=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false claim_ready=false'
grep -Fxq "$boundary" "$BUNDLE" || fail 'V11 evidence boundary drifted'

dependencies="$(ldd "$PLAN_ONE" 2>&1 || true)"
if printf '%s\n' "$dependencies" | grep -Eqi 'python|rust'; then
  fail 'Sounio V11 executable has a prohibited dependency'
fi

printf 'sounio-loom-process-witness-effect-policy-plan-v11-selftest: PASS producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE families=12 probes=13 mechanism_dimensions=18 vertices=40 mincuts=13 triple_hash_binding=true typed_observations=true crossed_is_not_completion=true unavailable_is_not_coverage=true full_treatments=REFUSED open_vertices=EFFECT_COMPLETED action_cases=14 complete=ALLOW current=DENY447 proc_treatment=CAPSULE_EMPTY_BIND legacy_proc_absence=false deterministic=true shell_expected_results=false source_sha256=%s executable_sha256=%s bundle_sha256=%s semantics_frozen=false parity_open=false material_hypercube=false material_coverage=false complete_effects=false material_execution=false action_9025_judged=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false claim_ready=false\n' \
  "$(sha256sum "$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v11_main.sio" | cut -d ' ' -f 1)" \
  "$(sha256sum "$PLAN_ONE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$BUNDLE" | cut -d ' ' -f 1)"
