#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-causal-mid-exec-selftest.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail() {
  printf 'sounio-loom-causal-workflow-mid-exec-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

for ordinal in one two; do
  SOUNIO_LOOM_CAUSAL_MID_EXEC_OUTPUT="$WORK/runtime-$ordinal" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_causal_workflow_mid_exec_fixture.sh" >/dev/null
done
RUNTIME_ONE="$WORK/runtime-one"
cmp "$RUNTIME_ONE" "$WORK/runtime-two" || fail 'two Sounio builds differ'

release="$(printf '9037 1 3071 1045\n' | "$RUNTIME_ONE")"
claim="$(printf '9037 2 4095 1365\n' | "$RUNTIME_ONE")"
[[ "${release%%$'\n'*}" == 'SOUNIO_CAUSAL_WORKFLOW_MID_EXEC RELEASE semantic_authority=Sounio action=9037 subordinate_contract=mid-exec-v1' && \
   "$release" == *$'decision_stage=RELEASE_ADMISSION\ncompletion_observed=0\nresult_count=0\nattestation_count=0'* ]] ||
  fail "release witness diverged: $release"
[[ "${claim%%$'\n'*}" == 'SOUNIO_CAUSAL_WORKFLOW_MID_EXEC CONTINUITY semantic_authority=Sounio action=9037 subordinate_contract=mid-exec-v1' && \
   "$claim" == *$'decision_stage=CLAIM_CONTINUITY\ncompletion_observed=1\nresult_count=1\nattestation_count=1'* && \
   "$claim" == *$'compile_count=1\nticket_count=1\nlaunch_count=1'* ]] ||
  fail "claim witness diverged: $claim"

declare -A CONTROLS=(
  [release_replacement_pid]='1 3063 1045|DENY593'
  [claim_replacement_pid]='2 4087 1365|DENY593'
  [claim_replacement_invocation]='2 4091 1365|DENY592'
  [claim_replacement_start_tick]='2 4079 1365|DENY594'
  [release_unbound_successor]='1 2943 1045|DENY597'
  [release_duplicate_launch]='1 3071 21|DENY599'
  [claim_duplicate_launch]='2 4095 341|DENY599'
  [claim_count_drift]='2 4095 1349|DENY599'
  [claim_barrier_nonce]='2 2047 1365|DENY600'
  [release_future_completion]='1 4095 1045|DENY601'
  [claim_completion_absent]='2 3071 1365|DENY598'
)
for label in release_replacement_pid claim_replacement_pid claim_replacement_invocation \
  claim_replacement_start_tick release_unbound_successor release_duplicate_launch \
  claim_duplicate_launch claim_count_drift claim_barrier_nonce \
  release_future_completion claim_completion_absent; do
  IFS='|' read -r frame expected <<< "${CONTROLS[$label]}"
  set +e
  observed="$(printf '9037 %s\n' "$frame" | "$RUNTIME_ONE")"
  code=$?
  set -e
  [[ $code -eq 42 && "$observed" == "SOUNIO_CAUSAL_WORKFLOW_MID_EXEC $expected semantic_authority=Sounio action=9037 subordinate_contract=mid-exec-v1" ]] ||
    fail "$label control diverged: $observed"
done

release_mutant_module="$WORK/release-mutant.sio"
sed 's/if observation.material_pid_equal != 1 { return 593 }/if false { return 593 }/' \
  "$ROOT_DIR/stdlib/coordination/loom_causal_workflow_mid_exec_authority.sio" > "$release_mutant_module"
release_mutant_runtime="$WORK/release-mutant-runtime"
SOUNIO_LOOM_CAUSAL_MID_EXEC_MODULE="$release_mutant_module" \
SOUNIO_LOOM_CAUSAL_MID_EXEC_OUTPUT="$release_mutant_runtime" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_causal_workflow_mid_exec_fixture.sh" >/dev/null 2>&1 || true
[[ -x "$release_mutant_runtime" ]] || fail 'load-bearing release PID mutant did not build'
release_mutant="$(printf '9037 1 3063 1045\n' | "$release_mutant_runtime")"
[[ "${release_mutant%%$'\n'*}" == 'SOUNIO_CAUSAL_WORKFLOW_MID_EXEC RELEASE semantic_authority=Sounio action=9037 subordinate_contract=mid-exec-v1' ]] ||
  fail 'load-bearing release PID mutant did not admit replacement witness'

claim_mutant_module="$WORK/claim-mutant.sio"
sed 's/if observation.barrier_nonce_equal != 1 { return 600 }/if false { return 600 }/' \
  "$ROOT_DIR/stdlib/coordination/loom_causal_workflow_mid_exec_authority.sio" > "$claim_mutant_module"
claim_mutant_runtime="$WORK/claim-mutant-runtime"
SOUNIO_LOOM_CAUSAL_MID_EXEC_MODULE="$claim_mutant_module" \
SOUNIO_LOOM_CAUSAL_MID_EXEC_OUTPUT="$claim_mutant_runtime" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_causal_workflow_mid_exec_fixture.sh" >/dev/null 2>&1 || true
[[ -x "$claim_mutant_runtime" ]] || fail 'load-bearing claim barrier-nonce mutant did not build'
claim_mutant="$(printf '9037 2 2047 1365\n' | "$claim_mutant_runtime")"
[[ "${claim_mutant%%$'\n'*}" == 'SOUNIO_CAUSAL_WORKFLOW_MID_EXEC CONTINUITY semantic_authority=Sounio action=9037 subordinate_contract=mid-exec-v1' ]] ||
  fail 'load-bearing claim barrier-nonce mutant did not admit stale barrier witness'

oracle_executed="$WORK/oracle-executed"
for name in python python3 rustc cargo node; do
  printf '#!/bin/sh\nprintf prohibited > %s\n' "$oracle_executed" > "$WORK/$name"
  chmod 0755 "$WORK/$name"
done
printf '9037 2 4095 1365\n' | env PATH="$WORK:$PATH" "$RUNTIME_ONE" >/dev/null
[[ ! -e "$oracle_executed" ]] || fail 'a prohibited oracle executed'
printf 'sounio-loom-causal-workflow-mid-exec-selftest: PASS semantic_authority=Sounio action=9037 subordinate_contract=mid-exec-v1 release=ADMIT claim=CONTINUITY release_replacement_pid=DENY593 claim_replacement_pid=DENY593 replacement_invocation=DENY592 replacement_start_tick=DENY594 unbound_successor=DENY597 duplicate_launch=DENY599 count_drift=DENY599 barrier_nonce=DENY600 release_future_completion=DENY601 claim_completion_absent=DENY598 causal_sabotage=PASS exact_counts=compile+ticket+launch+result+attestation:1 python_executed=false rust_executed=false node_executed=false material_execution=false pod_loss_measured=false production_activation=false parity_open=false claim_ready=false source_sha256=%s executable_sha256=%s\n' \
  "$(sha256sum "$ROOT_DIR/stdlib/coordination/loom_causal_workflow_mid_exec_authority.sio" | cut -d ' ' -f 1)" \
  "$(sha256sum "$RUNTIME_ONE" | cut -d ' ' -f 1)"
