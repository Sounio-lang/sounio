#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-causal-workflow.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT

fail() {
  printf 'sounio-loom-causal-workflow-kernel-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

RUNTIME_ONE="$TEST_ROOT/runtime-one"
RUNTIME_TWO="$TEST_ROOT/runtime-two"
for output in "$RUNTIME_ONE" "$RUNTIME_TWO"; do
  SOUNIO_LOOM_CAUSAL_WORKFLOW_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_causal_workflow_kernel_fixture.sh" >/dev/null
done
cmp "$RUNTIME_ONE" "$RUNTIME_TWO" || fail 'two Sounio builds differ'

FINAL='9037 4611686016979304447 4503591041609465'
RECOVERY='9037 144097867340972031 4065985942257721'
PREDECESSOR_MISMATCH='9037 144097867340972031 4065968762388537'
FIELDS='LOOM_CAUSAL_WORKFLOW_KERNEL_FIELDS_V1
LOOM_CAUSAL_WORKFLOW/1
COMPILE>RUN_EXACT>ATTEST
tests/verify-ir/call_b.sio
899d05ffe60528a6b71871e24fa0d1bc105cd033b7ae2c5a0a6d2bb808cdcad9
81c4929823460a807762abd4a878ca9adbd053313e7ac6662fa489456269a4a3
eff2ac0ef28b34d6cc4f008cfb08a30ba18a0874c8654c06a3c62ec2f48a249c
exit_code=0
stdout_sha256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
stderr_sha256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
run_ticket_is_bearer=false
run_ticket_is_execution_authority=false
launch_authority=action-9030
exactly_once_scope=live-HostGuardian-generation
guardian_host_or_store_loss=fail-closed'

final_result="$(printf '%s\n' "$FINAL" | "$RUNTIME_ONE")"
[[ "$final_result" == "SOUNIO_CAUSAL_WORKFLOW ADVANCE semantic_authority=Sounio action=9037
$FIELDS" ]] || fail "final workflow diverged: $final_result"
recovery_result="$(printf '%s\n' "$RECOVERY" | "$RUNTIME_ONE")"
[[ "$recovery_result" == "SOUNIO_CAUSAL_WORKFLOW RECOVER semantic_authority=Sounio action=9037
$FIELDS" ]] || fail "recovery workflow diverged: $recovery_result"

declare -A CONTROLS=(
  [graph]='9037 4611686016979304439 4503591041609465|DENY580'
  [parent]='9037 4611686016978780159 4503591041609465|DENY581'
  [edge]='9037 4611686016962527231 4503591041609465|DENY582'
  [lineage]='9037 4611686016979304447 4433222297431801|DENY583'
  [recompile]='9037 144097867340972031 4065985942257725|DENY584'
  [ticket]='9037 4611686016979304447 4503591041607417|DENY585'
  [replay]='9037 4611686016979304447 4503591041617657|DENY586'
  [successor]='9037 144097867340972031 4065985875148857|DENY587'
  [terminal]='9037 4611686016979304447 4503591041478393|DENY588'
  [predecessor]="$PREDECESSOR_MISMATCH|DENY589"
)
for label in graph parent edge lineage recompile ticket replay successor terminal predecessor; do
  IFS='|' read -r frame expected <<< "${CONTROLS[$label]}"
  set +e
  observed="$(printf '%s\n' "$frame" | "$RUNTIME_ONE")"
  code=$?
  set -e
  [[ $code -eq 42 && "$observed" == \
     "SOUNIO_CAUSAL_WORKFLOW $expected semantic_authority=Sounio action=9037" ]] ||
    fail "$label control diverged: $observed"
done

set +e
malformed_result="$(printf '9037 3\n' | "$RUNTIME_ONE")"
malformed_code=$?
set -e
[[ $malformed_code -eq 42 && "$malformed_result" == \
   'SOUNIO_CAUSAL_WORKFLOW DENY424 reason=malformed-frame semantic_authority=Sounio action=9037' ]] ||
  fail "malformed-frame control diverged: $malformed_result"

MUTANT_MODULE="$TEST_ROOT/mutant.sio"
sed 's/if observation.predecessor_receipt_equal != 1 {/if false {/' \
  "$ROOT_DIR/stdlib/coordination/loom_causal_workflow_kernel_authority.sio" > "$MUTANT_MODULE"
[[ "$(cmp -l "$ROOT_DIR/stdlib/coordination/loom_causal_workflow_kernel_authority.sio" "$MUTANT_MODULE" 2>/dev/null | wc -l)" -gt 0 ]] ||
  fail 'causal mutation did not change the Sounio source'
MUTANT_RUNTIME="$TEST_ROOT/mutant-runtime"
SOUNIO_LOOM_CAUSAL_WORKFLOW_MODULE="$MUTANT_MODULE" \
SOUNIO_LOOM_CAUSAL_WORKFLOW_OUTPUT="$MUTANT_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_causal_workflow_kernel_fixture.sh" >/dev/null || true
[[ -x "$MUTANT_RUNTIME" ]] || fail 'mutant Sounio runtime did not build'
mutant_result="$(printf '%s\n' "$PREDECESSOR_MISMATCH" | "$MUTANT_RUNTIME")"
[[ "$mutant_result" == "SOUNIO_CAUSAL_WORKFLOW RECOVER semantic_authority=Sounio action=9037
$FIELDS" ]] ||
  fail "load-bearing mutation did not admit the unchanged witness: $mutant_result"

ORACLE_EXECUTED="$TEST_ROOT/oracle-executed"
for name in python python3 rustc cargo; do
  printf '#!/bin/sh\nprintf prohibited > %s\n' "$ORACLE_EXECUTED" > "$TEST_ROOT/$name"
  chmod 0755 "$TEST_ROOT/$name"
done
PATH="$TEST_ROOT:$PATH" printf '%s\n' "$FINAL" | "$RUNTIME_ONE" >/dev/null
[[ ! -e "$ORACLE_EXECUTED" ]] || fail 'a prohibited oracle executed'
DEPENDENCIES="$(ldd "$RUNTIME_ONE" 2>&1 || true)"
printf '%s\n' "$DEPENDENCIES" | grep -Eqi 'python|rust' &&
  fail 'Sounio runtime has a prohibited dependency'

printf 'sounio-loom-causal-workflow-kernel-selftest: PASS semantic_authority=Sounio action=9037 stage=SOUNIO_EXECUTABLE states=12 final=ADVANCE recovery=RECOVER graph=DENY580 parent=DENY581 edge=DENY582 lineage=DENY583 recompile=DENY584 ticket=DENY585 replay=DENY586 successor=DENY587 terminal=DENY588 predecessor=DENY589 malformed=DENY424 causal_sabotage=PASS source=tests/verify-ir/call_b.sio expected_exit=0 expected_stdout=empty expected_stderr=empty launch_authority=action-9030 run_ticket_bearer=false run_ticket_execution_authority=false exactly_once_scope=live-HostGuardian-generation guardian_loss=fail-closed expected_results_encoded_in_material_layer=false python_executed=false rust_executed=false runtime_dependencies=clean source_sha256=%s executable_sha256=%s ocaml_journal_attached=false hostguardian_attachment=false controller_loss_measured=false pod_loss_measured=false dynamic_user_workflow_attached=false material_execution=false production_activation=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$ROOT_DIR/stdlib/coordination/loom_causal_workflow_kernel_authority.sio" | cut -d ' ' -f 1)" \
  "$(sha256sum "$RUNTIME_ONE" | cut -d ' ' -f 1)"
