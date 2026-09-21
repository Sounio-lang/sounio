#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-host-durable-lane.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT

fail() {
  printf 'sounio-loom-host-durable-lane-supervisor-fixture-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

RUNTIME_ONE="$TEST_ROOT/runtime-one"
RUNTIME_TWO="$TEST_ROOT/runtime-two"
for output in "$RUNTIME_ONE" "$RUNTIME_TWO"; do
  SOUNIO_LOOM_HOST_DURABLE_LANE_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_host_durable_lane_supervisor_fixture.sh" >/dev/null
done
cmp "$RUNTIME_ONE" "$RUNTIME_TWO" || fail 'two Sounio builds differ'

same='9032 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 1 1 1'
lineage='9032 3 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1'
start_mismatch='9032 3 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 1 1 1'
loss_without_lineage='9032 3 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 1 1 1 1'

same_result="$(printf '%s\n' "$same" | "$RUNTIME_ONE")"
lineage_result="$(printf '%s\n' "$lineage" | "$RUNTIME_ONE")"
set +e
start_result="$(printf '%s\n' "$start_mismatch" | "$RUNTIME_ONE")"
start_code=$?
loss_result="$(printf '%s\n' "$loss_without_lineage" | "$RUNTIME_ONE")"
loss_code=$?
set -e
[[ "$same_result" == 'SOUNIO_HOST_DURABLE_LANE SAME_PHYSICAL_REATTACH '* ]] ||
  fail "same-physical fixture diverged: $same_result"
[[ "$lineage_result" == 'SOUNIO_HOST_DURABLE_LANE LINEAGE_RESURRECTION '* ]] ||
  fail "lineage fixture diverged: $lineage_result"
[[ $start_code -eq 42 && "$start_result" == 'SOUNIO_HOST_DURABLE_LANE DENY526 '* ]] ||
  fail "Guardian start-tick control diverged: $start_result"
[[ $loss_code -eq 42 && "$loss_result" == 'SOUNIO_HOST_DURABLE_LANE DENY529 '* ]] ||
  fail "unproven Guardian-loss control diverged: $loss_result"

MUTANT_MODULE="$TEST_ROOT/mutant.sio"
sed 's/observation\.guardian_start_equal != 1 ||/false ||/' \
  "$ROOT_DIR/stdlib/coordination/loom_host_durable_lane_supervisor.sio" > "$MUTANT_MODULE"
[[ "$(cmp -l "$ROOT_DIR/stdlib/coordination/loom_host_durable_lane_supervisor.sio" "$MUTANT_MODULE" 2>/dev/null | wc -l)" -gt 0 ]] ||
  fail 'causal mutation did not change the Sounio source'
MUTANT_RUNTIME="$TEST_ROOT/mutant-runtime"
SOUNIO_LOOM_HOST_DURABLE_LANE_MODULE="$MUTANT_MODULE" \
SOUNIO_LOOM_HOST_DURABLE_LANE_OUTPUT="$MUTANT_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_host_durable_lane_supervisor_fixture.sh" >/dev/null || true
[[ -x "$MUTANT_RUNTIME" ]] || fail 'mutant Sounio runtime did not build'
mutant_result="$(printf '%s\n' "$start_mismatch" | "$MUTANT_RUNTIME")"
[[ "$mutant_result" == 'SOUNIO_HOST_DURABLE_LANE SAME_PHYSICAL_REATTACH '* ]] ||
  fail "load-bearing mutation did not admit the unchanged witness: $mutant_result"

oracle_executed="$TEST_ROOT/oracle-executed"
for name in python3 rustc; do
  printf '#!/bin/sh\nprintf prohibited > %s\n' "$oracle_executed" > "$TEST_ROOT/$name"
  chmod 0755 "$TEST_ROOT/$name"
done
PATH="$TEST_ROOT:$PATH" printf '%s\n' "$same" | "$RUNTIME_ONE" >/dev/null
[[ ! -e "$oracle_executed" ]] || fail 'a prohibited oracle executed'
dependencies="$(ldd "$RUNTIME_ONE" 2>&1 || true)"
printf '%s\n' "$dependencies" | grep -Eqi 'python|rust' &&
  fail 'Sounio runtime has a prohibited dependency'

printf 'sounio-loom-host-durable-lane-supervisor-fixture-selftest: PASS semantic_authority=Sounio action=9032 stage=SOUNIO_EXECUTABLE cases=10 same_physical=SAME_PHYSICAL_REATTACH lineage=LINEAGE_RESURRECTION guardian_start_mismatch=DENY526 guardian_loss_without_lineage=DENY529 causal_sabotage=PASS expected_results_encoded_in_material_layer=false python_executed=false rust_executed=false runtime_dependencies=clean source_sha256=%s executable_sha256=%s material_execution=false production_activation=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$ROOT_DIR/stdlib/coordination/loom_host_durable_lane_supervisor.sio" | cut -d ' ' -f 1)" \
  "$(sha256sum "$RUNTIME_ONE" | cut -d ' ' -f 1)"
