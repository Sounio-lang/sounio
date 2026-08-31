#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GARDEN="$ROOT_DIR/tools/loom/GARDEN_PRODUCT_DYNAMIC_USER_EXEC_ATTACHMENT_V1.md"
CONTRACT="$ROOT_DIR/tools/loom/PRODUCT_DYNAMIC_USER_GARDEN_REMAINING_V1.md"
COUNTEREXAMPLE="$ROOT_DIR/scripts/ci/sounio_loom_product_dynamic_user_exec_counterexample_selftest.sh"
COHERENCE="$ROOT_DIR/scripts/ci/sounio_loom_handshake_exec_cell_payload_coherence_selftest.sh"
LANE="$ROOT_DIR/tools/loom/product_dynamic_user_lane_cell_host_canary.runtime.v1"
EXEC_CELL="$ROOT_DIR/tools/loom/product_exec_cell_host_canary.runtime.v1"
INGRESS="$ROOT_DIR/tools/loom/product_exec_ingress_dark.runtime.v1"

fail() {
  printf 'sounio-loom-product-dynamic-user-garden-remaining-selftest: FAIL: %s\n' \
    "$*" >&2
  exit 1
}

field() {
  local path="$1" key="$2" count line
  count="$(grep -c "^${key}=" "$path" || true)"
  [[ "$count" == 1 ]] || fail "${path#$ROOT_DIR/} field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$path")"
  printf '%s' "${line#*=}"
}

expect_field() {
  local path="$1" key="$2" expected="$3" actual
  actual="$(field "$path" "$key")"
  [[ "$actual" == "$expected" ]] ||
    fail "${path#$ROOT_DIR/} $key expected=$expected actual=$actual"
}

file_hash() {
  local path="$1" sum
  [[ -f "$path" && ! -L "$path" ]] || fail "${path#$ROOT_DIR/} is missing or linked"
  sum="$(sha256sum "$path")"
  printf '%s' "${sum%% *}"
}

pins_live() {
  local manifest="$1" path_key="$2" hash_key="$3" relative expected actual
  relative="$(field "$manifest" "$path_key")"
  expected="$(field "$manifest" "$hash_key")"
  actual="$(file_hash "$ROOT_DIR/$relative")"
  [[ "$actual" == "$expected" ]]
}

for path in "$GARDEN" "$CONTRACT" "$COUNTEREXAMPLE" "$COHERENCE" "$LANE" \
  "$EXEC_CELL" "$INGRESS"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "required parent is absent: ${path#$ROOT_DIR/}"
done

grep -Fq 'Replace the same-UID OCaml `fork/exec` material path' "$GARDEN" ||
  fail 'Garden gate 4 text drifted'
grep -Fq 'Make descriptor absence fail closed for product execution tools.' \
  "$GARDEN" || fail 'Garden gate 5 text drifted'

expect_field "$INGRESS" required_mode_default false
expect_field "$INGRESS" production_activation false
expect_field "$INGRESS" exec_attached false
expect_field "$INGRESS" parity_open false
expect_field "$INGRESS" claim_ready false
expect_field "$INGRESS" recycle_open false

expect_field "$LANE" stage MATERIAL_CANARY_FROZEN
expect_field "$LANE" product_lane_cell_canary true
expect_field "$LANE" fleet_lane_cell_attached false
expect_field "$LANE" exec_cell_attached false
expect_field "$LANE" production_activation false
expect_field "$LANE" parity_open false
expect_field "$LANE" claim_ready false

expect_field "$EXEC_CELL" stage MATERIAL_EXEC_CELL_CANARY_FROZEN
expect_field "$EXEC_CELL" exec_cell_attached true
expect_field "$EXEC_CELL" test_only true
expect_field "$EXEC_CELL" production_activation false
expect_field "$EXEC_CELL" exec_attached false
expect_field "$EXEC_CELL" parity_open false
expect_field "$EXEC_CELL" claim_ready false

LANE_FREEZE_LIVE=true
if ! pins_live "$LANE" ingress_source_path ingress_source_sha256 ||
   ! pins_live "$LANE" host_canary_source_path host_canary_source_sha256 ||
   ! pins_live "$LANE" broker_source_path broker_source_sha256; then
  LANE_FREEZE_LIVE=false
fi

EXEC_CELL_FREEZE_LIVE=true
if ! pins_live "$EXEC_CELL" composition_source_path composition_source_sha256 ||
   ! pins_live "$EXEC_CELL" ingress_source_path ingress_source_sha256 ||
   ! pins_live "$EXEC_CELL" broker_source_path broker_source_sha256; then
  EXEC_CELL_FREEZE_LIVE=false
fi

COUNTER_RESULT="$(bash "$COUNTEREXAMPLE")"
[[ "$COUNTER_RESULT" == sounio-loom-product-dynamic-user-exec-counterexample-selftest:\ PASS* ]] ||
  fail 'counterexample failed; do not infer that Garden gate 4 closed'
[[ "$COUNTER_RESULT" == *' child_execution=fork+execve '* ]] ||
  fail 'counterexample no longer reports same-UID fork/exec'
[[ "$COUNTER_RESULT" == *' probe=named-let-broker_command_kernel+supervise_child '* ]] ||
  fail 'counterexample is not the named-let probe'

COHERENCE_RESULT="$(bash "$COHERENCE")"
[[ "$COHERENCE_RESULT" == sounio-loom-handshake-exec-cell-payload-coherence-selftest:\ PASS* ]] ||
  fail 'handshake ExecCell payload identity failed'
[[ "$COHERENCE_RESULT" == *' required_mode_default=false '* ]] ||
  fail 'coherence gate no longer pins required_mode_default=false'

[[ "$(field "$INGRESS" required_mode_default)" == false ]] ||
  fail 'required_mode_default flipped while Garden gate 4 is still open'

printf 'sounio-loom-product-dynamic-user-garden-remaining-selftest: PASS semantic_authority=Sounio producer=Bash role=CENSUS_ONLY garden=GARDEN_PRODUCT_DYNAMIC_USER_EXEC_ATTACHMENT_V1 gate1_counterexample_live=true gate2_lane_canary_recorded=true gate2_lane_freeze_live=%s gate3_exec_cell_canary_recorded=true gate3_exec_cell_test_only=true gate3_exec_cell_freeze_live=%s gate4_fork_exec_replaced=false gate5_descriptor_fail_closed=false gate6_crash_recovery=false gate7_immutable_receipt=false gate8_fleet_rollout=false remaining=4+5+6+7+8 required_mode_default=false fleet_lane_cell_attached=false canary_exec_cell_attached=true canary_test_only=true product_exec_attached=false material_execution=false production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false python_executed=false rust_executed=false next=replace-same-uid-fork-exec-then-fail-closed-descriptor\n' \
  "$LANE_FREEZE_LIVE" "$EXEC_CELL_FREEZE_LIVE"
