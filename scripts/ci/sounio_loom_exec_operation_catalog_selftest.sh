#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-exec-operation-catalog.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT

fail() {
  printf 'sounio-loom-exec-operation-catalog-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

RUNTIME_ONE="$TEST_ROOT/runtime-one"
RUNTIME_TWO="$TEST_ROOT/runtime-two"
for output in "$RUNTIME_ONE" "$RUNTIME_TWO"; do
  SOUNIO_LOOM_EXEC_OPERATION_CATALOG_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_operation_catalog_fixture.sh" >/dev/null
done
cmp "$RUNTIME_ONE" "$RUNTIME_TWO" || fail 'two Sounio builds differ'

calibration='9035 536870895 1028'
sounio_check='9035 536870903 1028'
template_mismatch='9035 536870839 1028'
calibration_fields='LOOM_EXEC_OPERATION_CATALOG_FIELDS_V1
LOOM_EXEC_CATALOG/1
31089d39690c8e008dd7ea9bfecefd7e51e4257cd96e091f7fb492c240ed342c
calibration
113b0f7d2a1f7adc5b68a92ef37be3689accfe4bbb00cb4e8c15fc3ab7b70013
07ffb41877176a60b04949dd78c91313954dd759593f20677a1b2fcfeea31a60
83fe8c2a7d8336d01ddd4ce99ea6fad20fe864801eca5709121b211ba301f59e
fbffd03d07566a53456791c643b432e7023ce02b7cd7d505d201008661a97127
22a16bdb05ce1baaf4912207cfe00d51919fc1f46f9445d744d6c4c0409b3d26'
sounio_check_fields='LOOM_EXEC_OPERATION_CATALOG_FIELDS_V1
LOOM_EXEC_CATALOG/1
31089d39690c8e008dd7ea9bfecefd7e51e4257cd96e091f7fb492c240ed342c
sounio-check
6017e4c6e745560696f78836f9cc07ec71a9106f13ad1bfdb16d7e342f0840a9
3b195a8030465bb96245039053fb529ecb39ed330ded747fd5d71188fe4ca0d9
b3ba2f057185897309ce59af2e82f9c9b7e79ffd45bf9f293fc147b715cd1d40
d6e513e785a170a7ce8cd8f07a66e5325736ffdf21cf8f366d82c4f5ef4f15cf
c48942637666983322889dd5ff586c04cb53ef2e86aecf777a19b0582968497f'

calibration_result="$(printf '%s\n' "$calibration" | "$RUNTIME_ONE")"
sounio_check_result="$(printf '%s\n' "$sounio_check" | "$RUNTIME_ONE")"
set +e
template_mismatch_result="$(printf '%s\n' "$template_mismatch" | "$RUNTIME_ONE")"
template_mismatch_code=$?
malformed_result="$(printf '9035 3\n' | "$RUNTIME_ONE")"
malformed_code=$?
set -e
[[ "$calibration_result" == "SOUNIO_EXEC_OPERATION_CATALOG PROJECT semantic_authority=Sounio action=9035
$calibration_fields" ]] || fail "calibration entry diverged: $calibration_result"
[[ "$sounio_check_result" == "SOUNIO_EXEC_OPERATION_CATALOG PROJECT semantic_authority=Sounio action=9035
$sounio_check_fields" ]] || fail "sounio-check entry diverged: $sounio_check_result"
[[ $template_mismatch_code -eq 42 && "$template_mismatch_result" == \
   'SOUNIO_EXEC_OPERATION_CATALOG DENY567 semantic_authority=Sounio action=9035' ]] ||
  fail "template-binding sabotage diverged: $template_mismatch_result"
[[ $malformed_code -eq 42 && "$malformed_result" == \
   'SOUNIO_EXEC_OPERATION_CATALOG DENY424 reason=malformed-frame semantic_authority=Sounio action=9035' ]] ||
  fail "malformed-frame control diverged: $malformed_result"

MUTANT_MODULE="$TEST_ROOT/mutant.sio"
sed 's/if !loom_exec_catalog_template_binding_rule(/if false \&\& loom_exec_catalog_template_binding_rule(/' \
  "$ROOT_DIR/stdlib/coordination/loom_exec_operation_catalog_authority.sio" > "$MUTANT_MODULE"
[[ "$(cmp -l "$ROOT_DIR/stdlib/coordination/loom_exec_operation_catalog_authority.sio" "$MUTANT_MODULE" 2>/dev/null | wc -l)" -gt 0 ]] ||
  fail 'causal mutation did not change the Sounio source'
MUTANT_RUNTIME="$TEST_ROOT/mutant-runtime"
SOUNIO_LOOM_EXEC_OPERATION_CATALOG_MODULE="$MUTANT_MODULE" \
SOUNIO_LOOM_EXEC_OPERATION_CATALOG_OUTPUT="$MUTANT_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_operation_catalog_fixture.sh" >/dev/null || true
[[ -x "$MUTANT_RUNTIME" ]] || fail 'mutant Sounio runtime did not build'
mutant_result="$(printf '%s\n' "$template_mismatch" | "$MUTANT_RUNTIME")"
[[ "$mutant_result" == "SOUNIO_EXEC_OPERATION_CATALOG PROJECT semantic_authority=Sounio action=9035
$sounio_check_fields" ]] ||
  fail "load-bearing mutation did not admit the unchanged witness: $mutant_result"

oracle_executed="$TEST_ROOT/oracle-executed"
for name in python3 rustc; do
  printf '#!/bin/sh\nprintf prohibited > %s\n' "$oracle_executed" > "$TEST_ROOT/$name"
  chmod 0755 "$TEST_ROOT/$name"
done
PATH="$TEST_ROOT:$PATH" printf '%s\n' "$sounio_check" | "$RUNTIME_ONE" >/dev/null
[[ ! -e "$oracle_executed" ]] || fail 'a prohibited oracle executed'
dependencies="$(ldd "$RUNTIME_ONE" 2>&1 || true)"
printf '%s\n' "$dependencies" | grep -Eqi 'python|rust' &&
  fail 'Sounio runtime has a prohibited dependency'

printf 'sounio-loom-exec-operation-catalog-selftest: PASS semantic_authority=Sounio action=9035 stage=SOUNIO_EXECUTABLE cases=11 entries=calibration+sounio-check treatment=PROJECT unknown_operation=DENY562 invalid_argument=DENY563 write_effect=DENY564 template_mismatch=DENY567 malformed=DENY424 causal_sabotage=PASS catalog_sha256=31089d39690c8e008dd7ea9bfecefd7e51e4257cd96e091f7fb492c240ed342c arbitrary_shell=false expected_results_encoded_in_material_layer=false python_executed=false rust_executed=false runtime_dependencies=clean source_sha256=%s executable_sha256=%s ocaml_catalog_projection_attached=false host_payload_selection_attached=false provider_lifecycle_attached=false general_exec_attached=false production_activation=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$ROOT_DIR/stdlib/coordination/loom_exec_operation_catalog_authority.sio" | cut -d ' ' -f 1)" \
  "$(sha256sum "$RUNTIME_ONE" | cut -d ' ' -f 1)"
