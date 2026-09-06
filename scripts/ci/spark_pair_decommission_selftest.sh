#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_SPARK_PAIR_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_SPARK_PAIR_ENGINE:-madaros}"
BASE="$ROOT_DIR/stdlib/coordination/spark_pair_arbiter.sio"
EXTENSION="$ROOT_DIR/stdlib/coordination/spark_pair_decommission.sio"
ENTRYPOINT="$ROOT_DIR/tools/cluster/spark_pair_decommission_main.sio"
VECTORS="$ROOT_DIR/tests/fixtures/spark_pair_arbiter/spark_pair_decommission_vectors.sio"
BUILD="$ROOT_DIR/scripts/dev/build_sounio_spark_pair_decommission.sh"
PARENT_BUILD="$ROOT_DIR/scripts/dev/build_sounio_spark_pair_arbiter.sh"
PARENT_FREEZE="$ROOT_DIR/tools/cluster/spark_pair_arbiter.freeze.v1"
PARENT_CONTROLLER="$ROOT_DIR/scripts/dev/spark_pair_arbiter.sh"
FIRST="$ROOT_DIR/tools/cluster/spark_pair_decommission.first.v1"
FREEZE="$ROOT_DIR/tools/cluster/spark_pair_decommission.freeze.v1"
PARITY="$ROOT_DIR/tools/cluster/spark_pair_decommission.parity-open.v1"
MODE="${1:-}"

[[ $# -le 1 ]] || {
  printf 'spark-pair-decommission-selftest: FAIL: expected zero args or --semantics-only\n' >&2
  exit 1
}
[[ -z "$MODE" || "$MODE" == --semantics-only ]] || {
  printf 'spark-pair-decommission-selftest: FAIL: unknown mode: %s\n' "$MODE" >&2
  exit 1
}

fail() {
  printf 'spark-pair-decommission-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-spark-pair-decommission-selftest.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined_vectors="$work/spark_pair_decommission_vectors.sio"
vectors_executable="$work/spark-pair-decommission-vectors"
adapter="$work/sounio-spark-pair-decommission-plan"
parent_adapter="$work/sounio-spark-pair-arbiter"

sed -n '1,$p' "$BASE" "$EXTENSION" "$VECTORS" > "$combined_vectors"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined_vectors" -o "$vectors_executable"
vectors_result="$($vectors_executable)"
[[ "$vectors_result" == \
    'SOUNIO_SPARK_PAIR_DECOMMISSION_SELFTEST_PASS vectors=69 authority=Sounio effect=NONE' ]] || \
  fail "Sounio decommission vectors failed: $vectors_result"

SOUNIO_SPARK_PAIR_ENGINE="$ENGINE" SOUNIO_SPARK_PAIR_DECOMMISSION_OUTPUT="$adapter" \
  "$BUILD" >/dev/null
plan="$($adapter 9026 34 1 1 1 2 1017 219 1009 131071 7)"
[[ "$plan" == SOUNIO_SPARK_PAIR_DECOMMISSION_PLAN_ALLOW* ]] || \
  fail "Sounio decommission adapter refused the positive witness: $plan"
[[ "$plan" == *' effect=NONE '* ]] || fail 'decommission plan omitted effect=NONE'
case "$plan" in
  SOUNIO_SPARK_PAIR_ALLOW*) fail 'frame 9026 aliases the material 9025 allow prefix' ;;
esac

set +e
wrong_new_schema="$($adapter 9025 33 1 1 1 2 1017 219 1009 131071 7 2>&1)"
wrong_new_status=$?
set -e
[[ $wrong_new_status -eq 64 && "$wrong_new_schema" == *'reason=MALFORMED_FRAME code=104'* ]] || \
  fail "new adapter accepted frame 9025: status=$wrong_new_status output=$wrong_new_schema"

SOUNIO_SPARK_PAIR_ENGINE="$ENGINE" SOUNIO_SPARK_PAIR_OUTPUT="$parent_adapter" \
  "$PARENT_BUILD" >/dev/null
set +e
wrong_parent_schema="$($parent_adapter 9026 33 1 1 1 1017 219 1009 131071 2>&1)"
wrong_parent_status=$?
old_action="$($parent_adapter 9025 33 1 1 1 1017 219 1009 131071 2>&1)"
old_action_status=$?
set -e
[[ $wrong_parent_status -eq 64 && "$wrong_parent_schema" == *'reason=MALFORMED_FRAME code=104'* ]] || \
  fail "parent adapter accepted frame 9026: status=$wrong_parent_status output=$wrong_parent_schema"
[[ $old_action_status -eq 42 && "$old_action" == *'reason=UNSUPPORTED_ACTION'* ]] || \
  fail "parent frame accepted action 33: status=$old_action_status output=$old_action"

grep -Fq '[[ "$result" == SOUNIO_SPARK_PAIR_ALLOW* ]]' "$PARENT_CONTROLLER" || \
  fail 'parent controller result prefix gate drifted'
case "$plan" in
  SOUNIO_SPARK_PAIR_ALLOW*) fail 'decommission plan crossed the parent prefix gate' ;;
esac

parent_value() {
  local key="$1"
  sed -n "s/^${key}=//p" "$PARENT_FREEZE"
}
[[ "$(sha256sum "$BASE" | cut -d ' ' -f 1)" == "$(parent_value authority_sha256)" ]] || \
  fail 'base Sounio authority drifted from parent freeze'
[[ "$(sha256sum "$ROOT_DIR/tools/cluster/spark_pair_arbiter_main.sio" | cut -d ' ' -f 1)" == \
    "$(parent_value adapter_sha256)" ]] || fail 'parent adapter drifted'
[[ "$(sha256sum "$ROOT_DIR/tests/fixtures/spark_pair_arbiter/spark_pair_arbiter_vectors.sio" | cut -d ' ' -f 1)" == \
    "$(parent_value expectations_sha256)" ]] || fail 'parent vectors drifted'

if grep -Eiq '\<(python[0-9.]*|rustc|cargo)\>' "$EXTENSION" "$ENTRYPOINT" "$VECTORS" "$BUILD"; then
  fail 'decommission semantic path invokes a prohibited oracle'
fi

if [[ "$MODE" != --semantics-only ]]; then
  [[ -r "$FIRST" && -r "$FREEZE" && -r "$PARITY" ]] || \
    fail 'first executable receipt, freeze, or parity-open receipt is missing'

  artifact_value() {
    local file="$1" key="$2" count value
    count="$(sed -n "s/^${key}=//p" "$file" | wc -l | tr -d ' ')"
    [[ "$count" == 1 ]] || fail "artifact key missing or duplicated: $key"
    value="$(sed -n "s/^${key}=//p" "$file")"
    [[ -n "$value" ]] || fail "artifact key empty: $key"
    printf '%s\n' "$value"
  }

  verify_artifact_file() {
    local source_key="$1" hash_key="$2" path expected actual
    path="$ROOT_DIR/$(artifact_value "$FREEZE" "$source_key")"
    expected="$(artifact_value "$FREEZE" "$hash_key")"
    [[ -r "$path" ]] || fail "frozen source missing: $path"
    actual="$(sha256sum "$path" | cut -d ' ' -f 1)"
    [[ "$actual" == "$expected" ]] || fail "frozen source drifted: $source_key"
  }

  [[ "$(artifact_value "$FIRST" status)" == SOUNIO_EXECUTABLE ]] || \
    fail 'first receipt is not SOUNIO_EXECUTABLE'
  [[ "$(artifact_value "$FREEZE" status)" == SEMANTICS_FROZEN ]] || \
    fail 'decommission semantics are not frozen'
  [[ "$(artifact_value "$PARITY" status)" == PARITY_OPEN ]] || \
    fail 'decommission parity is not open'
  [[ "$(sha256sum "$FIRST" | cut -d ' ' -f 1)" == \
      "$(artifact_value "$FREEZE" first_executable_receipt_sha256)" ]] || \
    fail 'first receipt drifted from freeze'
  [[ "$(sha256sum "$PARENT_FREEZE" | cut -d ' ' -f 1)" == \
      "$(artifact_value "$FREEZE" parent_freeze_sha256)" ]] || \
    fail 'parent frame 9025 freeze drifted'
  [[ "$(sha256sum "$FREEZE" | cut -d ' ' -f 1)" == \
      "$(artifact_value "$PARITY" semantics_freeze_sha256)" ]] || \
    fail 'parity-open receipt is not bound to the decommission freeze'
  [[ "$(artifact_value "$FREEZE" effect)" == NONE ]] || fail 'freeze permits a material effect'
  [[ "$(artifact_value "$FREEZE" material_dispatch)" == false ]] || \
    fail 'freeze exposes a material dispatcher'
  [[ "$(artifact_value "$FREEZE" claim_ready)" == false ]] || \
    fail 'freeze prematurely claims readiness'

  verify_artifact_file base_authority_source base_authority_sha256
  verify_artifact_file extension_authority_source extension_authority_sha256
  verify_artifact_file adapter_source adapter_sha256
  verify_artifact_file expectations_source expectations_sha256
  verify_artifact_file build_source build_sha256
  verify_artifact_file selftest_source selftest_sha256
  verify_artifact_file compiler_source compiler_sha256
  [[ "$(sha256sum "$adapter" | cut -d ' ' -f 1)" == \
      "$(artifact_value "$FREEZE" native_executable_sha256)" ]] || \
    fail 'rebuilt decommission adapter differs from frozen executable'
  [[ "$(sha256sum "$vectors_executable" | cut -d ' ' -f 1)" == \
      "$(artifact_value "$FREEZE" expectations_executable_sha256)" ]] || \
    fail 'rebuilt Sounio expectations differ from frozen executable'
fi

printf 'SPARK_PAIR_DECOMMISSION_SELFTEST_PASS vectors=69 schema_crossing=DENY parent_dispatch=DENY parent_freeze=UNCHANGED effect=NONE\n'
