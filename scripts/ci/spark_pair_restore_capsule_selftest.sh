#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_SPARK_PAIR_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_SPARK_PAIR_ENGINE:-madaros}"
AUTHORITY="$ROOT_DIR/stdlib/coordination/spark_pair_restore_capsule.sio"
ENTRYPOINT="$ROOT_DIR/tools/cluster/spark_pair_restore_capsule_main.sio"
VECTORS="$ROOT_DIR/tests/fixtures/spark_pair_arbiter/spark_pair_restore_capsule_vectors.sio"
BUILD="$ROOT_DIR/scripts/dev/build_sounio_spark_pair_restore_capsule.sh"
PARENT_AUTHORITY="$ROOT_DIR/stdlib/coordination/spark_pair_decommission.sio"
PARENT_ENTRYPOINT="$ROOT_DIR/tools/cluster/spark_pair_decommission_main.sio"
PARENT_BASE="$ROOT_DIR/stdlib/coordination/spark_pair_arbiter.sio"
PARENT_BUILD="$ROOT_DIR/scripts/dev/build_sounio_spark_pair_decommission.sh"
PARENT_FREEZE="$ROOT_DIR/tools/cluster/spark_pair_decommission.freeze.v1"
PARENT_CONTROLLER="$ROOT_DIR/scripts/dev/spark_pair_arbiter.sh"
FIRST="$ROOT_DIR/tools/cluster/spark_pair_restore_capsule.first.v1"
FREEZE="$ROOT_DIR/tools/cluster/spark_pair_restore_capsule.freeze.v1"
PARITY="$ROOT_DIR/tools/cluster/spark_pair_restore_capsule.parity-open.v1"
MODE="${1:-}"

[[ $# -le 1 ]] || {
  printf 'spark-pair-restore-capsule-selftest: FAIL: expected zero args or --semantics-only\n' >&2
  exit 1
}
[[ -z "$MODE" || "$MODE" == --semantics-only ]] || {
  printf 'spark-pair-restore-capsule-selftest: FAIL: unknown mode: %s\n' "$MODE" >&2
  exit 1
}

fail() {
  printf 'spark-pair-restore-capsule-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-spark-pair-restore-capsule-selftest.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined_vectors="$work/spark_pair_restore_capsule_vectors.sio"
vectors_executable="$work/spark-pair-restore-capsule-vectors"
adapter="$work/sounio-spark-pair-restore-capsule-plan"
parent_adapter="$work/sounio-spark-pair-decommission-plan"

sed -n '1,$p' "$AUTHORITY" "$VECTORS" > "$combined_vectors"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined_vectors" -o "$vectors_executable"
vectors_result="$($vectors_executable)"
[[ "$vectors_result" == \
    'SOUNIO_SPARK_PAIR_RESTORE_CAPSULE_SELFTEST_PASS vectors=77 authority=Sounio effect=NONE' ]] || \
  fail "Sounio restore capsule vectors failed: $vectors_result"

SOUNIO_SPARK_PAIR_ENGINE="$ENGINE" SOUNIO_SPARK_PAIR_RESTORE_CAPSULE_OUTPUT="$adapter" \
  "$BUILD" >/dev/null

zero=0000000000000000000000000000000000000000000000000000000000000000
capsule=1111111111111111111111111111111111111111111111111111111111111111
node0=2222222222222222222222222222222222222222222222222222222222222222
node1=3333333333333333333333333333333333333333333333333333333333333333
witness=4444444444444444444444444444444444444444444444444444444444444444
parent=77dde525618f0f7a683aee1c0744163db84306d4a0eef9ab45d3f79ce1eb4d8e
prior=5555555555555555555555555555555555555555555555555555555555555555
plan="$($adapter 9027 51 20 "$capsule" "$node0" "$node1" "$zero" "$parent" "$zero" 473 524287 0)"
[[ "$plan" == SOUNIO_SPARK_PAIR_RESTORE_CAPSULE_PLAN_ALLOW* ]] || \
  fail "Sounio restore capsule adapter refused the positive witness: $plan"
[[ "$plan" == *' effect=NONE material_dispatch=false '* ]] || \
  fail 'restore capsule plan omitted effect=NONE or exposed material dispatch'
case "$plan" in
  SOUNIO_SPARK_PAIR_ALLOW*|SOUNIO_SPARK_PAIR_DECOMMISSION_PLAN_ALLOW*)
    fail 'frame 9027 aliases an existing material or decommission prefix' ;;
esac

replay_plan="$($adapter 9027 52 21 "$capsule" "$node0" "$node1" "$witness" \
  "$parent" "$prior" 505 1048575 131071)"
[[ "$replay_plan" == SOUNIO_SPARK_PAIR_RESTORE_CAPSULE_PLAN_ALLOW* ]] || \
  fail "Sounio restore capsule adapter refused the bound replay witness: $replay_plan"
[[ "$replay_plan" == *" parent_9026_sha256=$parent prior_receipt_sha256=$prior "* ]] || \
  fail 'restore capsule plan omitted its parent or predecessor receipt identity'

set +e
wrong_parent="$($adapter 9027 51 20 "$capsule" "$node0" "$node1" "$zero" \
  "$node0" "$zero" 473 524287 0 2>&1)"
wrong_parent_status=$?
missing_prior="$($adapter 9027 52 21 "$capsule" "$node0" "$node1" "$witness" \
  "$parent" "$zero" 505 1048575 131071 2>&1)"
missing_prior_status=$?
set -e
[[ $wrong_parent_status -eq 42 && "$wrong_parent" == *'reason=PARENT_9026_FREEZE code=307'* ]] || \
  fail "restore capsule accepted the wrong parent freeze: status=$wrong_parent_status output=$wrong_parent"
[[ $missing_prior_status -eq 42 && "$missing_prior" == *'reason=PRIOR_RECEIPT code=324'* ]] || \
  fail "restore capsule accepted a missing prior receipt: status=$missing_prior_status output=$missing_prior"

expect_malformed() {
  local label="$1"
  shift
  local output status
  set +e
  output="$($adapter "$@" 2>&1)"
  status=$?
  set -e
  [[ $status -eq 64 && "$output" == *'reason=MALFORMED_CAPSULE_FRAME code=304'* ]] || \
    fail "$label was not refused as malformed: status=$status output=$output"
}

expect_malformed short-digest 9027 51 20 1111 "$node0" "$node1" "$zero" "$parent" "$zero" 473 524287 0
expect_malformed uppercase-digest 9027 51 20 \
  A111111111111111111111111111111111111111111111111111111111111111 \
  "$node0" "$node1" "$zero" "$parent" "$zero" 473 524287 0
expect_malformed nonhex-digest 9027 51 20 \
  g111111111111111111111111111111111111111111111111111111111111111 \
  "$node0" "$node1" "$zero" "$parent" "$zero" 473 524287 0
expect_malformed long-digest 9027 51 20 \
  11111111111111111111111111111111111111111111111111111111111111111 \
  "$node0" "$node1" "$zero" "$parent" "$zero" 473 524287 0
expect_malformed wrong-schema 9026 51 20 "$capsule" "$node0" "$node1" "$zero" "$parent" "$zero" 473 524287 0
expect_malformed missing-field 9027 51 20 "$capsule" "$node0" "$node1" "$zero" "$parent" "$zero" 473 524287

SOUNIO_SPARK_PAIR_ENGINE="$ENGINE" SOUNIO_SPARK_PAIR_DECOMMISSION_OUTPUT="$parent_adapter" \
  "$PARENT_BUILD" >/dev/null
set +e
parent_accepts_9027="$($parent_adapter 9027 51 20 1 1 1 1 1 1 1 1 2>&1)"
parent_accepts_9027_status=$?
set -e
[[ $parent_accepts_9027_status -eq 64 && \
    "$parent_accepts_9027" == *'reason=MALFORMED_FRAME code=104'* ]] || \
  fail "frame 9026 adapter accepted frame 9027: status=$parent_accepts_9027_status output=$parent_accepts_9027"

grep -Fq '[[ "$result" == SOUNIO_SPARK_PAIR_ALLOW* ]]' "$PARENT_CONTROLLER" || \
  fail 'frame 9025 controller result prefix gate drifted'
case "$plan" in
  SOUNIO_SPARK_PAIR_ALLOW*) fail 'restore capsule plan crossed the frame 9025 prefix gate' ;;
esac

parent_freeze_sha="$(sha256sum "$PARENT_FREEZE" | cut -d ' ' -f 1)"
[[ "$parent_freeze_sha" == 77dde525618f0f7a683aee1c0744163db84306d4a0eef9ab45d3f79ce1eb4d8e ]] || \
  fail 'parent frame 9026 freeze drifted from the prerequisite identity'

parent_value() {
  local key="$1"
  sed -n "s/^${key}=//p" "$PARENT_FREEZE"
}
[[ "$(sha256sum "$PARENT_BASE" | cut -d ' ' -f 1)" == "$(parent_value base_authority_sha256)" ]] || \
  fail 'frame 9025 base authority drifted from frame 9026 freeze'
[[ "$(sha256sum "$PARENT_AUTHORITY" | cut -d ' ' -f 1)" == "$(parent_value extension_authority_sha256)" ]] || \
  fail 'frame 9026 authority drifted from its freeze'
[[ "$(sha256sum "$PARENT_ENTRYPOINT" | cut -d ' ' -f 1)" == "$(parent_value adapter_sha256)" ]] || \
  fail 'frame 9026 adapter drifted from its freeze'

if grep -Eiq '\<(python[0-9.]*|rustc|cargo|kubectl|scontrol|systemctl|bpftool)\>' \
    "$AUTHORITY" "$ENTRYPOINT" "$VECTORS" "$BUILD"; then
  fail 'restore capsule semantic path invokes a prohibited oracle or material dispatcher'
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

  [[ "$(artifact_value "$FIRST" status)" == SOUNIO_CAPSULE_EXECUTABLE ]] || \
    fail 'first receipt is not SOUNIO_CAPSULE_EXECUTABLE'
  [[ "$(artifact_value "$FREEZE" status)" == CAPSULE_SCHEMA_FROZEN ]] || \
    fail 'restore capsule schema is not frozen'
  [[ "$(artifact_value "$PARITY" status)" == PARITY_OPEN ]] || \
    fail 'restore capsule parity is not open'
  [[ "$(sha256sum "$FIRST" | cut -d ' ' -f 1)" == \
      "$(artifact_value "$FREEZE" first_executable_receipt_sha256)" ]] || \
    fail 'first receipt drifted from freeze'
  [[ "$parent_freeze_sha" == "$(artifact_value "$FREEZE" parent_9026_freeze_sha256)" ]] || \
    fail 'parent frame 9026 freeze drifted'
  [[ "$(sha256sum "$FREEZE" | cut -d ' ' -f 1)" == \
      "$(artifact_value "$PARITY" capsule_schema_freeze_sha256)" ]] || \
    fail 'parity-open receipt is not bound to the capsule freeze'
  [[ "$(artifact_value "$FREEZE" effect)" == NONE ]] || fail 'freeze permits a material effect'
  [[ "$(artifact_value "$FREEZE" material_dispatch)" == false ]] || \
    fail 'freeze exposes a material dispatcher'
  [[ "$(artifact_value "$FREEZE" claim_ready)" == false ]] || \
    fail 'freeze prematurely claims readiness'

  verify_artifact_file authority_source authority_sha256
  verify_artifact_file adapter_source adapter_sha256
  verify_artifact_file expectations_source expectations_sha256
  verify_artifact_file build_source build_sha256
  verify_artifact_file selftest_source selftest_sha256
  verify_artifact_file compiler_source compiler_sha256
  [[ "$(sha256sum "$adapter" | cut -d ' ' -f 1)" == \
      "$(artifact_value "$FREEZE" native_executable_sha256)" ]] || \
    fail 'rebuilt restore capsule adapter differs from frozen executable'
  [[ "$(sha256sum "$vectors_executable" | cut -d ' ' -f 1)" == \
      "$(artifact_value "$FREEZE" expectations_executable_sha256)" ]] || \
    fail 'rebuilt Sounio expectations differ from frozen executable'
fi

printf 'SPARK_PAIR_RESTORE_CAPSULE_SELFTEST_PASS vectors=77 schema_crossing=DENY parent_dispatch=DENY parent_9026_freeze=UNCHANGED effect=NONE material_dispatch=false\n'
