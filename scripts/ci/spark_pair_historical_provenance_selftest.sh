#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_SPARK_PAIR_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_SPARK_PAIR_ENGINE:-madaros}"
AUTHORITY="$ROOT_DIR/stdlib/coordination/spark_pair_historical_provenance.sio"
ENTRYPOINT="$ROOT_DIR/tools/cluster/spark_pair_historical_provenance_main.sio"
VECTORS="$ROOT_DIR/tests/fixtures/spark_pair_arbiter/spark_pair_historical_provenance_vectors.sio"
BUILD="$ROOT_DIR/scripts/dev/build_sounio_spark_pair_historical_provenance.sh"
PARENT_AUTHORITY="$ROOT_DIR/stdlib/coordination/spark_pair_restore_capsule.sio"
PARENT_ENTRYPOINT="$ROOT_DIR/tools/cluster/spark_pair_restore_capsule_main.sio"
PARENT_VECTORS="$ROOT_DIR/tests/fixtures/spark_pair_arbiter/spark_pair_restore_capsule_vectors.sio"
PARENT_BUILD="$ROOT_DIR/scripts/dev/build_sounio_spark_pair_restore_capsule.sh"
PARENT_FREEZE="$ROOT_DIR/tools/cluster/spark_pair_restore_capsule.freeze.v1"
FIRST="$ROOT_DIR/tools/cluster/spark_pair_historical_provenance.first.v1"
FREEZE="$ROOT_DIR/tools/cluster/spark_pair_historical_provenance.freeze.v1"
PARITY="$ROOT_DIR/tools/cluster/spark_pair_historical_provenance.parity-open.v1"
MODE="${1:-}"

[[ $# -le 1 ]] || {
  printf 'spark-pair-historical-provenance-selftest: FAIL: expected zero args or --semantics-only\n' >&2
  exit 1
}
[[ -z "$MODE" || "$MODE" == --semantics-only ]] || {
  printf 'spark-pair-historical-provenance-selftest: FAIL: unknown mode: %s\n' "$MODE" >&2
  exit 1
}

fail() {
  printf 'spark-pair-historical-provenance-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-spark-pair-historical-provenance-selftest.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined_vectors="$work/spark_pair_historical_provenance_vectors.sio"
vectors_executable="$work/spark-pair-historical-provenance-vectors"
adapter="$work/sounio-spark-pair-historical-provenance-plan"
parent_adapter="$work/sounio-spark-pair-restore-capsule-plan"

sed -n '1,$p' "$AUTHORITY" "$VECTORS" > "$combined_vectors"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined_vectors" -o "$vectors_executable"
vectors_result="$($vectors_executable)"
[[ "$vectors_result" == \
    'SOUNIO_SPARK_PAIR_HISTORICAL_PROVENANCE_SELFTEST_PASS vectors=95 authority=Sounio effect=NONE' ]] || \
  fail "Sounio historical provenance vectors failed: $vectors_result"

SOUNIO_SPARK_PAIR_ENGINE="$ENGINE" \
  SOUNIO_SPARK_PAIR_HISTORICAL_PROVENANCE_OUTPUT="$adapter" "$BUILD" >/dev/null

zero=0000000000000000000000000000000000000000000000000000000000000000
source=1111111111111111111111111111111111111111111111111111111111111111
node0=2222222222222222222222222222222222222222222222222222222222222222
node1=3333333333333333333333333333333333333333333333333333333333333333
pair=4444444444444444444444444444444444444444444444444444444444444444
anchor=5555555555555555555555555555555555555555555555555555555555555555
prior=6666666666666666666666666666666666666666666666666666666666666666
parent=d1d67253355be3deab0b3faf05fb345497b1c98dfc15f1194b787830e632fb50

bind_plan="$($adapter 9028 55 30 1 "$source" "$node0" "$node1" "$pair" \
  "$anchor" "$parent" "$zero" 473 255 0 0)"
[[ "$bind_plan" == SOUNIO_SPARK_PAIR_HISTORICAL_PROVENANCE_PLAN_ALLOW* ]] || \
  fail "Sounio historical authority refused a valid bound candidate: $bind_plan"
[[ "$bind_plan" == *' effect=NONE material_dispatch=false '* ]] || \
  fail 'historical plan omitted effect=NONE or exposed material dispatch'

admit_plan="$($adapter 9028 56 31 1 "$source" "$node0" "$node1" "$pair" \
  "$anchor" "$parent" "$prior" 4089 255 255 524287)"
[[ "$admit_plan" == SOUNIO_SPARK_PAIR_HISTORICAL_PROVENANCE_PLAN_ALLOW* ]] || \
  fail "Sounio historical authority refused a complete pre-install source: $admit_plan"
[[ "$admit_plan" == *" parent_9027_sha256=$parent prior_receipt_sha256=$prior "* ]] || \
  fail 'historical admission omitted its parent or predecessor identity'

status_plan="$($adapter 9028 54 30 0 "$zero" "$zero" "$zero" "$zero" \
  "$zero" "$parent" "$zero" 473 0 0 0)"
[[ "$status_plan" == SOUNIO_SPARK_PAIR_HISTORICAL_PROVENANCE_PLAN_ALLOW* ]] || \
  fail "empty historical state is not representable: $status_plan"

set +e
not_present="$($adapter 9028 55 30 1 "$zero" "$node0" "$node1" "$pair" \
  "$anchor" "$parent" "$zero" 473 254 0 0 2>&1)"
not_present_status=$?
partial_admit="$($adapter 9028 56 31 5 "$source" "$node0" "$node1" "$pair" \
  "$anchor" "$parent" "$prior" 4089 255 255 524287 2>&1)"
partial_admit_status=$?
postinstall="$($adapter 9028 55 30 6 "$source" "$node0" "$node1" "$pair" \
  "$anchor" "$parent" "$zero" 473 255 255 524287 2>&1)"
postinstall_status=$?
wrong_parent="$($adapter 9028 55 30 1 "$source" "$node0" "$node1" "$pair" \
  "$anchor" "$node0" "$zero" 473 255 0 0 2>&1)"
wrong_parent_status=$?
missing_prior="$($adapter 9028 56 31 1 "$source" "$node0" "$node1" "$pair" \
  "$anchor" "$parent" "$zero" 4089 255 255 524287 2>&1)"
missing_prior_status=$?
bad_transcript="$($adapter 9028 56 31 1 "$source" "$node0" "$node1" "$pair" \
  "$anchor" "$parent" "$prior" 2041 255 255 524287 2>&1)"
bad_transcript_status=$?
missing_receipt_exact="$($adapter 9028 56 31 1 "$source" "$node0" "$node1" "$pair" \
  "$anchor" "$parent" "$prior" 4057 255 255 524287 2>&1)"
missing_receipt_exact_status=$?
empty_transcript_claim="$($adapter 9028 54 30 0 "$zero" "$zero" "$zero" "$zero" \
  "$zero" "$parent" "$zero" 985 0 0 0 2>&1)"
empty_transcript_claim_status=$?
set -e

[[ $not_present_status -eq 42 && \
    "$not_present" == *'reason=HISTORICAL_SOURCE_NOT_PRESENT code=343'* ]] || \
  fail "NOT_PRESENT did not fail with exact reason 343: status=$not_present_status output=$not_present"
[[ $partial_admit_status -eq 42 && \
    "$partial_admit" == *'reason=PARTIAL_PREINSTALL_BACKUP code=357'* ]] || \
  fail "partial backup was promoted: status=$partial_admit_status output=$partial_admit"
[[ $postinstall_status -eq 42 && "$postinstall" == *'reason=POSTINSTALL_SOURCE code=354'* ]] || \
  fail "post-install capture was promoted: status=$postinstall_status output=$postinstall"
[[ $wrong_parent_status -eq 42 && "$wrong_parent" == *'reason=PARENT_9027_FREEZE code=342'* ]] || \
  fail "wrong parent freeze was accepted: status=$wrong_parent_status output=$wrong_parent"
[[ $missing_prior_status -eq 42 && \
    "$missing_prior" == *'reason=PRIOR_HISTORICAL_RECEIPT code=370'* ]] || \
  fail "missing prior receipt was accepted: status=$missing_prior_status output=$missing_prior"
[[ $bad_transcript_status -eq 42 && \
    "$bad_transcript" == *'reason=PRIOR_HISTORICAL_RECEIPT code=370'* ]] || \
  fail "receipt not bound to the candidate was accepted: status=$bad_transcript_status output=$bad_transcript"
[[ $missing_receipt_exact_status -eq 42 && \
    "$missing_receipt_exact" == *'reason=PRIOR_HISTORICAL_RECEIPT code=370'* ]] || \
  fail "receipt without exact-decision fact was accepted: status=$missing_receipt_exact_status output=$missing_receipt_exact"
[[ $empty_transcript_claim_status -eq 42 && \
    "$empty_transcript_claim" == *'reason=PRIOR_HISTORICAL_RECEIPT code=370'* ]] || \
  fail "empty state accepted a contradictory receipt claim: status=$empty_transcript_claim_status output=$empty_transcript_claim"

expect_malformed() {
  local label="$1"
  shift
  local output status
  set +e
  output="$($adapter "$@" 2>&1)"
  status=$?
  set -e
  [[ $status -eq 64 && "$output" == *'reason=MALFORMED_HISTORICAL_FRAME code=336'* ]] || \
    fail "$label was not refused as malformed: status=$status output=$output"
}

expect_malformed short-digest 9028 55 30 1 1111 "$node0" "$node1" "$pair" \
  "$anchor" "$parent" "$zero" 473 255 0 0
expect_malformed uppercase-digest 9028 55 30 1 \
  A111111111111111111111111111111111111111111111111111111111111111 \
  "$node0" "$node1" "$pair" "$anchor" "$parent" "$zero" 473 255 0 0
expect_malformed wrong-schema 9027 55 30 1 "$source" "$node0" "$node1" "$pair" \
  "$anchor" "$parent" "$zero" 473 255 0 0
expect_malformed mask-overflow 9028 55 30 1 "$source" "$node0" "$node1" "$pair" \
  "$anchor" "$parent" "$zero" 4096 255 0 524287
expect_malformed missing-field 9028 55 30 1 "$source" "$node0" "$node1" "$pair" \
  "$anchor" "$parent" "$zero" 473 255 0

SOUNIO_SPARK_PAIR_ENGINE="$ENGINE" SOUNIO_SPARK_PAIR_RESTORE_CAPSULE_OUTPUT="$parent_adapter" \
  "$PARENT_BUILD" >/dev/null
parent_9026=77dde525618f0f7a683aee1c0744163db84306d4a0eef9ab45d3f79ce1eb4d8e
set +e
parent_accepts_9028="$($parent_adapter 9028 51 20 "$source" "$node0" "$node1" \
  "$zero" "$parent_9026" "$zero" 473 524287 0 2>&1)"
parent_accepts_9028_status=$?
set -e
[[ $parent_accepts_9028_status -eq 64 && \
    "$parent_accepts_9028" == *'reason=MALFORMED_CAPSULE_FRAME code=304'* ]] || \
  fail "frame 9027 adapter accepted frame 9028: status=$parent_accepts_9028_status output=$parent_accepts_9028"
case "$bind_plan" in
  SOUNIO_SPARK_PAIR_RESTORE_CAPSULE_PLAN_ALLOW*)
    fail 'frame 9028 crossed the frame 9027 output prefix' ;;
esac

parent_freeze_sha="$(sha256sum "$PARENT_FREEZE" | cut -d ' ' -f 1)"
[[ "$parent_freeze_sha" == "$parent" ]] || \
  fail 'parent frame 9027 freeze drifted from the prerequisite identity'

parent_value() {
  local key="$1"
  sed -n "s/^${key}=//p" "$PARENT_FREEZE"
}
[[ "$(sha256sum "$PARENT_AUTHORITY" | cut -d ' ' -f 1)" == \
    "$(parent_value authority_sha256)" ]] || fail 'frame 9027 authority drifted from its freeze'
[[ "$(sha256sum "$PARENT_ENTRYPOINT" | cut -d ' ' -f 1)" == \
    "$(parent_value adapter_sha256)" ]] || fail 'frame 9027 adapter drifted from its freeze'
[[ "$(sha256sum "$PARENT_VECTORS" | cut -d ' ' -f 1)" == \
    "$(parent_value expectations_sha256)" ]] || fail 'frame 9027 vectors drifted from its freeze'

if grep -Eiq '\<(python[0-9.]*|rustc|cargo|node|ruby|awk|bc|kubectl|scontrol|systemctl|bpftool)\>' \
    "$AUTHORITY" "$ENTRYPOINT" "$VECTORS" "$BUILD"; then
  fail 'historical provenance semantic path invokes a prohibited oracle or material dispatcher'
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

  [[ "$(artifact_value "$FIRST" status)" == SOUNIO_HISTORICAL_PROVENANCE_EXECUTABLE ]] || \
    fail 'first receipt is not SOUNIO_HISTORICAL_PROVENANCE_EXECUTABLE'
  [[ "$(artifact_value "$FREEZE" status)" == HISTORICAL_PROVENANCE_SCHEMA_FROZEN ]] || \
    fail 'historical provenance schema is not frozen'
  [[ "$(artifact_value "$PARITY" status)" == PARITY_OPEN ]] || \
    fail 'historical provenance parity is not open'
  [[ "$(sha256sum "$FIRST" | cut -d ' ' -f 1)" == \
      "$(artifact_value "$FREEZE" first_executable_receipt_sha256)" ]] || \
    fail 'first receipt drifted from freeze'
  [[ "$parent_freeze_sha" == "$(artifact_value "$FREEZE" parent_9027_freeze_sha256)" ]] || \
    fail 'parent frame 9027 freeze drifted'
  [[ "$(sha256sum "$FREEZE" | cut -d ' ' -f 1)" == \
      "$(artifact_value "$PARITY" historical_provenance_freeze_sha256)" ]] || \
    fail 'parity-open receipt is not bound to the historical provenance freeze'
  [[ "$(artifact_value "$FREEZE" effect)" == NONE ]] || fail 'freeze permits a material effect'
  [[ "$(artifact_value "$FREEZE" material_dispatch)" == false ]] || \
    fail 'freeze exposes a material dispatcher'
  [[ "$(artifact_value "$FREEZE" offline_replay)" == CLOSED ]] || \
    fail 'freeze opened offline replay without an admitted source'
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
    fail 'rebuilt historical provenance adapter differs from frozen executable'
  [[ "$(sha256sum "$vectors_executable" | cut -d ' ' -f 1)" == \
      "$(artifact_value "$FREEZE" expectations_executable_sha256)" ]] || \
    fail 'rebuilt Sounio expectations differ from frozen executable'
fi

printf 'SPARK_PAIR_HISTORICAL_PROVENANCE_SELFTEST_PASS vectors=95 transcript_substitution=DENY370 empty_transcript_claim=DENY370 not_present=DENY343 partial_backup=DENY357 postinstall=DENY354 schema_crossing=DENY parent_9027_freeze=UNCHANGED offline_replay=CLOSED effect=NONE material_dispatch=false\n'
