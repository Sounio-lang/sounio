#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_SPARK_PAIR_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_SPARK_PAIR_ENGINE:-madaros}"
RESTORE_AUTHORITY="$ROOT_DIR/stdlib/coordination/spark_pair_restore_capsule.sio"
RESTORE_ADAPTER="$ROOT_DIR/tools/cluster/spark_pair_restore_capsule_main.sio"
RESTORE_FREEZE="$ROOT_DIR/tools/cluster/spark_pair_restore_capsule.freeze.v1"
PROFILE="$ROOT_DIR/stdlib/coordination/spark_pair_read_only_capture_profile.sio"
ENTRYPOINT="$ROOT_DIR/tools/cluster/spark_pair_read_only_capture_profile_main.sio"
VECTORS="$ROOT_DIR/tests/fixtures/spark_pair_arbiter/spark_pair_read_only_capture_profile_vectors.sio"
BUILD="$ROOT_DIR/scripts/dev/build_sounio_spark_pair_read_only_capture_profile.sh"
FIRST="$ROOT_DIR/tools/cluster/spark_pair_read_only_capture_profile.first.v1"
FREEZE="$ROOT_DIR/tools/cluster/spark_pair_read_only_capture_profile.freeze.v1"
EXPECTED_RESTORE_FREEZE_SHA=d1d67253355be3deab0b3faf05fb345497b1c98dfc15f1194b787830e632fb50

fail() {
  printf 'spark-pair-read-only-capture-profile-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

artifact_value() {
  local file="$1"
  local key="$2"
  sed -n "s/^${key}=//p" "$file"
}

digest() {
  sha256sum "$1" | cut -d' ' -f1
}

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-spark-pair-read-only-capture-profile-selftest.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined_vectors="$work/spark_pair_read_only_capture_profile_vectors.sio"
vectors_executable="$work/spark-pair-read-only-capture-profile-vectors"
adapter="$work/sounio-spark-pair-read-only-capture-profile"

sed -n '1,$p' "$RESTORE_AUTHORITY" "$PROFILE" "$VECTORS" > "$combined_vectors"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined_vectors" -o "$vectors_executable"
vectors_result="$($vectors_executable)"
[[ "$vectors_result" == \
    'SOUNIO_SPARK_PAIR_READ_ONLY_CAPTURE_PROFILE_SELFTEST_PASS vectors=37 authority=Sounio' ]] || \
  fail "Sounio capture profile vectors failed: $vectors_result"

SOUNIO_SPARK_PAIR_ENGINE="$ENGINE" \
SOUNIO_SPARK_PAIR_READ_ONLY_CAPTURE_PROFILE_OUTPUT="$adapter" \
  "$BUILD" >/dev/null

zero=0000000000000000000000000000000000000000000000000000000000000000
snapshot=1111111111111111111111111111111111111111111111111111111111111111
node0=2222222222222222222222222222222222222222222222222222222222222222
node1=3333333333333333333333333333333333333333333333333333333333333333
history=4444444444444444444444444444444444444444444444444444444444444444

profile="$($adapter "$snapshot" "$node0" "$node1" "$zero" 0 131071 127 0 0)"
[[ "$profile" == SOUNIO_SPARK_PAIR_READ_ONLY_CAPTURE_PROFILE_PASS* ]] || \
  fail "complete current observation did not pass its strict profile: $profile"
[[ "$profile" == *'authority_mask=473 snapshot_mask=491455 replay_mask=0 '* ]] || \
  fail "profile emitted the wrong frozen 9027 masks: $profile"
[[ "$profile" == *'historical_preinstall_receipt=NOT_PRESENT preinstall_provenance_exact=false '* ]] || \
  fail "profile promoted current observation into historical provenance: $profile"
[[ "$profile" == *'restore_allowed=false reason=PREINSTALL_PROVENANCE code=315 effect=NONE material_dispatch=false' ]] || \
  fail "profile did not prove exact DENY315 with effect-free 9027: $profile"
case "$profile" in
  *'restorable=true'*|*'state_transition=true'*|*'snapshot_binding_receipt=ISSUED'*)
    fail "profile emitted a forbidden restore promotion: $profile" ;;
esac

expect_status() {
  local expected_status="$1"
  local expected_text="$2"
  shift 2
  local output status
  set +e
  output="$($adapter "$@" 2>&1)"
  status=$?
  set -e
  [[ $status -eq $expected_status ]] || \
    fail "unexpected status=$status expected=$expected_status output=$output"
  [[ "$output" == *"$expected_text"* ]] || \
    fail "missing '$expected_text' in output: $output"
  [[ "$output" != SOUNIO_SPARK_PAIR_READ_ONLY_CAPTURE_PROFILE_PASS* ]] || \
    fail "negative profile input reached PASS: $output"
}

expect_status 42 'restore_reason=PAIR_MANIFEST_DIGEST restore_code=311' \
  "$snapshot" "$node0" "$node1" "$zero" 0 131070 127 0 0
expect_status 42 'restore_reason=SNAPSHOT_CAPTURE_EFFECT restore_code=322' \
  "$snapshot" "$node0" "$node1" "$zero" 0 131071 126 0 0
expect_status 42 'profile_reason=HISTORICAL_RECEIPT_CONTRADICTION profile_code=402' \
  "$snapshot" "$node0" "$node1" "$history" 0 131071 127 0 0
expect_status 42 'profile_reason=HISTORICAL_RECEIPT_CONTRADICTION profile_code=402' \
  "$snapshot" "$node0" "$node1" "$zero" 1 131071 127 0 0
expect_status 42 'profile_reason=HISTORICAL_VALIDATION_NOT_IMPLEMENTED profile_code=403' \
  "$snapshot" "$node0" "$node1" "$history" 1 131071 127 0 0
expect_status 64 'profile_reason=MALFORMED_PROFILE profile_code=401' \
  1111 "$node0" "$node1" "$zero" 0 131071 127 0 0
expect_status 64 'profile_reason=MALFORMED_PROFILE profile_code=401' \
  A111111111111111111111111111111111111111111111111111111111111111 \
  "$node0" "$node1" "$zero" 0 131071 127 0 0
expect_status 64 'profile_reason=MALFORMED_PROFILE profile_code=401' \
  "$snapshot" "$node0" "$node1" "$zero" 0 131071 127 0 0 1

node0_fixture="$work/node0.fixture"
node1_fixture="$work/node1.fixture"
pair_fixture="$work/pair.fixture"
node0_restorable_fixture="$work/node0-restorable.fixture"
node1_restorable_fixture="$work/node1-restorable.fixture"
node0_observation_fixture="$work/node0-observation.fixture"
node1_observation_fixture="$work/node1-observation.fixture"
domain_contract_fixture="$work/domain-contract.fixture"
"$adapter" --fixture-node0 > "$node0_fixture"
"$adapter" --fixture-node1 > "$node1_fixture"
"$adapter" --fixture-pair > "$pair_fixture"
"$adapter" --fixture-node0-restorable > "$node0_restorable_fixture"
"$adapter" --fixture-node1-restorable > "$node1_restorable_fixture"
"$adapter" --fixture-node0-observation > "$node0_observation_fixture"
"$adapter" --fixture-node1-observation > "$node1_observation_fixture"
"$adapter" --fixture-domain-contract > "$domain_contract_fixture"
[[ $(wc -l < "$node0_fixture") -eq 38 ]] || fail 'node0 fixture field count drifted'
[[ $(wc -l < "$node1_fixture") -eq 38 ]] || fail 'node1 fixture field count drifted'
[[ $(wc -l < "$pair_fixture") -eq 16 ]] || fail 'pair fixture field count drifted'
[[ $(wc -l < "$node0_restorable_fixture") -eq 17 ]] || \
  fail 'node0 restorable fixture field count drifted'
[[ $(wc -l < "$node1_restorable_fixture") -eq 17 ]] || \
  fail 'node1 restorable fixture field count drifted'
[[ $(wc -l < "$node0_observation_fixture") -eq 29 ]] || \
  fail 'node0 observation fixture field count drifted'
[[ $(wc -l < "$node1_observation_fixture") -eq 29 ]] || \
  fail 'node1 observation fixture field count drifted'
[[ $(wc -l < "$domain_contract_fixture") -eq 30 ]] || \
  fail 'domain contract fixture field count drifted'
[[ "$(sed -n '2p' "$node0_fixture")" == node_id=spark-3c59 ]] || \
  fail 'node0 fixture identity drifted'
[[ "$(sed -n '2p' "$node1_fixture")" == node_id=spark-8e54 ]] || \
  fail 'node1 fixture identity drifted'
[[ "$(sed -n '11p' "$pair_fixture")" == ordered_pair=true ]] || \
  fail 'pair fixture ordering drifted'
while IFS= read -r field; do
  value="${field#*=}"
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || \
    fail "fixture emitted a non-canonical SHA-256 field: $field"
done < <(rg 'sha256=' "$node0_fixture" "$node1_fixture" "$pair_fixture" \
  "$node0_restorable_fixture" "$node1_restorable_fixture" \
  "$node0_observation_fixture" "$node1_observation_fixture" --no-filename)

restore_freeze_sha="$(digest "$RESTORE_FREEZE")"
[[ "$restore_freeze_sha" == "$EXPECTED_RESTORE_FREEZE_SHA" ]] || \
  fail "frame 9027 freeze drifted: $restore_freeze_sha"
[[ "$(digest "$RESTORE_AUTHORITY")" == "$(artifact_value "$RESTORE_FREEZE" authority_sha256)" ]] || \
  fail 'frame 9027 authority drifted from its freeze'
[[ "$(digest "$RESTORE_ADAPTER")" == "$(artifact_value "$RESTORE_FREEZE" adapter_sha256)" ]] || \
  fail 'frame 9027 adapter drifted from its freeze'

if [[ -f "$FIRST" && -f "$FREEZE" ]]; then
  [[ "$(digest "$FIRST")" == "$(artifact_value "$FREEZE" first_executable_receipt_sha256)" ]] || \
    fail 'capture profile first receipt drifted'
  [[ "$(digest "$PROFILE")" == "$(artifact_value "$FREEZE" profile_authority_sha256)" ]] || \
    fail 'capture profile authority drifted'
  [[ "$(digest "$ENTRYPOINT")" == "$(artifact_value "$FREEZE" adapter_sha256)" ]] || \
    fail 'capture profile adapter drifted'
  [[ "$(digest "$VECTORS")" == "$(artifact_value "$FREEZE" vectors_sha256)" ]] || \
    fail 'capture profile vectors drifted'
  [[ "$(digest "$node0_fixture")" == "$(artifact_value "$FREEZE" node0_fixture_sha256)" ]] || \
    fail 'Sounio node0 fixture drifted'
  [[ "$(digest "$node1_fixture")" == "$(artifact_value "$FREEZE" node1_fixture_sha256)" ]] || \
    fail 'Sounio node1 fixture drifted'
  [[ "$(digest "$pair_fixture")" == "$(artifact_value "$FREEZE" pair_fixture_sha256)" ]] || \
    fail 'Sounio pair fixture drifted'
  [[ "$(digest "$node0_restorable_fixture")" == \
      "$(artifact_value "$FREEZE" node0_restorable_fixture_sha256)" ]] || \
    fail 'Sounio node0 restorable fixture drifted'
  [[ "$(digest "$node1_restorable_fixture")" == \
      "$(artifact_value "$FREEZE" node1_restorable_fixture_sha256)" ]] || \
    fail 'Sounio node1 restorable fixture drifted'
  [[ "$(digest "$node0_observation_fixture")" == \
      "$(artifact_value "$FREEZE" node0_observation_fixture_sha256)" ]] || \
    fail 'Sounio node0 observation fixture drifted'
  [[ "$(digest "$node1_observation_fixture")" == \
      "$(artifact_value "$FREEZE" node1_observation_fixture_sha256)" ]] || \
    fail 'Sounio node1 observation fixture drifted'
  [[ "$(digest "$domain_contract_fixture")" == \
      "$(artifact_value "$FREEZE" domain_contract_fixture_sha256)" ]] || \
    fail 'Sounio material digest domain contract fixture drifted'
fi

printf 'SPARK_PAIR_READ_ONLY_CAPTURE_PROFILE_SELFTEST_PASS vectors=37 expected_restore=DENY315 restorable=false node0_fixture_sha=%s node1_fixture_sha=%s pair_fixture_sha=%s\n' \
  "$(digest "$node0_fixture")" "$(digest "$node1_fixture")" "$(digest "$pair_fixture")"
