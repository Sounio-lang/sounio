#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOURCE="$ROOT_DIR/tools/cluster/spark_pair_read_only_capture.cpp"
PROFILE_SELFTEST="$ROOT_DIR/scripts/ci/spark_pair_read_only_capture_profile_selftest.sh"
PROFILE_BUILD="$ROOT_DIR/scripts/dev/build_sounio_spark_pair_read_only_capture_profile.sh"
PROFILE_FREEZE="$ROOT_DIR/tools/cluster/spark_pair_read_only_capture_profile.freeze.v1"
RESTORE_FREEZE="$ROOT_DIR/tools/cluster/spark_pair_restore_capsule.freeze.v1"
ARM64_GATE="$ROOT_DIR/scripts/dev/spark_pair_read_only_capture_arm64_gate.sh"
CXX="${CXX:-c++}"
EXPECTED_PROFILE_FREEZE_SHA=3edfa1e7394b8e82ce8d5e4c81e0450b88dc5b72e1eb71c6acf33f6e2c705223
EXPECTED_RESTORE_FREEZE_SHA=d1d67253355be3deab0b3faf05fb345497b1c98dfc15f1194b787830e632fb50
EXPECTED_COLLECTOR_SOURCE_SHA=385d2756c9ce607834ade6dc22d325090fa04841cfdbb1287278ab19ba34e479
EXPECTED_MULTUS_IMAGE_ID=ghcr.io/k8snetworkplumbingwg/multus-cni@sha256:3c20900b5381fac7f9cbbdfac8370ea10a2f6ed7fbecc678384a9db57047abb1
SELFTEST_MODE="${1:-}"

fail() {
  printf 'spark-pair-read-only-capture-material-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -z "$SELFTEST_MODE" || "$SELFTEST_MODE" == --local-only ]] || \
  fail 'usage: spark_pair_read_only_capture_material_selftest.sh [--local-only]'

digest() {
  sha256sum "$1" | cut -d' ' -f1
}

artifact_value() {
  local file="$1"
  local key="$2"
  sed -n "s/^${key}=//p" "$file"
}

assert_receipt_value() {
  local file="$1" key="$2" expected="$3" actual
  actual="$(artifact_value "$file" "$key")"
  [[ "$actual" == "$expected" ]] || \
    fail "$file has $key=$actual, expected $expected"
}

expect_rc() {
  local expected="$1"
  shift
  local status
  set +e
  "$@" >/dev/null 2>&1
  status=$?
  set -e
  [[ $status -eq $expected ]] || \
    fail "unexpected status=$status expected=$expected command=$*"
}

[[ -f "$SOURCE" ]] || fail "missing material collector: $SOURCE"
[[ "$(digest "$SOURCE")" == "$EXPECTED_COLLECTOR_SOURCE_SHA" ]] || \
  fail 'material collector source drifted from the pre-transport pin'
[[ "$(digest "$PROFILE_FREEZE")" == "$EXPECTED_PROFILE_FREEZE_SHA" ]] || \
  fail 'Sounio capture profile freeze drifted before material parity'
[[ "$(digest "$RESTORE_FREEZE")" == "$EXPECTED_RESTORE_FREEZE_SHA" ]] || \
  fail 'frame 9027 freeze drifted before material parity'

# The material observer may serialize to stdout. It may not acquire authority,
# launch helpers, open files, use the network, or expose a mutating host API.
if rg -n \
  '\b(system|popen|fork|vfork|exec[a-z0-9_]*|posix_spawn|socket|connect|bind|listen|accept|fopen|freopen|open|openat|write|pwrite|rename|unlink|remove|mkdir|chmod|chown|mount|umount|ioctl|kill)\s*\(' \
  "$SOURCE" >/dev/null; then
  fail 'collector contains a forbidden process, file, network, or mutation API'
fi
if rg -n '^#include <(filesystem|fstream|thread|future|cstdio|cstdlib|unistd.h|fcntl.h|sys/socket.h|sys/mount.h)>' \
  "$SOURCE" >/dev/null; then
  fail 'collector contains a forbidden capability-bearing header'
fi
if rg -n \
  '\bkubectl\b.*\b(apply|create|patch|delete|replace|label|taint|cordon|drain)\b|\bscontrol\b.*\b(update|hold|release|requeue|cancel)\b|\bscancel\b|\bsbatch\b|\bsrun\b|\bsystemctl\b.*\b(start|stop|restart|enable|disable|mask|unmask|daemon-reload)\b|\bdocker\b.*\b(run|start|stop|restart|rm|kill|update)\b|\bbpftool\b.*\b(load|attach|detach)\b' \
  "$ARM64_GATE" >/dev/null; then
  fail 'ARM64 gate contains a forbidden cluster or host mutation command'
fi
if rg -n \
  '(^|[[:space:];|&])((/[^[:space:]]*/)?python[0-9.]*|rustc|cargo)([[:space:];|&]|$)' \
  "$SOURCE" "$ARM64_GATE" >/dev/null; then
  fail 'material collector or ARM64 gate invokes a forbidden language oracle'
fi

set +e
oracle_refusal="$(PIREUS_CAPTURE_ORACLE=python bash "$ARM64_GATE" --check 2>&1)"
oracle_status=$?
set -e
[[ $oracle_status -eq 1 &&
   "$oracle_refusal" == *'external oracle injection is forbidden: python'* ]] || \
  fail "deliberate Python oracle injection did not fail closed: $oracle_refusal"

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-spark-pair-read-only-capture-material.XXXXXX")"
trap 'rm -rf "$work"' EXIT
collector="$work/spark-pair-read-only-capture"
profile_adapter="$work/sounio-spark-pair-read-only-capture-profile"

"$CXX" \
  -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-exceptions -fno-rtti -fstack-protector-strong -D_FORTIFY_SOURCE=3 \
  "$SOURCE" -o "$collector"
"$collector" --selftest | \
  rg -Fx 'PIREUS_SPARK_PAIR_READ_ONLY_CAPTURE_CPP_SELFTEST_PASS role=MATERIAL_OBSERVER_NON_AUTHORITY' \
  >/dev/null || fail 'C++20 collector selftest failed'
[[ "$(printf abc | "$collector" --sha256)" == \
    ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad ]] || \
  fail 'C++20 collector SHA-256 implementation failed the standard abc vector'
[[ "$(printf payload | "$collector" --hash-domain restorable.systemd_system)" == \
    daf458234fb8f6074294d663a5d71eb272fa680ef1c4edc1e231d70a7421b1c6 ]] || \
  fail 'C++20 collector domain framing digest drifted'
expect_rc 64 "$collector" --hash-domain NOT_A_FROZEN_DOMAIN

bash "$PROFILE_SELFTEST" >/dev/null
SOUNIO_SPARK_PAIR_READ_ONLY_CAPTURE_PROFILE_OUTPUT="$profile_adapter" \
  bash "$PROFILE_BUILD" >/dev/null

for fixture in node0 node1 node0-restorable node1-restorable \
    node0-observation node1-observation domain-contract pair; do
  "$profile_adapter" "--fixture-${fixture}" > "$work/sounio-${fixture}.v1"
  "$collector" "--fixture-${fixture}" > "$work/cpp-${fixture}.v1"
  cmp -s "$work/sounio-${fixture}.v1" "$work/cpp-${fixture}.v1" || \
    fail "C++20 ${fixture} bytes differ from frozen Sounio bytes"
  freeze_key="${fixture//-/_}_fixture_sha256"
  expected="$(artifact_value "$PROFILE_FREEZE" "$freeze_key")"
  [[ "$(digest "$work/cpp-${fixture}.v1")" == "$expected" ]] || \
    fail "C++20 ${fixture} fixture digest differs from Sounio freeze"
done

hashes=(
  1111111111111111111111111111111111111111111111111111111111111111
  2222222222222222222222222222222222222222222222222222222222222222
  3333333333333333333333333333333333333333333333333333333333333333
  4444444444444444444444444444444444444444444444444444444444444444
  5555555555555555555555555555555555555555555555555555555555555555
  6666666666666666666666666666666666666666666666666666666666666666
  7777777777777777777777777777777777777777777777777777777777777777
  8888888888888888888888888888888888888888888888888888888888888888
  9999999999999999999999999999999999999999999999999999999999999999
  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
  cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
  eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
  ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
  1212121212121212121212121212121212121212121212121212121212121212
  1313131313131313131313131313131313131313131313131313131313131313
  1414141414141414141414141414141414141414141414141414141414141414
  1515151515151515151515151515151515151515151515151515151515151515
  1515151515151515151515151515151515151515151515151515151515151515
)
node_args=(
  --node spark-3c59 "${hashes[@]}"
  1788250000000000000 1788250001000000000 true true true
)
"$collector" "${node_args[@]}" > "$work/node-valid.v1"
cmp -s "$work/node-valid.v1" "$work/cpp-node0.v1" || \
  fail 'validated --node path differs from the frozen node0 fixture'
"$collector" --restorable spark-3c59 "${hashes[@]:0:8}" > \
  "$work/restorable-valid.v1"
cmp -s "$work/restorable-valid.v1" "$work/cpp-node0-restorable.v1" || \
  fail 'validated --restorable path differs from the frozen node0 fixture'
"$collector" --observation spark-3c59 "${hashes[@]:8:12}" \
  1788250000000000000 1788250001000000000 true true true > \
  "$work/observation-valid.v1"
cmp -s "$work/observation-valid.v1" "$work/cpp-node0-observation.v1" || \
  fail 'validated --observation path differs from the frozen node0 fixture'

bad_hashes=("${hashes[@]}")
bad_hashes[0]=1111
expect_rc 64 "$collector" --node spark-3c59 "${bad_hashes[@]}" \
  1788250000000000000 1788250001000000000 true true true
bad_hashes=("${hashes[@]}")
bad_hashes[0]=A111111111111111111111111111111111111111111111111111111111111111
expect_rc 64 "$collector" --node spark-3c59 "${bad_hashes[@]}" \
  1788250000000000000 1788250001000000000 true true true
expect_rc 64 "$collector" --node spark-other "${hashes[@]}" \
  1788250000000000000 1788250001000000000 true true true
expect_rc 64 "$collector" --node spark-3c59 "${hashes[@]}" \
  01788250000000000000 1788250001000000000 true true true
expect_rc 64 "$collector" --node spark-3c59 "${hashes[@]}" \
  1788250000000000000 1788250000000000000 true true true
expect_rc 64 "$collector" --node spark-3c59 "${hashes[@]}" \
  1788250000000000000 1788250001000000000 yes true true
expect_rc 64 "$collector" "${node_args[@]}" trailing

unequal_hashes=("${hashes[@]}")
unequal_hashes[19]=1616161616161616161616161616161616161616161616161616161616161616
"$collector" --node spark-3c59 "${unequal_hashes[@]}" \
  1788250000000000000 1788250001000000000 true true true > "$work/node-unresolved.v1"
rg -Fx 'scheduler_mutation=UNRESOLVED' "$work/node-unresolved.v1" >/dev/null || \
  fail 'managed-state mismatch did not fail closed'
rg -Fx 'host_configuration_mutation=UNRESOLVED' "$work/node-unresolved.v1" >/dev/null || \
  fail 'managed-state mismatch did not fail closed for host configuration'

pair_args=(
  --pair
  1616161616161616161616161616161616161616161616161616161616161616
  1717171717171717171717171717171717171717171717171717171717171717
  1818181818181818181818181818181818181818181818181818181818181818
  1919191919191919191919191919191919191919191919191919191919191919
  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
)
"$collector" "${pair_args[@]}" > "$work/pair-valid.v1"
cmp -s "$work/pair-valid.v1" "$work/cpp-pair.v1" || \
  fail 'validated --pair path differs from the frozen pair fixture'
expect_rc 64 "$collector" --pair 1111 "${pair_args[@]:2}"
expect_rc 64 "$collector" --pair \
  A616161616161616161616161616161616161616161616161616161616161616 \
  "${pair_args[@]:2}"
expect_rc 64 "$collector" "${pair_args[@]}" trailing

if rg -n '^(restorable=true|snapshot_binding_receipt=ISSUED|state_transition=true)$' \
  "$work"/*.v1 >/dev/null; then
  fail 'material observer promoted a capture into restore authority'
fi

zero=0000000000000000000000000000000000000000000000000000000000000000
node0_sha="$(digest "$work/cpp-node0.v1")"
node1_sha="$(digest "$work/cpp-node1.v1")"
pair_sha="$(digest "$work/cpp-pair.v1")"
decision="$($profile_adapter "$pair_sha" "$node0_sha" "$node1_sha" "$zero" \
  0 131071 127 0 0)"
[[ "$decision" == SOUNIO_SPARK_PAIR_READ_ONLY_CAPTURE_PROFILE_PASS* ]] || \
  fail "material bytes were refused before frame 9027: $decision"
[[ "$decision" == *'frame_9027_invoked=true restore_allowed=false reason=PREINSTALL_PROVENANCE code=315 '* ]] || \
  fail "material bytes did not terminate at exact frame 9027 DENY315: $decision"
[[ "$decision" == *'restorable=false snapshot_binding_receipt=NOT_ISSUED state_transition=false'* ]] || \
  fail 'material parity promoted a forbidden restore or state transition'

if [[ "$SELFTEST_MODE" == --local-only ]]; then
  printf 'SPARK_PAIR_READ_ONLY_CAPTURE_MATERIAL_LOCAL_SELFTEST_PASS authority=Sounio material=C++20 byte_parity=8 negative_vectors=13 frame_9027=DENY315 restorable=false cpp_sha256=%s\n' \
    "$(digest "$SOURCE")"
  exit 0
fi

node0_restorable="$ROOT_DIR/tools/cluster/spark_pair_read_only_capture.node0-restorable.v1"
node0_observation="$ROOT_DIR/tools/cluster/spark_pair_read_only_capture.node0-observation.v1"
node0_manifest="$ROOT_DIR/tools/cluster/spark_pair_read_only_capture.node0-manifest.v1"
node1_restorable="$ROOT_DIR/tools/cluster/spark_pair_read_only_capture.node1-restorable.v1"
node1_observation="$ROOT_DIR/tools/cluster/spark_pair_read_only_capture.node1-observation.v1"
node1_manifest="$ROOT_DIR/tools/cluster/spark_pair_read_only_capture.node1-manifest.v1"
pair_receipt="$ROOT_DIR/tools/cluster/spark_pair_read_only_capture.pair-receipt.v1"
material_receipt="$ROOT_DIR/tools/cluster/spark_pair_read_only_capture.material-parity.v1"
for live in "$node0_restorable" "$node0_observation" "$node0_manifest" \
    "$node1_restorable" "$node1_observation" "$node1_manifest" \
    "$pair_receipt" "$material_receipt"; do
  [[ -f "$live" ]] || fail "live ARM64 receipt is missing: $live"
done
[[ $(wc -l < "$node0_restorable") -eq 17 &&
   $(wc -l < "$node1_restorable") -eq 17 &&
   $(wc -l < "$node0_observation") -eq 29 &&
   $(wc -l < "$node1_observation") -eq 29 &&
   $(wc -l < "$node0_manifest") -eq 38 &&
   $(wc -l < "$node1_manifest") -eq 38 &&
   $(wc -l < "$pair_receipt") -eq 16 ]] || \
  fail 'live receipt field count drifted'

for receipt in "$node0_restorable" "$node0_observation" "$node0_manifest" \
    "$node1_restorable" "$node1_observation" "$node1_manifest" "$pair_receipt"; do
  assert_receipt_value "$receipt" historical_preinstall_receipt NOT_PRESENT
  assert_receipt_value "$receipt" historical_preinstall_receipt_sha256 "$zero"
  assert_receipt_value "$receipt" restorable false
  assert_receipt_value "$receipt" snapshot_binding_receipt NOT_ISSUED
  assert_receipt_value "$receipt" state_transition false
done
for receipt in "$node0_observation" "$node0_manifest" \
    "$node1_observation" "$node1_manifest"; do
  assert_receipt_value "$receipt" scheduler_mutation NONE
  assert_receipt_value "$receipt" host_configuration_mutation NONE
done
[[ "$(artifact_value "$node0_observation" managed_state_pre_sha256)" == \
   "$(artifact_value "$node0_observation" managed_state_post_sha256)" ]] || \
  fail 'node0 live managed-state sentinels differ'
[[ "$(artifact_value "$node1_observation" managed_state_pre_sha256)" == \
   "$(artifact_value "$node1_observation" managed_state_post_sha256)" ]] || \
  fail 'node1 live managed-state sentinels differ'

assert_receipt_value "$pair_receipt" node0_manifest_sha256 "$(digest "$node0_manifest")"
assert_receipt_value "$pair_receipt" node0_restorable_sha256 "$(digest "$node0_restorable")"
assert_receipt_value "$pair_receipt" node0_observation_sha256 "$(digest "$node0_observation")"
assert_receipt_value "$pair_receipt" node1_manifest_sha256 "$(digest "$node1_manifest")"
assert_receipt_value "$pair_receipt" node1_restorable_sha256 "$(digest "$node1_restorable")"
assert_receipt_value "$pair_receipt" node1_observation_sha256 "$(digest "$node1_observation")"

live_pair_sha="$(digest "$pair_receipt")"
live_node0_sha="$(digest "$node0_manifest")"
live_node1_sha="$(digest "$node1_manifest")"
live_decision="$($profile_adapter "$live_pair_sha" "$live_node0_sha" \
  "$live_node1_sha" "$zero" 0 131071 127 0 0)"
[[ "$live_decision" == *'frame_9027_invoked=true restore_allowed=false reason=PREINSTALL_PROVENANCE code=315 '* ]] || \
  fail "live ARM64 receipts did not replay to exact DENY315: $live_decision"
[[ "$live_decision" != *'restore_allowed=true'* ]] || \
  fail 'live ARM64 receipt reached forbidden action 51 ALLOW'

assert_receipt_value "$material_receipt" status ARM64_PAIR_CAPTURED_FRAME_9027_DENY315
assert_receipt_value "$material_receipt" profile_freeze_sha256 "$EXPECTED_PROFILE_FREEZE_SHA"
assert_receipt_value "$material_receipt" collector_source_sha256 "$(digest "$SOURCE")"
assert_receipt_value "$material_receipt" capture_gate_source_sha256 "$(digest "$ARM64_GATE")"
assert_receipt_value "$material_receipt" arm64_node0_multus_image_id "$EXPECTED_MULTUS_IMAGE_ID"
assert_receipt_value "$material_receipt" arm64_node1_multus_image_id "$EXPECTED_MULTUS_IMAGE_ID"
assert_receipt_value "$material_receipt" pair_manifest_sha256 "$live_pair_sha"
assert_receipt_value "$material_receipt" native_domain_hashing true
assert_receipt_value "$material_receipt" scheduler_mutation NONE
assert_receipt_value "$material_receipt" host_configuration_mutation NONE
assert_receipt_value "$material_receipt" restore_allowed false
assert_receipt_value "$material_receipt" restore_code 315
assert_receipt_value "$material_receipt" restorable false
assert_receipt_value "$material_receipt" offline_replay NOT_OPEN
assert_receipt_value "$material_receipt" supersedes_rejected_pair_sha256 \
  c0f6235dc7b93aca8d674ba66c28d66ea34d00b0bcd5b36904cef8b8891120a9
[[ "$(artifact_value "$material_receipt" sounio_decision)" == "$live_decision" ]] || \
  fail 'material receipt did not bind the exact live Sounio decision'

printf 'SPARK_PAIR_READ_ONLY_CAPTURE_MATERIAL_SELFTEST_PASS authority=Sounio material=C++20 byte_parity=8 negative_vectors=13 live_pair=%s frame_9027=DENY315 restorable=false cpp_sha256=%s\n' \
  "$live_pair_sha" \
  "$(digest "$SOURCE")"
