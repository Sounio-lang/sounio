#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-invocation-material.XXXXXX")"
MANIFEST="$ROOT_DIR/tools/loom/kernel_invocation_cell_authority.freeze.v1"
MANIFEST_SHA256='61918604bf177753c6141f6cd0f05d342a1869ab8fc08d187306a481de33d70e'
AUTHORITY="$TEST_ROOT/invocation-authority"
BROKER_ONE="$TEST_ROOT/principal-broker-one"
BROKER_TWO="$TEST_ROOT/principal-broker-two"
VALID_FRAME="$TEST_ROOT/valid.frame"
CURRENT_FRAME="$TEST_ROOT/current.frame"
MALFORMED_FRAME="$TEST_ROOT/malformed.frame"
MULTILINE_FRAME="$TEST_ROOT/multiline.frame"
TAMPERED_MANIFEST="$TEST_ROOT/manifest-tampered.v1"
TAMPERED_AUTHORITY="$TEST_ROOT/authority-tampered"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-kernel-invocation-cell-material-admission-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

line_hash() {
  local sum
  sum="$(printf '%s\n' "$1" | sha256sum)"
  printf '%s' "${sum%% *}"
}

field() {
  local line="$1" key="$2" token
  for token in $line; do
    if [[ "$token" == "$key="* ]]; then
      printf '%s' "${token#*=}"
      return 0
    fi
  done
  fail "receipt omitted $key"
}

run_refusal() {
  local label="$1"
  shift
  local output status
  set +e
  output="$("$@" 2>&1)"
  status=$?
  set -e
  [[ "$status" == 70 ]] || fail "$label exited $status: $output"
  [[ "$output" == 'loom-kernel-principal-broker: REFUSE reason='* ]] ||
    fail "$label omitted fail-closed refusal: $output"
  printf '%s' "$output"
}

diagnose() {
  "$BROKER_ONE" --diagnose-invocation-cell \
    --invocation-manifest "$1" --invocation-authority "$2" --frame "$3"
}

assert_receipt() {
  local label="$1" output="$2" expected_decision="$3" expected_code="$4"
  local receipt receipt_hash decision
  receipt="$(printf '%s\n' "$output" | sed -n '1p')"
  receipt_hash="$(printf '%s\n' "$output" |
    sed -n '2s/^LOOM_KERNEL_INVOCATION_CELL_MATERIAL_RECEIPT_SHA256 //p')"
  decision="$(printf '%s\n' "$output" | sed -n '3p')"
  [[ "$receipt" == 'LOOM_KERNEL_INVOCATION_CELL_MATERIAL_ADMISSION '* ]] ||
    fail "$label omitted admission receipt"
  [[ "$receipt_hash" =~ ^[0-9a-f]{64}$ ]] || fail "$label receipt hash is malformed"
  [[ "$(line_hash "$receipt")" == "$receipt_hash" ]] || fail "$label receipt hash differs"
  [[ "$(field "$receipt" producing_language)" == C++20 ]] || fail "$label producer drifted"
  [[ "$(field "$receipt" language_role)" == MATERIAL_PARITY ]] || fail "$label role drifted"
  [[ "$(field "$receipt" semantic_authority)" == Sounio ]] || fail "$label laundered authority"
  [[ "$(field "$receipt" action)" == 9029 ]] || fail "$label action drifted"
  [[ "$(field "$receipt" manifest_sha256)" == "$MANIFEST_SHA256" ]] ||
    fail "$label omitted frozen action 9029"
  [[ "$(field "$receipt" decision)" == "$expected_decision" ]] ||
    fail "$label decision class differs"
  [[ "$(field "$receipt" decision_code)" == "$expected_code" ]] ||
    fail "$label decision code differs"
  [[ "$(field "$receipt" material_invocation)" == false &&
     "$(field "$receipt" same_uid_peer_isolation)" == false &&
     "$(field "$receipt" launch_open)" == false ]] ||
    fail "$label promoted a material boundary"
  [[ "$(field "$receipt" decision_sha256)" == "$(line_hash "$decision")" ]] ||
    fail "$label decision hash differs"
}

[[ "$(sha256sum "$MANIFEST" | cut -d' ' -f1)" == "$MANIFEST_SHA256" ]] ||
  fail 'frozen action 9029 manifest drifted'
bash "$ROOT_DIR/scripts/ci/sounio_loom_kernel_invocation_cell_authority_freeze_selftest.sh" >/dev/null

SOUNIO_LOOM_KERNEL_INVOCATION_CELL_OUTPUT="$AUTHORITY" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_invocation_cell_authority.sh" >/dev/null
SOUNIO_LOOM_KERNEL_PRINCIPAL_BROKER_OUTPUT="$BROKER_ONE" \
  bash "$ROOT_DIR/scripts/dev/build_loom_kernel_principal_broker.sh" >/dev/null
SOUNIO_LOOM_KERNEL_PRINCIPAL_BROKER_OUTPUT="$BROKER_TWO" \
  bash "$ROOT_DIR/scripts/dev/build_loom_kernel_principal_broker.sh" >/dev/null
cmp "$BROKER_ONE" "$BROKER_TWO" || fail 'two C++ broker builds differ'

dependencies="$(ldd "$BROKER_ONE")"
if printf '%s\n' "$dependencies" | grep -Eqi 'python|rust'; then
  fail 'material admission acquired a forbidden runtime dependency'
fi
if grep -Eq 'code=481|parent-semantic-join-incomplete' \
    "$ROOT_DIR/tools/loom/src/loom_kernel_principal_broker.cpp"; then
  fail 'C++ broker encodes the expected current-material refusal'
fi

parent_9028='1991017987 113822720 1367310835 4264184359 1117900107 2622180275 1259621157 4224578159'
parent_9025='3253784467 4165106381 4153681002 298013982 643434942 312724736 195896759 132696721'
parent_9023='2365323 2301161672 762924345 38070334 1558458629 1166539901 3590963442 1546541903'
one='1 1 1 1 1 1 1 1'
bindings="$parent_9028 $parent_9025 $parent_9023 $one $one $one $one $one $one $one $one $one"
join='1 1 1 1 1 1'
capsule='1 1 5 6 7 1 1 0 0 1 1 1'
membrane='1 8 9 10 11 1 1 1 1'
scope='1 1 1 1 1 1'
coverage='1 100 1 50 1 1 1 1'
lifecycle='1 1 1 12 13 1 0 0 0'
outcome='0 0 0 0 0 0 0 0 0 0'
authority='1 1 1 1 1 1'
evidence='1 1 10 10'
valid="9029 3 1 $join $capsule $membrane $scope $coverage $lifecycle $outcome $authority $evidence $bindings"
current="9029 3 1 1 1 0 0 1 0 $capsule $membrane $scope $coverage $lifecycle $outcome $authority $evidence $bindings"
printf '%s\n' "$valid" > "$VALID_FRAME"
printf '%s\n' "$current" > "$CURRENT_FRAME"
printf '%s\n' '9029 3' > "$MALFORMED_FRAME"
printf '%s\n%s\n' "$valid" "$current" > "$MULTILINE_FRAME"

positive="$(diagnose "$MANIFEST" "$AUTHORITY" "$VALID_FRAME")"
assert_receipt positive "$positive" ALLOW 0
positive_decision="$(printf '%s\n' "$positive" | sed -n '3p')"
[[ "$positive_decision" == 'SOUNIO_KERNEL_INVOCATION_CELL_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN' ]] ||
  fail 'frozen positive decision differs'

current_result="$(diagnose "$MANIFEST" "$AUTHORITY" "$CURRENT_FRAME")"
assert_receipt current-material "$current_result" DENY 481
current_decision="$(printf '%s\n' "$current_result" | sed -n '3p')"
[[ "$current_decision" == 'SOUNIO_KERNEL_INVOCATION_CELL_DENY code=481 reason=parent-semantic-join-incomplete stage=SEMANTICS_FROZEN' ]] ||
  fail 'frozen Sounio rule did not refuse current material'

malformed="$(diagnose "$MANIFEST" "$AUTHORITY" "$MALFORMED_FRAME")"
assert_receipt malformed "$malformed" DENY 424

cp "$MANIFEST" "$TAMPERED_MANIFEST"
printf '\n' >> "$TAMPERED_MANIFEST"
manifest_refusal="$(run_refusal manifest-tamper diagnose \
  "$TAMPERED_MANIFEST" "$AUTHORITY" "$VALID_FRAME")"
[[ "$manifest_refusal" == *'action 9029 manifest hash mismatch'* ]] ||
  fail 'manifest tamper was not classified'

cp "$AUTHORITY" "$TAMPERED_AUTHORITY"
printf 'X' >> "$TAMPERED_AUTHORITY"
authority_refusal="$(run_refusal authority-tamper diagnose \
  "$MANIFEST" "$TAMPERED_AUTHORITY" "$VALID_FRAME")"
[[ "$authority_refusal" == *'action 9029 authority executable hash mismatch'* ]] ||
  fail 'authority tamper was not classified'

multiline_refusal="$(run_refusal multiline-frame diagnose \
  "$MANIFEST" "$AUTHORITY" "$MULTILINE_FRAME")"
[[ "$multiline_refusal" == *'frame is empty, multiline, or oversized'* ]] ||
  fail 'multiline frame was not refused before Sounio execution'

protocol="$($BROKER_ONE --selftest-protocol)"
[[ "$protocol" == *'launch=closed recycle=closed unknown=denied'* ]] ||
  fail 'material admission opened a broker operation'

printf '%s\n' \
  "sounio-loom-kernel-invocation-cell-material-admission-selftest: PASS semantic_authority=Sounio material_parity=C++20 action=9029 positive=ALLOW current_material=DENY481 malformed=DENY424 manifest_tamper=REFUSE authority_tamper=REFUSE multiline=REFUSE deterministic_rebuild=PASS launch_open=false material_invocation=false same_uid_peer_isolation=false parity_open=false claim_ready=false"
