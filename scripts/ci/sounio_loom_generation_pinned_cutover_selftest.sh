#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-generation-pin-test.XXXXXX")"
trap 'rm -rf "$work"' EXIT
runtime="$work/runtime"
SOUNIO_LOOM_GENERATION_PINNED_CUTOVER_OUTPUT="$runtime" \
  bash "$ROOT/scripts/dev/build_sounio_loom_generation_pinned_cutover.sh" >/dev/null

expect_allow() {
  local frame="$1" decision="$2" out
  out="$(printf '%s\n' "$frame" | "$runtime")"
  [[ "$out" == "SOUNIO_GENERATION_PINNED_CUTOVER $decision semantic_authority=Sounio action=9048" ]]
}
expect_deny() {
  local frame="$1" decision="$2" out rc
  set +e; out="$(printf '%s\n' "$frame" | "$runtime")"; rc=$?; set -e
  [[ $rc -eq 42 && "$out" == "SOUNIO_GENERATION_PINNED_CUTOVER $decision semantic_authority=Sounio action=9048" ]]
}

expect_allow '9048 1 3 33546239 1 2 3 4 5 14 14' SEALED_CAPABILITY
expect_allow '9048 2 3 33546239 1 2 3 4 5 14 14' SEALED_LEGACY
expect_allow '9048 3 3 33550335 1 2 3 4 5 14 14' CONTINUE
expect_allow '9048 4 3 33550335 1 2 3 4 5 14 14' FORWARD
expect_allow '9048 5 3 33021951 1 2 3 4 5 14 14' BIRTH_PINNED
expect_allow '9048 6 3 33554431 1 2 3 4 5 14 14' CUTOVER_READY
expect_deny '9048 1 2 33546239 1 2 3 4 5 14 14' DENY691
expect_deny '9048 1 3 33546235 1 2 3 4 5 14 14' DENY692
expect_deny '9048 1 3 33546231 1 2 3 4 5 14 14' DENY693
expect_deny '9048 1 3 33545727 1 2 3 4 5 14 14' DENY694
expect_deny '9048 1 3 33542143 1 2 3 4 5 14 14' DENY695
expect_deny '9048 1 3 33480703 1 2 3 4 5 14 14' DENY696
expect_deny '9048 2 3 33021951 1 2 3 4 5 14 14' DENY697
expect_deny '9048 3 3 29356031 1 2 3 4 5 14 14' DENY698
expect_deny '9048 6 3 25165823 1 2 3 4 5 14 14' DENY699
set +e
malformed="$(printf '%s\n' '9048 7 3 33554431 1 2 3 4 5 14 14' | "$runtime")"
malformed_rc=$?
set -e
[[ $malformed_rc -eq 42 && "$malformed" == \
  'SOUNIO_GENERATION_PINNED_CUTOVER DENY424 reason=malformed-frame semantic_authority=Sounio action=9048' ]]

grep -q 'o.python_oracle_absent == 1' "$ROOT/stdlib/coordination/loom_generation_pinned_cutover_authority.sio"
grep -q 'o.rust_oracle_absent == 1' "$ROOT/stdlib/coordination/loom_generation_pinned_cutover_authority.sio"
printf 'sounio-loom-generation-pinned-cutover-selftest: PASS semantic_authority=Sounio action=9048 stage=SOUNIO_EXECUTABLE cases=16 python_oracle_attempt=DENY696 python_executed=false rust_executed=false disposable_oracle_executed=false\n'
