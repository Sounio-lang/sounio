#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GATE="$ROOT_DIR/scripts/ci/madaros_changed_tests_gate.sh"

bash -n "$GATE"

selected="$(SOUNIO_MADAROS_CHANGED_TESTS_STACK_KB=0 "$GATE" --select-only \
  tests/run-pass/array_repeat_i8_binding.sio \
  tests/run-pass/generic_struct_return.sio \
  docs/internal/coordination/CI_WATCH_CONTRACT.md)"

grep -Fxq 'tests/run-pass/array_repeat_i8_binding.sio' <<<"$selected"
if grep -Fq 'MADAROS_CHANGED_TESTS_STACK' <<<"$selected"; then
  echo 'madaros-changed-tests: select-only emitted a stack receipt' >&2
  exit 1
fi
if grep -Fq 'generic_struct_return.sio' <<<"$selected"; then
  echo 'madaros-changed-tests: selected an unannotated test' >&2
  exit 1
fi
if grep -Fq 'CI_WATCH_CONTRACT.md' <<<"$selected"; then
  echo 'madaros-changed-tests: selected a path outside tests/run-pass' >&2
  exit 1
fi

non_pr_selected="$(
  CI_EVENT_NAME=push \
  SOUNIO_MADAROS_CHANGED_TESTS_STACK_KB=0 \
    "$GATE" --select-only
)"
grep -Fxq 'tests/run-pass/array_repeat_i8_binding.sio' <<<"$non_pr_selected"
if grep -Fq 'MADAROS_CHANGED_TESTS_STACK' <<<"$non_pr_selected"; then
  echo 'madaros-changed-tests: non-PR select-only emitted a stack receipt' >&2
  exit 1
fi

skip="$(
  (
    ulimit -S -s 8192
    SOUNIO_MADAROS_CHANGED_TESTS_STACK_KB=65536 \
      "$GATE" -- docs/internal/coordination/CI_WATCH_CONTRACT.md
  )
)"
grep -Eq '^MADAROS_CHANGED_TESTS_STACK status=ready scope=linux_ci requested_kb=65536 soft_before_kb=8192 hard_before_kb=(unlimited|[0-9]+) soft_after_kb=65536 hard_after_kb=(unlimited|[0-9]+)$' \
  <<<"$skip"
grep -Fxq 'MADAROS_CHANGED_TESTS_SKIP reason=no_changed_requires_madaros_tests' <<<"$skip"
grep -Fxq 'MADAROS_CHANGED_TESTS_SKIP reason=no_changed_requires_madaros_tests' \
  < <(tail -n 1 <<<"$skip")

set +e
missing_bin="$(
  env -u SOUNIO_MADAROS_CHANGED_TESTS_BIN \
    "$GATE" -- tests/run-pass/array_repeat_i8_binding.sio 2>&1
)"
missing_bin_rc=$?
set -e
if [[ "$missing_bin_rc" == "0" ]]; then
  echo 'madaros-changed-tests: accepted a selected test without an explicit Madaros binary' >&2
  exit 1
fi
grep -Fxq 'MADAROS_CHANGED_TESTS_FAIL reason=missing_explicit_madaros_bin' <<<"$missing_bin"

set +e
wrapper_bin="$(
  SOUNIO_MADAROS_CHANGED_TESTS_BIN="$ROOT_DIR/bin/souc" \
    "$GATE" -- tests/run-pass/array_repeat_i8_binding.sio 2>&1
)"
wrapper_bin_rc=$?
set -e
if [[ "$wrapper_bin_rc" == "0" ]]; then
  echo 'madaros-changed-tests: accepted the public wrapper as a Madaros ELF' >&2
  exit 1
fi
grep -Fxq 'MADAROS_CHANGED_TESTS_FAIL reason=madaros_bin_is_wrapper' <<<"$wrapper_bin"

set +e
wrong_binary="$(
  SOUNIO_MADAROS_CHANGED_TESTS_BIN=/bin/true \
    "$GATE" -- tests/run-pass/array_repeat_i8_binding.sio 2>&1
)"
wrong_binary_rc=$?
set -e
if [[ "$wrong_binary_rc" == "0" ]]; then
  echo 'madaros-changed-tests: accepted a non-Madaros ELF' >&2
  exit 1
fi
grep -Fxq 'MADAROS_CHANGED_TESTS_FAIL reason=madaros_identity_banner_missing' \
  <<<"$wrong_binary"

for invalid_stack_kb in 0 999999999999999999999999999; do
  set +e
  invalid_stack="$(
    SOUNIO_MADAROS_CHANGED_TESTS_BIN=/bin/true \
    SOUNIO_MADAROS_CHANGED_TESTS_STACK_KB="$invalid_stack_kb" \
      "$GATE" -- tests/run-pass/array_repeat_i8_binding.sio 2>&1
  )"
  invalid_stack_rc=$?
  set -e
  if [[ "$invalid_stack_rc" == "0" ]]; then
    echo "madaros-changed-tests: accepted invalid stack override: $invalid_stack_kb" >&2
    exit 1
  fi
  grep -Fxq 'MADAROS_CHANGED_TESTS_FAIL reason=invalid_stack_kb' <<<"$invalid_stack"
done

set +e
raised_stack="$(
  (
    ulimit -S -s 8192
    SOUNIO_MADAROS_CHANGED_TESTS_BIN=/bin/true \
    SOUNIO_MADAROS_CHANGED_TESTS_STACK_KB=65536 \
      "$GATE" -- tests/run-pass/array_repeat_i8_binding.sio
  ) 2>&1
)"
raised_stack_rc=$?
set -e
if [[ "$raised_stack_rc" == "0" ]]; then
  echo 'madaros-changed-tests: accepted a non-Madaros ELF after raising stack' >&2
  exit 1
fi
raised_stack_receipt="$(grep -F 'MADAROS_CHANGED_TESTS_STACK status=ready' <<<"$raised_stack")"
ready_stack_pattern='^MADAROS_CHANGED_TESTS_STACK status=ready scope=linux_ci requested_kb=65536 soft_before_kb=8192 hard_before_kb=(unlimited|[0-9]+) soft_after_kb=65536 hard_after_kb=(unlimited|[0-9]+)$'
if [[ ! "$raised_stack_receipt" =~ $ready_stack_pattern ]]; then
  echo 'madaros-changed-tests: stack raise receipt is malformed' >&2
  exit 1
fi
raised_hard_before="${BASH_REMATCH[1]}"
raised_hard_after="${BASH_REMATCH[2]}"
if [[ "$raised_hard_before" != "$raised_hard_after" ]]; then
  echo 'madaros-changed-tests: stack raise changed the hard limit' >&2
  exit 1
fi
if [[ "$raised_hard_before" != "unlimited" ]] && ((raised_hard_before < 65536)); then
  echo 'madaros-changed-tests: stack raise passed below the finite hard limit' >&2
  exit 1
fi
grep -Fxq 'MADAROS_CHANGED_TESTS_FAIL reason=madaros_identity_banner_missing' \
  < <(tail -n 1 <<<"$raised_stack")

set +e
hard_limited_stack="$(
  (
    ulimit -S -s 8192
    ulimit -H -s 8192
    SOUNIO_MADAROS_CHANGED_TESTS_BIN=/bin/true \
    SOUNIO_MADAROS_CHANGED_TESTS_STACK_KB=65536 \
      "$GATE" -- tests/run-pass/array_repeat_i8_binding.sio
  ) 2>&1
)"
hard_limited_stack_rc=$?
set -e
if [[ "$hard_limited_stack_rc" == "0" ]]; then
  echo 'madaros-changed-tests: accepted a stack target above the hard limit' >&2
  exit 1
fi
grep -Fq 'MADAROS_CHANGED_TESTS_STACK status=blocked scope=linux_ci requested_kb=65536 soft_before_kb=8192 hard_before_kb=8192' \
  <<<"$hard_limited_stack"
grep -Fxq 'MADAROS_CHANGED_TESTS_FAIL reason=stack_hard_limit_too_low' \
  < <(tail -n 1 <<<"$hard_limited_stack")

set +e
bad_diff="$(
  CI_EVENT_NAME=pull_request \
  CI_BASE_SHA=0000000000000000000000000000000000000000 \
  CI_HEAD_SHA=HEAD \
    "$GATE" 2>&1
)"
bad_diff_rc=$?
set -e
if [[ "$bad_diff_rc" == "0" ]]; then
  echo 'madaros-changed-tests: accepted an invalid pull-request diff' >&2
  exit 1
fi
grep -Fxq 'MADAROS_CHANGED_TESTS_FAIL reason=changed_path_diff_failed' \
  < <(tail -n 1 <<<"$bad_diff")

echo 'MADAROS_CHANGED_TESTS_SELFTEST_PASS'
