#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVENT_NAME="${CI_EVENT_NAME:-${GITHUB_EVENT_NAME:-pull_request}}"
BASE_SHA="${CI_BASE_SHA:-}"
HEAD_SHA="${CI_HEAD_SHA:-HEAD}"
MADAROS_BIN="${SOUNIO_MADAROS_CHANGED_TESTS_BIN:-}"
SELECT_ONLY=0

fail() {
  echo "MADAROS_CHANGED_TESTS_FAIL reason=$1" >&2
  exit 1
}

if [[ "${1:-}" == "--select-only" ]]; then
  SELECT_ONLY=1
  shift
fi

paths=()
if (($#)); then
  paths=("$@")
elif [[ "$EVENT_NAME" == "pull_request" ]]; then
  [[ -n "$BASE_SHA" ]] || fail "missing_pull_request_base_sha"
  if ! changed_paths="$(
    git -C "$ROOT_DIR" diff --name-only --diff-filter=ACMR "$BASE_SHA...$HEAD_SHA"
  )"; then
    fail "changed_path_diff_failed"
  fi
  if [[ -n "$changed_paths" ]]; then
    mapfile -t paths <<<"$changed_paths"
  fi
else
  paths=("tests/run-pass/array_repeat_i8_binding.sio")
fi

selected=()
for path in "${paths[@]}"; do
  case "$path" in
    tests/run-pass/*.sio) ;;
    *) continue ;;
  esac
  [[ -f "$ROOT_DIR/$path" ]] || continue
  grep -Fq '//@ requires: madaros' "$ROOT_DIR/$path" || continue
  selected+=("$path")
done

if [[ "$SELECT_ONLY" == "1" ]]; then
  printf '%s\n' "${selected[@]}"
  exit 0
fi

# This gate is wired to the Linux current-source CI job. Darwin's common
# 65532 KiB ceiling is intentionally not normalized into this Linux contract.
#
# Dual-module witnesses (gum+knowledge) need more than 64 MiB soft stack under
# current Madaros multi-module lower+codegen: measured 2026-07-20, 65536 KiB
# completes lower (final_fn_count 225) but fails to emit the ELF (wrapper then
# reports "run exited 1" / typecheck: failed); >= ~120000 KiB writes and runs
# DUAL_GUM_KNOWLEDGE_OK.
#
# 131072 was calibrated on THAT witness alone, and "leaves headroom" did not
# hold. This gate runs tests through `$SOUC_BIN run <file>`, which executes
# the program on top of the compiler's own frame rather than in a fresh
# process (the same in-process shape the retired scripts/dev/
# run_sio_test_suite_v1.sh used at its :153), so it needs
# materially more stack than `compile` + exec. Measured 2026-08-09 on a
# current-source Madaros, six tests, unanimous:
#
#   131072 KiB -> SIGSEGV (rc 139)   524288 KiB -> rc 0
#     gpu_kernel_lane_loop            epistemic_var_accumulator_slots
#     _diag_sobol                     approx_basic
#     arima_levinson_ar2              array_elem_field_store
#
# CI surfaced this as `run timed out after 30s` while the same binary faults in
# under a second locally, so it read as slowness and hid a crash.
# madaros_imported_call_arity_13_gate.sh already recorded the same lesson
# ("262144 KiB passes; 131072 fails. Default 512 MiB for headroom"); 524288 is
# what 16 other gates in scripts/ci already use.
stack_kb="${SOUNIO_MADAROS_CHANGED_TESTS_STACK_KB:-524288}"
[[ "$stack_kb" =~ ^[1-9][0-9]*$ && ${#stack_kb} -le 9 ]] \
  || fail "invalid_stack_kb"

if ! stack_soft_before="$(ulimit -S -s 2>/dev/null)"; then
  fail "stack_soft_limit_unavailable"
fi
if ! stack_hard_before="$(ulimit -H -s 2>/dev/null)"; then
  fail "stack_hard_limit_unavailable"
fi
[[ "$stack_soft_before" == "unlimited" || "$stack_soft_before" =~ ^[0-9]+$ ]] \
  || fail "invalid_stack_soft_limit"
[[ "$stack_hard_before" == "unlimited" || "$stack_hard_before" =~ ^[0-9]+$ ]] \
  || fail "invalid_stack_hard_limit"

if [[ "$stack_soft_before" != "unlimited" ]] && ((stack_soft_before < stack_kb)); then
  if [[ "$stack_hard_before" != "unlimited" ]] && ((stack_hard_before < stack_kb)); then
    echo "MADAROS_CHANGED_TESTS_STACK status=blocked scope=linux_ci requested_kb=$stack_kb soft_before_kb=$stack_soft_before hard_before_kb=$stack_hard_before soft_after_kb=$stack_soft_before hard_after_kb=$stack_hard_before"
    fail "stack_hard_limit_too_low"
  fi
  if ! ulimit -S -s "$stack_kb" 2>/dev/null; then
    stack_soft_after="$(ulimit -S -s 2>/dev/null || echo unavailable)"
    stack_hard_after="$(ulimit -H -s 2>/dev/null || echo unavailable)"
    echo "MADAROS_CHANGED_TESTS_STACK status=blocked scope=linux_ci requested_kb=$stack_kb soft_before_kb=$stack_soft_before hard_before_kb=$stack_hard_before soft_after_kb=$stack_soft_after hard_after_kb=$stack_hard_after"
    fail "stack_raise_failed"
  fi
fi

if ! stack_soft_after="$(ulimit -S -s 2>/dev/null)"; then
  fail "stack_soft_limit_after_unavailable"
fi
if ! stack_hard_after="$(ulimit -H -s 2>/dev/null)"; then
  fail "stack_hard_limit_after_unavailable"
fi
[[ "$stack_soft_after" == "unlimited" || "$stack_soft_after" =~ ^[0-9]+$ ]] \
  || fail "invalid_stack_soft_limit_after"
[[ "$stack_hard_after" == "unlimited" || "$stack_hard_after" =~ ^[0-9]+$ ]] \
  || fail "invalid_stack_hard_limit_after"
if [[ "$stack_soft_after" != "unlimited" ]] && ((stack_soft_after < stack_kb)); then
  echo "MADAROS_CHANGED_TESTS_STACK status=blocked scope=linux_ci requested_kb=$stack_kb soft_before_kb=$stack_soft_before hard_before_kb=$stack_hard_before soft_after_kb=$stack_soft_after hard_after_kb=$stack_hard_after"
  fail "stack_raise_not_effective"
fi
echo "MADAROS_CHANGED_TESTS_STACK status=ready scope=linux_ci requested_kb=$stack_kb soft_before_kb=$stack_soft_before hard_before_kb=$stack_hard_before soft_after_kb=$stack_soft_after hard_after_kb=$stack_hard_after"

if ((${#selected[@]} == 0)); then
  echo 'MADAROS_CHANGED_TESTS_SKIP reason=no_changed_requires_madaros_tests'
  exit 0
fi

[[ -n "$MADAROS_BIN" ]] || fail "missing_explicit_madaros_bin"
[[ "$MADAROS_BIN" == /* ]] || fail "madaros_bin_must_be_absolute"
[[ -f "$MADAROS_BIN" && -r "$MADAROS_BIN" && -x "$MADAROS_BIN" ]] \
  || fail "madaros_bin_not_executable"
[[ "$(head -c 2 "$MADAROS_BIN" 2>/dev/null)" != '#!' ]] || fail "madaros_bin_is_wrapper"

set +e
identity="$("$MADAROS_BIN" --version 2>&1)"
identity_rc=$?
set -e
[[ "$identity_rc" == "0" ]] || fail "madaros_identity_command_failed"
grep -Fxq 'Madaros v0.80.0 -- the Sounio self-hosted compiler' <<<"$identity" \
  || fail "madaros_identity_banner_missing"

set +e
negative_control="$("$MADAROS_BIN" --check \
  "$ROOT_DIR/tests/compile-fail/array_repeat_i8_binding_type_mismatch.sio" 2>&1)"
negative_control_rc=$?
set -e
[[ "$negative_control_rc" != "0" ]] || fail "madaros_negative_control_accepted"
grep -Fq 'this binding expects a different type' <<<"$negative_control" \
  || fail "madaros_negative_control_diagnostic_missing"

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/sounio-madaros-changed.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT
test_list="$work_dir/tests.txt"
printf '%s\n' "${selected[@]}" >"$test_list"

compiler_sha256="$(sha256sum "$MADAROS_BIN" | cut -d' ' -f1)"
echo "MADAROS_CHANGED_TESTS_START count=${#selected[@]} event=$EVENT_NAME compiler=$MADAROS_BIN compiler_sha256=$compiler_sha256"
printf 'test=%s\n' "${selected[@]}"

SOUNIO_MADAROS_AVAILABLE=1 \
SOUNIO_SOUC_RAW_MODE=modular \
SOUNIO_TEST_SOUC_BIN="$MADAROS_BIN" \
  bash "$ROOT_DIR/scripts/run_sio_test_suite.sh" \
    --test-list "$test_list" \
    --jobs "${SOUNIO_TEST_JOBS:-4}"

echo "MADAROS_CHANGED_TESTS_PASS count=${#selected[@]}"

# Changed-tests only sees the PR diff. Recheck every suite-visible
# requires:madaros known-failure so a compiler-only change cannot rot a tag
# the way the 240 imported/native 139s did.
bash "$ROOT_DIR/scripts/ci/known_failure_madaros_recheck.sh"
