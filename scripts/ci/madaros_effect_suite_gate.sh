#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MADAROS_BIN="${SOUNIO_MADAROS_EFFECT_SUITE_BIN:-${1:-}}"
SELECT_ONLY=0
SELFTEST=0

fail() {
  echo "MADAROS_EFFECT_SUITE_FAIL reason=$1" >&2
  exit 1
}

while (($#)); do
  case "$1" in
    --select-only)
      SELECT_ONLY=1
      shift
      ;;
    --selftest)
      SELFTEST=1
      shift
      ;;
    *)
      MADAROS_BIN="$1"
      shift
      ;;
  esac
done

select_effect_tests() {
  local root="$1"
  local inventory_file="$2"
  local path
  local base
  local lower
  local selected=()

  while IFS= read -r path || [[ -n "$path" ]]; do
    [[ -n "$path" ]] || continue
    [[ -f "$root/$path" ]] || continue
    base="$(basename "$path")"
    lower="${base,,}"
    case "$path" in
      tests/ui/effect/*.sio)
        selected+=("$path")
        continue
        ;;
    esac
    case "$lower" in
      effect*.sio|effects*.sio|fx[0-9]*.sio|handle*.sio|handler*.sio|*handler*.sio|*handle*.sio)
        selected+=("$path")
        ;;
      closure_effect*.sio|correlated_slot_identity_requires_effect.sio|epistemic_kernel_io_effect.sio)
        selected+=("$path")
        ;;
      io_effect*.sio|kernel_*_effect.sio|missing_effect.sio|mut_effect*.sio|read_file_effect*.sio)
        selected+=("$path")
        ;;
      sample_requires_sampling_effect.sio|scoped_effect*.sio|session_missing_effect.sio|user_effect*.sio)
        selected+=("$path")
        ;;
    esac
  done <"$inventory_file"

  printf '%s\n' "${selected[@]}" | sort -u
}

collect_suite_inventory() {
  local root="$1"
  local file

  for file in "$root"/tests/run-pass/*.sio; do
    [[ -f "$file" ]] && printf '%s\n' "${file#$root/}"
  done
  for file in "$root"/tests/compile-fail/*.sio; do
    [[ -f "$file" ]] && printf '%s\n' "${file#$root/}"
  done
  for file in "$root"/tests/ui/type/*.sio "$root"/tests/ui/effect/*.sio "$root"/tests/ui/ownership/*.sio "$root"/tests/ui/resolve/*.sio "$root"/tests/ui/module/*.sio "$root"/tests/ui/pattern/*.sio; do
    [[ -f "$file" ]] && printf '%s\n' "${file#$root/}"
  done
  for file in "$root"/tests/stdlib/*/test_*.sio; do
    [[ -f "$file" ]] && printf '%s\n' "${file#$root/}"
  done
  for file in "$root"/tests/gpu/*.sio; do
    [[ -f "$file" ]] && printf '%s\n' "${file#$root/}"
  done
  for file in "$root"/tests/self-hosted/*.sio; do
    [[ -f "$file" ]] && printf '%s\n' "${file#$root/}"
  done
  return 0
}
run_selftest() {
  local tmp
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/sounio-madaros-effect-suite-selftest.XXXXXX")"
  trap 'rm -rf "$tmp"' RETURN

  local repo="$tmp/repo"
  mkdir -p "$repo/scripts/ci" "$repo/scripts" "$repo/tests/compile-fail" "$repo/tests/run-pass" "$repo/tests/ui/effect"
  cp "$ROOT_DIR/scripts/ci/madaros_effect_suite_gate.sh" "$repo/scripts/ci/madaros_effect_suite_gate.sh"
  cat >"$repo/scripts/run_sio_test_suite.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--list-tests" ]]; then
  cat <<'LIST'
tests/compile-fail/effects_unhandled.sio
tests/compile-fail/user_effect_missing_in_caller_madaros.sio
tests/compile-fail/not_selected.sio
tests/run-pass/fx1_install.sio
tests/run-pass/handler_discharge.sio
tests/ui/effect/io_not_declared.sio
LIST
  exit 0
fi
[[ "${1:-}" == "--test-list" ]] || exit 64
cp "$2" "$SELFTEST_CAPTURE"
if [[ "${SOUNIO_MADAROS_AVAILABLE:-}" != "1" ]]; then
  echo "SELFTEST_FAIL bad SOUNIO_MADAROS_AVAILABLE=${SOUNIO_MADAROS_AVAILABLE:-<unset>}" >&2
  exit 1
fi
if [[ "${SOUNIO_TEST_SOUC_BIN:-}" != "$SELFTEST_MADAROS" ]]; then
  echo "SELFTEST_FAIL bad SOUNIO_TEST_SOUC_BIN=${SOUNIO_TEST_SOUC_BIN:-<unset>} expected=$SELFTEST_MADAROS" >&2
  exit 1
fi
if [[ "${SOUNIO_XPAS_FATAL:-}" != "1" ]]; then
  echo "SELFTEST_FAIL bad SOUNIO_XPAS_FATAL=${SOUNIO_XPAS_FATAL:-<unset>}" >&2
  exit 1
fi
if [[ "${SOUNIO_TEST_KNOWN_FAILURES_FILE:-}" != */tests/known_failures/madaros_effect_suite.txt ]]; then
  echo "SELFTEST_FAIL bad SOUNIO_TEST_KNOWN_FAILURES_FILE=${SOUNIO_TEST_KNOWN_FAILURES_FILE:-<unset>}" >&2
  exit 1
fi
echo "SELFTEST_RUNNER_PASS"
EOF
  chmod +x "$repo/scripts/run_sio_test_suite.sh"
  for file in \
    tests/compile-fail/effects_unhandled.sio \
    tests/compile-fail/user_effect_missing_in_caller_madaros.sio \
    tests/compile-fail/not_selected.sio \
    tests/run-pass/fx1_install.sio \
    tests/run-pass/handler_discharge.sio \
    tests/ui/effect/io_not_declared.sio; do
    mkdir -p "$repo/$(dirname "$file")"
    printf '//@ compile-fail\n' >"$repo/$file"
  done
  local mock="$tmp/mock-madaros"
  printf '\177ELFselftest-madaros-mock\n' >"$mock"
  chmod +x "$mock"
  local capture="$tmp/selected.list"
  local log="$tmp/selftest.log"

  if ! (cd "$repo" && export SELFTEST_CAPTURE="$capture" SELFTEST_MADAROS="$mock" SOUNIO_MADAROS_EFFECT_SUITE_BIN="$mock" && bash scripts/ci/madaros_effect_suite_gate.sh) >"$log" 2>&1; then
    cat "$log" >&2
    exit 1
  fi
  if ! grep -Fxq 'SELFTEST_RUNNER_PASS' "$log"; then
    echo "selftest runner did not report success" >&2
    cat "$log" >&2
    exit 1
  fi
  for expected in \
    tests/compile-fail/effects_unhandled.sio \
    tests/compile-fail/user_effect_missing_in_caller_madaros.sio \
    tests/run-pass/fx1_install.sio \
    tests/run-pass/handler_discharge.sio \
    tests/ui/effect/io_not_declared.sio; do
    if ! grep -Fxq "$expected" "$capture"; then
      echo "selftest did not select required fixture: $expected" >&2
      cat "$capture" >&2
      exit 1
    fi
  done
  if grep -Fxq 'tests/compile-fail/not_selected.sio' "$capture"; then
    echo "selftest selected an unrelated fixture" >&2
    exit 1
  fi
  echo "MADAROS_EFFECT_SUITE_SELFTEST_PASS"
}

if ((SELFTEST)); then
  run_selftest
  exit 0
fi

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/sounio-madaros-effect-suite.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT
inventory="$work_dir/inventory.list"
test_list="$work_dir/selected.list"

collect_suite_inventory "$ROOT_DIR" >"$inventory"
select_effect_tests "$ROOT_DIR" "$inventory" >"$test_list"

if [[ ! -s "$test_list" ]]; then
  fail "empty_selection"
fi
grep -Fxq 'tests/compile-fail/effects_unhandled.sio' "$test_list" || fail "effects_unhandled_not_selected"
grep -Fxq 'tests/compile-fail/user_effect_missing_in_caller_madaros.sio' "$test_list" || fail "user_effect_missing_madaros_not_selected"

if ((SELECT_ONLY)); then
  cat "$test_list"
  exit 0
fi

[[ -n "$MADAROS_BIN" ]] || fail "missing_madaros_bin"
[[ "$MADAROS_BIN" == /* ]] || fail "madaros_bin_must_be_absolute"
[[ -f "$MADAROS_BIN" && -r "$MADAROS_BIN" && -x "$MADAROS_BIN" ]] || fail "madaros_bin_not_executable"
[[ "$(head -c 2 "$MADAROS_BIN" 2>/dev/null)" != '#!' ]] || fail "madaros_bin_is_wrapper"

count="$(wc -l <"$test_list" | tr -d ' ')"
echo "MADAROS_EFFECT_SUITE_START count=$count compiler=$MADAROS_BIN"
sed 's/^/test=/' "$test_list"

SOUNIO_MADAROS_AVAILABLE=1 \
SOUNIO_SOUC_RAW_MODE=modular \
SOUNIO_TEST_SOUC_BIN="$MADAROS_BIN" \
SOUNIO_TEST_KNOWN_FAILURES_FILE="$ROOT_DIR/tests/known_failures/madaros_effect_suite.txt" \
SOUNIO_XPAS_FATAL=1 \
  bash "$ROOT_DIR/scripts/run_sio_test_suite.sh" \
    --test-list "$test_list" \
    --jobs "${SOUNIO_TEST_JOBS:-4}"

echo "MADAROS_EFFECT_SUITE_PASS count=$count"
