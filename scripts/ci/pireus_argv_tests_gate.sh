#!/usr/bin/env bash
# Pireus material tests that need argv are RUN, with their argv.
#
# tests/stdlib/hardware/test_pireus_*.sio.args names, one per line, the
# evidence and tamper fixtures a test ingests (get_arg(0..n)). The run-pass
# harness (scripts/dev/run_sio_test_suite.sh) has no argv channel: `souc run`
# takes exactly one argument. Run there, these tests can only print their usage
# line or return their own arg_count() refusal (rc 2 / rc 100) -- which is what
# the full suite measured on #2821. So the harness type-checks them
# (//@ check-only) and this gate executes them the way they were written:
# compile once, run the ELF with the sidecar's arguments, require rc 0.
#
# Selection is by sidecar, not by a list in this script: every
# test_pireus_*.sio with a .args file is run unless its header declares
# //@ known-failure (those need vendor material that is not in the repo; see
# #2820). A sidecar naming a file that does not exist is a failure, not a skip.
#
# Positive control first: the same ELF run with NO arguments must fail. If it
# passes, the test does not read its argv and this gate would measure nothing.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "pireus_argv_tests"

SOUC="${SOUC:-./bin/souc}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
RUN_TIMEOUT="${PIREUS_ARGV_RUN_TIMEOUT:-300}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

selected=0; passed=0; failed=0
for args in tests/stdlib/hardware/test_pireus_*.sio.args; do
  [[ -e "$args" ]] || continue
  src="${args%.args}"
  tag="$(basename "$src" .sio)"
  require_file "$src"
  if head -n 20 "$src" | grep -qE '^//@ *known-failure'; then
    printf 'PIREUS_ARGV_SKIP %s (declares //@ known-failure)\n' "$tag"
    continue
  fi
  selected=$((selected + 1))
  argv=()
  while IFS= read -r a || [[ -n "$a" ]]; do
    [[ -z "$a" ]] && continue
    if [[ ! -e "$a" ]]; then
      printf 'PIREUS_ARGV_FAIL %s: sidecar names a missing file: %s\n' "$tag" "$a" >&2
      failed=$((failed + 1)); continue 2
    fi
    argv+=("$a")
  done < "$args"
  elf="$work/$tag.elf"
  if ! timeout "$RUN_TIMEOUT" "$SOUC" compile "$src" -o "$elf" >"$work/$tag.compile.log" 2>&1 || [[ ! -s "$elf" ]]; then
    printf 'PIREUS_ARGV_FAIL %s: did not compile\n' "$tag" >&2
    tail -20 "$work/$tag.compile.log" >&2
    failed=$((failed + 1)); continue
  fi
  chmod +x "$elf"
  if timeout "$RUN_TIMEOUT" "$elf" >"$work/$tag.noargs.log" 2>&1; then
    printf 'PIREUS_ARGV_FAIL %s: CONTROL passed with no arguments -- it does not read its argv\n' "$tag" >&2
    failed=$((failed + 1)); continue
  fi
  rc=0
  timeout "$RUN_TIMEOUT" "$elf" "${argv[@]}" >"$work/$tag.run.log" 2>&1 || rc=$?
  if [[ "$rc" -ne 0 ]]; then
    printf 'PIREUS_ARGV_FAIL %s: rc=%s with %s argument(s)\n' "$tag" "$rc" "${#argv[@]}" >&2
    tail -20 "$work/$tag.run.log" >&2
    failed=$((failed + 1)); continue
  fi
  printf 'PIREUS_ARGV_PASS %s args=%s\n' "$tag" "${#argv[@]}"
  passed=$((passed + 1))
done

# Non-vacuity: #2821 brought six such tests. A glob that matches nothing
# reports zero failures and is not a pass.
require_min_count "$selected" 6 "argv-driven Pireus tests selected"
if [[ "$failed" -ne 0 ]]; then
  gate_fail "$failed of $selected argv-driven Pireus test(s) failed"
fi
gate_pass "$passed of $selected argv-driven Pireus tests ran with their sidecar arguments and returned 0"
