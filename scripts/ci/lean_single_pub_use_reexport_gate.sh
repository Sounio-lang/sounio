#!/usr/bin/env bash
# Contract gate for named function re-exports in braced `pub use` lists.
# Types, constants, aliases, globs, renames, and chained re-exports are out of scope.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KEEP_WORK="${SOUNIO_LEAN_SINGLE_PUB_USE_GATE_KEEP:-0}"

fail() {
  echo "[lean-single-pub-use] FAIL: $*" >&2
  exit 1
}

if [[ -n "${SOUNIO_LEAN_SINGLE_PUB_USE_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_LEAN_SINGLE_PUB_USE_GATE_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir "$WORK" || fail "could not create gate directory: $WORK"
else
  WORK="$(mktemp -d /tmp/sounio-lean-single-pub-use.XXXXXX)"
fi

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

semantic_unknown_rejection() {
  local rc="$1"
  local log="$2"
  local symbol="$3"

  [[ "$rc" -eq 1 ]] || return 1
  grep -Eq "^error: unknown identifier \`$symbol\` at " "$log" || return 1
  grep -Fxq 'typecheck: failed' "$log" || return 1
  if grep -Eiq 'segmentation fault|core dumped|terminated by signal|fatal:|bus error|illegal instruction' "$log"; then
    return 1
  fi
  return 0
}

adversarial_log="$WORK/adversarial-rc139.log"
printf '%s\n' \
  'error: unknown identifier `synthetic_value` at <main>:1' \
  'typecheck: failed' \
  'Segmentation fault (core dumped)' >"$adversarial_log"
if semantic_unknown_rejection 139 "$adversarial_log" synthetic_value; then
  fail "adversarial expected-diagnostic plus rc=139 was accepted"
fi

valid_semantic_log="$WORK/valid-semantic-rc1.log"
printf '%s\n' \
  'error: unknown identifier `synthetic_value` at <main>:1' \
  'typecheck: failed' >"$valid_semantic_log"
semantic_unknown_rejection 1 "$valid_semantic_log" synthetic_value || {
  fail "valid semantic rc=1 diagnostic was rejected"
}

fatal_log="$WORK/adversarial-fatal-rc1.log"
printf '%s\n' \
  'error: unknown identifier `synthetic_value` at <main>:1' \
  'typecheck: failed' \
  'fatal: synthetic compiler abort' >"$fatal_log"
if semantic_unknown_rejection 1 "$fatal_log" synthetic_value; then
  fail "fatal diagnostic with rc=1 was accepted as semantic rejection"
fi

echo "[lean-single-pub-use] PASS: semantic_rc1_accepted=1 adversarial_rc139_rejected=1 fatal_rc1_rejected=1"

if [[ "${SOUNIO_LEAN_SINGLE_PUB_USE_GATE_SELF_TEST_ONLY:-0}" == "1" ]]; then
  exit 0
fi

COMPILER="${SOUNIO_LEAN_SINGLE_PUB_USE_GATE_BIN:-$WORK/lean_single}"
if [[ -z "${SOUNIO_LEAN_SINGLE_PUB_USE_GATE_BIN:-}" ]]; then
  SEED="${SOUNIO_LEAN_SINGLE_PUB_USE_GATE_SEED:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
  [[ -x "$SEED" ]] || fail "lean_single seed is missing or not executable: $SEED"
  if ! "$ROOT_DIR/scripts/dev/souc-build-lock.sh" \
      "$SEED" "$ROOT_DIR/self-hosted/compiler/lean_single.sio" "$COMPILER" \
      >"$WORK/build.log" 2>&1; then
    tail -n 80 "$WORK/build.log" >&2 || true
    fail "current-source lean_single build failed"
  fi
  chmod +x "$COMPILER"
fi
[[ -x "$COMPILER" ]] || fail "lean_single candidate is missing or not executable: $COMPILER"

compile_case() {
  local name="$1"
  local source="$2"
  local elf="$WORK/$name.elf"
  local log="$WORK/$name.compile.log"

  CASE_RC=0
  set +e
  SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" "$COMPILER" "$source" "$elf" >"$log" 2>&1
  CASE_RC=$?
  set -e
}

run_case() {
  local name="$1"
  local elf="$WORK/$name.elf"
  local log="$WORK/$name.run.log"

  chmod +x "$elf"
  CASE_RC=0
  set +e
  "$elf" >"$log" 2>&1
  CASE_RC=$?
  set -e
}

compile_case direct "$ROOT_DIR/tests/run-pass/pub_use_reexport_direct_control.sio"
[[ "$CASE_RC" -eq 0 ]] || {
  cat "$WORK/direct.compile.log" >&2
  fail "direct-import positive control did not compile"
}
run_case direct
[[ "$CASE_RC" -eq 0 ]] || {
  cat "$WORK/direct.run.log" >&2
  fail "direct-import positive control did not execute"
}
grep -Fxq 'PASS pub_use_reexport_direct_control' "$WORK/direct.run.log" || {
  cat "$WORK/direct.run.log" >&2
  fail "direct-import exact PASS marker missing"
}

facade_forwarding=0
compile_case facade "$ROOT_DIR/tests/compiler/pub_use_reexport/public_consumer.sio"
if [[ "$CASE_RC" -eq 0 ]]; then
  run_case facade
  [[ "$CASE_RC" -eq 0 ]] || {
    cat "$WORK/facade.run.log" >&2
    fail "facade compiled but did not execute"
  }
  grep -Fxq 'PASS pub_use_named_function_reexport' "$WORK/facade.run.log" || {
    cat "$WORK/facade.run.log" >&2
    fail "facade exact PASS marker missing"
  }
  facade_forwarding=1
elif ! semantic_unknown_rejection "$CASE_RC" "$WORK/facade.compile.log" public_route_value; then
  cat "$WORK/facade.compile.log" >&2
  fail "facade failed for a reason other than absent forwarding"
fi

compile_case missing "$ROOT_DIR/tests/compiler/pub_use_reexport/missing_consumer.sio"
[[ "$CASE_RC" -ne 0 ]] || fail "missing re-exported symbol compiled unexpectedly"
semantic_unknown_rejection "$CASE_RC" "$WORK/missing.compile.log" missing_route_value || {
  cat "$WORK/missing.compile.log" >&2
  fail "missing-symbol diagnostic was not preserved"
}

selective_reexport=0
compile_case not-reexported "$ROOT_DIR/tests/compiler/pub_use_reexport/not_reexported_consumer.sio"
if [[ "$CASE_RC" -ne 0 ]]; then
  semantic_unknown_rejection "$CASE_RC" "$WORK/not-reexported.compile.log" not_reexported_value || {
    cat "$WORK/not-reexported.compile.log" >&2
    fail "non-reexported witness failed for an unrelated reason"
  }
  selective_reexport=1
fi

# This is a separate known residual: private imports currently globalize public
# declarations. It is reported in the receipt but does not substitute for the
# selected public re-export contract above.
private_import_isolated=0
compile_case private "$ROOT_DIR/tests/compiler/pub_use_reexport/private_consumer.sio"
if [[ "$CASE_RC" -eq 0 ]]; then
  echo "[lean-single-pub-use] XFAIL BLK-20260713-lean-single-private-import-visibility: private use still globalizes a public leaf"
elif semantic_unknown_rejection "$CASE_RC" "$WORK/private.compile.log" private_route_value; then
  private_import_isolated=1
  echo "[lean-single-pub-use] PASS: private use no longer re-exports its public leaf"
else
  cat "$WORK/private.compile.log" >&2
  fail "private-use witness failed for a reason other than route visibility"
fi

echo "[lean-single-pub-use] receipt scope=named_function_reexports direct_import=PASS missing_symbol=REJECTED facade_forwarding=$facade_forwarding selective_reexport=$selective_reexport private_import_isolated=$private_import_isolated"

if [[ "$facade_forwarding" -eq 0 ]]; then
  echo "[lean-single-pub-use] BLOCKED BLK-20260713-lean-single-import-visibility: public forwarding is absent" >&2
  exit 1
fi
if [[ "$selective_reexport" -eq 0 ]]; then
  echo "[lean-single-pub-use] BLOCKED BLK-20260713-lean-single-import-visibility: forwarding exposes the whole leaf instead of the selected export" >&2
  exit 1
fi
echo "[lean-single-pub-use] PASS: named function facade forwarding is selective and missing symbols remain rejected"
