#!/usr/bin/env bash
# Classify #854 without weakening real cross-module privacy diagnostics.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MADAROS="${SOUNIO_MADAROS_VISIBILITY_CONTEXT_BIN:-$ROOT_DIR/bin/madaros}"
RAW_MADAROS="${MADAROS_RAW_BIN:-}"
KEEP_WORK="${SOUNIO_MADAROS_VISIBILITY_CONTEXT_KEEP:-0}"
EXPECT="${SOUNIO_MADAROS_VISIBILITY_CONTEXT_EXPECT:-classify}"

fail() {
  echo "[madaros-visibility-context] FAIL: $*" >&2
  exit 1
}

if [[ -n "${SOUNIO_MADAROS_VISIBILITY_CONTEXT_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_VISIBILITY_CONTEXT_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir "$WORK" || fail "could not create gate directory: $WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-visibility-context.XXXXXX)"
fi

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

case "$EXPECT" in
  classify|baseline|resolved) ;;
  *) fail "invalid SOUNIO_MADAROS_VISIBILITY_CONTEXT_EXPECT=$EXPECT" ;;
esac

[[ -x "$MADAROS" ]] || fail "Madaros wrapper is missing or not executable: $MADAROS"
[[ -n "$RAW_MADAROS" ]] || fail "MADAROS_RAW_BIN must name an explicit current-source Madaros ELF"
[[ -x "$RAW_MADAROS" ]] || fail "explicit current-source Madaros is missing or not executable: $RAW_MADAROS"

is_fatal_log() {
  grep -Eiq 'segmentation fault|core dumped|terminated by signal|fatal:|bus error|illegal instruction' "$1"
}

count_marker() {
  local marker="$1"
  local log="$2"
  grep -Fc "$marker" "$log" || true
}

classify_context_check() {
  local label="$1"
  local source="$2"
  local expected_e175="$3"
  local log="$WORK/$label.check.log"
  local rc=0

  set +e
  MADAROS_RAW_BIN="$RAW_MADAROS" "$MADAROS" check "$source" >"$log" 2>&1
  rc=$?
  set -e

  is_fatal_log "$log" && {
    cat "$log" >&2
    fail "$label produced a fatal compiler/runtime log"
  }

  local e175_count
  local message_count
  local diagnostic_count
  e175_count="$(count_marker 'error[E175' "$log")"
  message_count="$(count_marker 'function is private in its defining module' "$log")"
  diagnostic_count="$(count_marker 'error[E' "$log")"

  if [[ "$rc" -eq 0 ]]; then
    [[ "$diagnostic_count" -eq 0 ]] || fail "$label returned 0 while still emitting compiler diagnostics"
    [[ "$message_count" -eq 0 ]] || fail "$label returned 0 while still emitting the E175 message"
    grep -Fq 'check: OK' "$log" || {
      cat "$log" >&2
      fail "$label returned 0 without the check success marker"
    }
    CASE_STATE="resolved"
    return 0
  fi

  if [[ "$rc" -eq 1 && "$diagnostic_count" -eq "$expected_e175" && "$e175_count" -eq "$expected_e175" && "$message_count" -eq "$expected_e175" ]]; then
    grep -Fq 'run_check_mode: verdict=1' "$log" || {
      cat "$log" >&2
      fail "$label emitted E175 without the visibility-preflight verdict"
    }
    CASE_STATE="baseline"
    return 0
  fi

  cat "$log" >&2
  fail "$label was neither the exact #854 baseline nor the contextual-lookup result (rc=$rc diagnostics=$diagnostic_count E175=$e175_count message=$message_count)"
}

expect_context_runtime() {
  local label="$1"
  local source="$2"
  local pass_marker="$3"
  local log="$WORK/$label.run.log"
  local rc=0

  set +e
  MADAROS_RAW_BIN="$RAW_MADAROS" "$MADAROS" run "$source" >"$log" 2>&1
  rc=$?
  set -e

  is_fatal_log "$log" && {
    cat "$log" >&2
    fail "$label produced a fatal compiler/runtime log after clean check"
  }
  [[ "$rc" -eq 0 ]] || {
    cat "$log" >&2
    fail "$label checked clean but runtime returned rc=$rc"
  }
  [[ "$(count_marker 'error[E' "$log")" -eq 0 ]] || {
    cat "$log" >&2
    fail "$label runtime emitted compiler diagnostics after clean check"
  }
  grep -Fxq "$pass_marker" "$log" || {
    cat "$log" >&2
    fail "$label runtime returned 0 without its exact marker"
  }
}

expect_context_check_only() {
  local label="$1"
  local source="$2"
  local log="$WORK/$label.check-only.log"
  local rc=0

  set +e
  MADAROS_RAW_BIN="$RAW_MADAROS" "$MADAROS" check "$source" >"$log" 2>&1
  rc=$?
  set -e

  is_fatal_log "$log" && {
    cat "$log" >&2
    fail "$label produced a fatal compiler log"
  }
  [[ "$rc" -eq 0 ]] || {
    cat "$log" >&2
    fail "$label must pass its check-only aggregate witness (rc=$rc)"
  }
  [[ "$(count_marker 'error[E' "$log")" -eq 0 ]] || {
    cat "$log" >&2
    fail "$label returned 0 while emitting compiler diagnostics"
  }
  grep -Fq 'check: OK' "$log" || {
    cat "$log" >&2
    fail "$label returned 0 without the check success marker"
  }
}

expect_private_rejection() {
  local label="$1"
  local source="$2"
  local code="$3"
  local message="$4"
  local log="$WORK/$label.log"
  local rc=0

  set +e
  MADAROS_RAW_BIN="$RAW_MADAROS" "$MADAROS" check "$source" >"$log" 2>&1
  rc=$?
  set -e

  is_fatal_log "$log" && {
    cat "$log" >&2
    fail "$label produced a fatal compiler log"
  }
  [[ "$rc" -eq 1 ]] || {
    cat "$log" >&2
    fail "$label must reject with rc=1, got rc=$rc"
  }
  [[ "$(count_marker 'error[E' "$log")" -eq 1 ]] || {
    cat "$log" >&2
    fail "$label must emit exactly one compiler diagnostic"
  }
  [[ "$(count_marker "error[$code" "$log")" -eq 1 ]] || {
    cat "$log" >&2
    fail "$label must emit exactly one $code diagnostic"
  }
  [[ "$(count_marker "$message" "$log")" -eq 1 ]] || {
    cat "$log" >&2
    fail "$label must emit exactly one canonical privacy message"
  }
}

expect_ambiguous_rejection() {
  local label="$1"
  local source="$2"
  local log="$WORK/$label.log"
  local rc=0

  set +e
  MADAROS_RAW_BIN="$RAW_MADAROS" "$MADAROS" check "$source" >"$log" 2>&1
  rc=$?
  set -e

  is_fatal_log "$log" && {
    cat "$log" >&2
    fail "$label produced a fatal compiler log"
  }
  [[ "$rc" -eq 1 ]] || {
    cat "$log" >&2
    fail "$label must fail closed instead of selecting a candidate (rc=$rc)"
  }
  [[ "$(count_marker 'error[E' "$log")" -eq 1 ]] || {
    cat "$log" >&2
    fail "$label must emit exactly one compiler diagnostic"
  }
  [[ "$(count_marker 'error[E137' "$log")" -eq 1 ]] || {
    cat "$log" >&2
    fail "$label must remain unresolved with exactly one E137 diagnostic"
  }
  [[ "$(count_marker 'use of undeclared variable' "$log")" -eq 1 ]] || {
    cat "$log" >&2
    fail "$label must emit the canonical unresolved-name message"
  }
}

FIXTURES="$ROOT_DIR/tests/compiler/madaros_visibility_context"
classify_context_check single "$FIXTURES/duplicate_private_single_main.sio" 1
single_state="$CASE_STATE"
classify_context_check matrix18 "$FIXTURES/duplicate_private_18_main.sio" 18
matrix_state="$CASE_STATE"

[[ "$single_state" == "$matrix_state" ]] || {
  fail "partial contextual lookup: single=$single_state matrix18=$matrix_state"
}

runtime_state="not-run-baseline"
ambiguous_global="not-run-baseline"
ambiguous_bare_variant="not-run-baseline"
duplicate_private_struct="not-run-baseline"
duplicate_private_enum="not-run-baseline"
variant_call_form="not-run-baseline"
struct_variant="not-run-baseline"
cross_kind="not-run-baseline"
bare_variant="not-run-baseline"
true_private_struct_variant="not-run-baseline"
if [[ "$single_state" == resolved ]]; then
  expect_ambiguous_rejection ambiguous-global "$FIXTURES/ambiguous_public_main.sio"
  ambiguous_global="E137"
  expect_ambiguous_rejection ambiguous-bare-variant \
    "$FIXTURES/ambiguous_bare_variant_main.sio"
  ambiguous_bare_variant="E137"
  expect_context_check_only duplicate-private-struct \
    "$FIXTURES/duplicate_private_struct_main.sio"
  duplicate_private_struct="pass"
  expect_context_check_only duplicate-private-enum \
    "$FIXTURES/duplicate_private_enum_main.sio"
  duplicate_private_enum="pass"
  variant_call_form="pass"
  struct_variant="pass"
  cross_kind="pass"
  bare_variant="pass"
  expect_private_rejection private-enum-structured \
    "$FIXTURES/private_structured_enum_main.sio" E177 \
    'enum constructor is private in its defining module'
  true_private_struct_variant="E177"
  expect_context_runtime single "$FIXTURES/duplicate_private_single_main.sio" \
    'PASS duplicate_private_single_context'
  expect_context_runtime matrix18 "$FIXTURES/duplicate_private_18_main.sio" \
    'PASS duplicate_private_18_context'
  runtime_state="pass"
fi

expect_private_rejection private-fn \
  "$ROOT_DIR/tests/multimodule/visibility_fn_private_main.sio" E175 \
  'function is private in its defining module'
expect_private_rejection private-struct \
  "$ROOT_DIR/tests/multimodule/visibility_struct_private_main.sio" E176 \
  'struct constructor is private in its defining module'
expect_private_rejection private-enum \
  "$ROOT_DIR/tests/multimodule/visibility_enum_private_main.sio" E177 \
  'enum constructor is private in its defining module'

echo "[madaros-visibility-context] receipt issue=854 context_state=$single_state runtime_state=$runtime_state ambiguous_global=$ambiguous_global ambiguous_bare_variant=$ambiguous_bare_variant aggregate_witness_mode=check-only duplicate_private_struct=$duplicate_private_struct duplicate_private_enum=$duplicate_private_enum variant_call_form=$variant_call_form struct_variant=$struct_variant cross_kind=$cross_kind bare_variant=$bare_variant single_e175=$([[ "$single_state" == baseline ]] && echo 1 || echo 0) matrix_e175=$([[ "$matrix_state" == baseline ]] && echo 18 || echo 0) true_private_fn=E175 true_private_struct=E176 true_private_enum=E177 true_private_struct_variant=$true_private_struct_variant"

if [[ "$EXPECT" == baseline && "$single_state" != baseline ]]; then
  fail "expected the pinned baseline, got $single_state"
fi
if [[ "$EXPECT" == resolved && "$single_state" != resolved ]]; then
  echo '[madaros-visibility-context] BLOCKED BLK-20260713-MADAROS-EISA-E5-VISIBILITY-E175: contextual module binding lookup is not implemented' >&2
  exit 1
fi

if [[ "$single_state" == baseline ]]; then
  echo '[madaros-visibility-context] KNOWN_BLOCKER: name-only lookup selects a private same-name binding from the wrong module'
else
  echo '[madaros-visibility-context] PASS: contextual lookup preserves same-name private bindings and real private access remains rejected'
fi
