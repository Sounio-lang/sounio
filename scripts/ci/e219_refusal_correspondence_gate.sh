#!/usr/bin/env bash
# Guard the correspondence between SounioRefusalHonesty and the Madaros
# E250 / empty-stub path. A well-typed unimplemented call must refuse
# (not return the declared type) and must not compile to a stub that
# reads 0.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 1

CHECKER_SOURCE="self-hosted/check/check.sio"
CODEGEN_SOURCE="self-hosted/native/codegen_x86_linux.sio"
MODEL_SOURCE="formal/lean4/SounioRefusalHonesty.lean"
ARTIFACT="${TMPDIR:-/tmp}/e219_refusal_correspondence_gate.v1.json"
RUN_POSITIVE_CONTROLS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checker) CHECKER_SOURCE="${2:?missing path after --checker}"; shift 2 ;;
    --codegen) CODEGEN_SOURCE="${2:?missing path after --codegen}"; shift 2 ;;
    --model) MODEL_SOURCE="${2:?missing path after --model}"; shift 2 ;;
    --artifact) ARTIFACT="${2:?missing path after --artifact}"; shift 2 ;;
    --control-child) RUN_POSITIVE_CONTROLS=0; shift ;;
    *) echo "e219_refusal_correspondence_gate: unknown argument: $1" >&2; exit 2 ;;
  esac
done

TOTAL=0
PASSED=0
FAILED=0
NOT_RUN=0
FAILURES=""
WORK="$(mktemp -d "${TMPDIR:-/tmp}/e219-correspondence.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

record_failure() {
  local label="$1"
  if [[ -n "$FAILURES" ]]; then
    FAILURES="$FAILURES,$label"
  else
    FAILURES="$label"
  fi
  echo "[e219-correspondence] FAIL: $label" >&2
}

check_grep() {
  local label="$1"
  local pattern="$2"
  local path="$3"
  TOTAL=$((TOTAL + 1))
  if grep -qE "$pattern" "$path"; then
    PASSED=$((PASSED + 1))
  else
    FAILED=$((FAILED + 1))
    record_failure "$label"
  fi
}

check_count_ge() {
  local label="$1"
  local pattern="$2"
  local path="$3"
  local min="$4"
  TOTAL=$((TOTAL + 1))
  local n
  n="$(grep -cE "$pattern" "$path" || true)"
  if [[ "$n" -ge "$min" ]]; then
    PASSED=$((PASSED + 1))
  else
    FAILED=$((FAILED + 1))
    record_failure "$label:got_$n"
  fi
}

if [[ ! -f "$CHECKER_SOURCE" ]]; then
  TOTAL=$((TOTAL + 1)); NOT_RUN=$((NOT_RUN + 1))
  record_failure "checker_source_missing:$CHECKER_SOURCE"
fi
if [[ ! -f "$CODEGEN_SOURCE" ]]; then
  TOTAL=$((TOTAL + 1)); NOT_RUN=$((NOT_RUN + 1))
  record_failure "codegen_source_missing:$CODEGEN_SOURCE"
fi
if [[ ! -f "$MODEL_SOURCE" ]]; then
  TOTAL=$((TOTAL + 1)); NOT_RUN=$((NOT_RUN + 1))
  record_failure "model_source_missing:$MODEL_SOURCE"
fi

if [[ "$NOT_RUN" -eq 0 ]]; then
  # Checker: E250 still exists, and refuse infects the expression type.
  check_count_ge "e250_sites" \
    'name_is_native_backend_builtin.*\{|!name_is_native_backend_builtin' \
    "$CHECKER_SOURCE" 3
  check_count_ge "e250_reports" \
    ', 250, 0, 0, 0\)' "$CHECKER_SOURCE" 3
  check_count_ge "refuse_infects_ty_error" \
    'refused_unimplemented' "$CHECKER_SOURCE" 6
  check_grep "decl_is_not_call" \
    'if \(\*fd\)\.is_extern \{' "$CHECKER_SOURCE"

  # Codegen: empty unimplemented stubs trap instead of returning 0.
  check_grep "fabricate_predicate" \
    '^fn native_v2_empty_stub_would_fabricate\(' "$CODEGEN_SOURCE"
  check_grep "trap_emitter" \
    '^fn nc_emit_unimplemented_extern_trap_into\(' "$CODEGEN_SOURCE"
  check_grep "trap_is_ud2" \
    'nc_emit_byte\(nc, 0x0b\)' "$CODEGEN_SOURCE"
  check_grep "live_path_uses_predicate" \
    'native_v2_empty_stub_would_fabricate\(func\)' "$CODEGEN_SOURCE"
  check_grep "empty_stub_refusal_names_function" \
    'NATIVE_REFUSAL kind=empty_stub_ud2 fn=' "$CODEGEN_SOURCE"
  check_grep "empty_stub_refusal_has_stable_reason" \
    'reason=missing_lowered_body' "$CODEGEN_SOURCE"
  check_grep "empty_stub_refusal_summary" \
    'NATIVE_REFUSAL_SUMMARY empty_stub_ud2=' "$CODEGEN_SOURCE"

  # Lean model: the two relations disagree exactly when there is something
  # to refuse, and refuse infects add.
  check_grep "model_add_refuse_l" \
    'add_refuse_l' "$MODEL_SOURCE"
  check_grep "model_not_zero" \
    'unimplemented_call_not_zero' "$MODEL_SOURCE"
  check_grep "model_disagrees" \
    'unimplemented_disagrees' "$MODEL_SOURCE"
  check_grep "model_legacy_fabricates" \
    'legacy_fabricates' "$MODEL_SOURCE"
fi

if [[ "$RUN_POSITIVE_CONTROLS" -eq 1 && "$NOT_RUN" -eq 0 ]]; then
  CHECKER_MUTANT="scripts/ci/fixtures/e219_refusal_correspondence/checker_returns_declared_type.sio"
  CODEGEN_MUTANT="scripts/ci/fixtures/e219_refusal_correspondence/codegen_falls_through_empty_stub.sio"
  MODEL_MUTANT="scripts/ci/fixtures/e219_refusal_correspondence/model_no_add_refuse.lean"

  TOTAL=$((TOTAL + 1))
  if "$0" --control-child --checker "$CHECKER_MUTANT" --codegen "$CODEGEN_SOURCE" \
      --model "$MODEL_SOURCE" --artifact "$WORK/checker-mutant.json" \
      > "$WORK/checker-mutant.log" 2>&1; then
    FAILED=$((FAILED + 1))
    record_failure "positive_control_checker_returns_declared_type_was_not_rejected"
  else
    PASSED=$((PASSED + 1))
    echo "[e219-correspondence] POSITIVE_CONTROL_FIRED: checker_returns_declared_type rejected"
    sed 's/^/[checker-control] /' "$WORK/checker-mutant.log"
  fi

  TOTAL=$((TOTAL + 1))
  if "$0" --control-child --checker "$CHECKER_SOURCE" --codegen "$CODEGEN_MUTANT" \
      --model "$MODEL_SOURCE" --artifact "$WORK/codegen-mutant.json" \
      > "$WORK/codegen-mutant.log" 2>&1; then
    FAILED=$((FAILED + 1))
    record_failure "positive_control_codegen_falls_through_was_not_rejected"
  else
    PASSED=$((PASSED + 1))
    echo "[e219-correspondence] POSITIVE_CONTROL_FIRED: codegen_falls_through_empty_stub rejected"
    sed 's/^/[codegen-control] /' "$WORK/codegen-mutant.log"
  fi

  TOTAL=$((TOTAL + 1))
  if "$0" --control-child --checker "$CHECKER_SOURCE" --codegen "$CODEGEN_SOURCE" \
      --model "$MODEL_MUTANT" --artifact "$WORK/model-mutant.json" \
      > "$WORK/model-mutant.log" 2>&1; then
    FAILED=$((FAILED + 1))
    record_failure "positive_control_model_no_add_refuse_was_not_rejected"
  else
    PASSED=$((PASSED + 1))
    echo "[e219-correspondence] POSITIVE_CONTROL_FIRED: model_no_add_refuse rejected"
    sed 's/^/[model-control] /' "$WORK/model-mutant.log"
  fi
fi

STATUS="pass"
if [[ "$FAILED" -gt 0 || "$NOT_RUN" -gt 0 ]]; then
  STATUS="fail"
fi

mkdir -p "$(dirname "$ARTIFACT")"
cat > "$ARTIFACT" <<JSON
{
  "schema": "sounio.e219-refusal-correspondence-gate.v1",
  "status": "$STATUS",
  "checker_source": "$CHECKER_SOURCE",
  "codegen_source": "$CODEGEN_SOURCE",
  "model_source": "$MODEL_SOURCE",
  "metrics": {
    "total": $TOTAL,
    "passed": $PASSED,
    "failed": $FAILED,
    "not_run": $NOT_RUN
  },
  "failures_csv": "$FAILURES"
}
JSON

echo "e219_refusal_correspondence_gate: status=$STATUS total=$TOTAL passed=$PASSED failed=$FAILED not_run=$NOT_RUN artifact=$ARTIFACT"
if [[ "$STATUS" != "pass" ]]; then
  exit 1
fi
