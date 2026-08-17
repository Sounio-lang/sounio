#!/usr/bin/env bash
# Guard the payload-type correspondence between Madaros measure/Knowledge
# typing and the value-carrying Lean epistemic calculus.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 1

CHECKER_SOURCE="self-hosted/check/check.sio"
MODEL_SOURCE="formal/lean4/EpistemicEffectsV2.lean"
ARTIFACT="${TMPDIR:-/tmp}/epistemic_measure_correspondence_gate.v1.json"
RUN_POSITIVE_CONTROLS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checker) CHECKER_SOURCE="${2:?missing path after --checker}"; shift 2 ;;
    --model) MODEL_SOURCE="${2:?missing path after --model}"; shift 2 ;;
    --artifact) ARTIFACT="${2:?missing path after --artifact}"; shift 2 ;;
    --control-child) RUN_POSITIVE_CONTROLS=0; shift ;;
    *) echo "epistemic_measure_correspondence_gate: unknown argument: $1" >&2; exit 2 ;;
  esac
done

TOTAL=0
PASSED=0
FAILED=0
NOT_RUN=0
FAILURES=""
WORK="$(mktemp -d "${TMPDIR:-/tmp}/epistemic-correspondence.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

record_failure() {
  local label="$1"
  if [[ -n "$FAILURES" ]]; then
    FAILURES="$FAILURES,$label"
  else
    FAILURES="$label"
  fi
  echo "[epistemic-correspondence] FAIL: $label" >&2
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

check_absent() {
  local label="$1"
  local pattern="$2"
  local path="$3"
  TOTAL=$((TOTAL + 1))
  if grep -qE "$pattern" "$path"; then
    FAILED=$((FAILED + 1))
    record_failure "$label"
  else
    PASSED=$((PASSED + 1))
  fi
}

extract_checker_function() {
  local fn_name="$1"
  local source="$2"
  local output="$3"
  awk -v signature="fn ${fn_name}(" '
    index($0, signature) == 1 { inside = 1 }
    inside { print }
    inside && /^}$/ { exit }
  ' "$source" > "$output"
}

extract_t_kraw_rule() {
  local source="$1"
  local output="$2"
  awk '
    /^[[:space:]]*\| t_kraw[[:space:]]/ { inside = 1 }
    inside && /^[[:space:]]*\| t_sub[[:space:]]/ { exit }
    inside { print }
  ' "$source" > "$output"
}

if [[ ! -f "$CHECKER_SOURCE" ]]; then
  TOTAL=$((TOTAL + 1)); NOT_RUN=$((NOT_RUN + 1))
  record_failure "checker_source_missing:$CHECKER_SOURCE"
fi
if [[ ! -f "$MODEL_SOURCE" ]]; then
  TOTAL=$((TOTAL + 1)); NOT_RUN=$((NOT_RUN + 1))
  record_failure "model_source_missing:$MODEL_SOURCE"
fi

if [[ "$NOT_RUN" -eq 0 ]]; then
  MEASURE_BODY="$WORK/measure.body"
  KNOWLEDGE_BODY="$WORK/knowledge.body"
  KRAW_RULE="$WORK/t_kraw.rule"
  extract_checker_function "checker_check_measure_expr_inplace" "$CHECKER_SOURCE" "$MEASURE_BODY"
  extract_checker_function "checker_check_knowledge_ctor_expr_inplace" "$CHECKER_SOURCE" "$KNOWLEDGE_BODY"
  extract_t_kraw_rule "$MODEL_SOURCE" "$KRAW_RULE"

  check_grep "measure_handler_extracted" \
    '^fn checker_check_measure_expr_inplace\(' "$MEASURE_BODY"
  check_grep "measure_checks_first_argument" \
    'let v_ty = checker_check_opt_expr_inplace\(c, first_arg\)' "$MEASURE_BODY"
  check_grep "measure_propagates_argument_type" \
    'ty_knowledge\(v_ty,[[:space:]]*0\.0[[:space:]]*-[[:space:]]*1\.0\)' "$MEASURE_BODY"
  check_absent "measure_must_not_pin_scalar_type" \
    'ty_knowledge\((ty_f64\(\)|ty_real\(\)|ty_real|TypeEntry::Real|\.treal)' "$MEASURE_BODY"

  check_grep "knowledge_ctor_handler_extracted" \
    '^fn checker_check_knowledge_ctor_expr_inplace\(' "$KNOWLEDGE_BODY"
  check_grep "knowledge_ctor_checks_first_argument" \
    'let v_ty = checker_check_opt_expr_inplace\(c, first_arg\)' "$KNOWLEDGE_BODY"
  check_grep "knowledge_ctor_propagates_argument_type" \
    'ty_knowledge\(v_ty,[[:space:]]*0\.0[[:space:]]*-[[:space:]]*1\.0\)' "$KNOWLEDGE_BODY"
  check_absent "knowledge_ctor_must_not_pin_scalar_type" \
    'ty_knowledge\((ty_f64\(\)|ty_real\(\)|ty_real|TypeEntry::Real|\.treal)' "$KNOWLEDGE_BODY"

  check_grep "t_kraw_rule_extracted" \
    '^[[:space:]]*\| t_kraw[[:space:]]' "$KRAW_RULE"
  check_grep "t_kraw_binds_payload_type_variable" \
    't_kraw[[:space:]]*:[[:space:]]*∀ Γ T v m' "$KRAW_RULE"
  check_grep "t_kraw_types_payload_at_variable" \
    'HasTy Γ v T emptyE' "$KRAW_RULE"
  check_grep "t_kraw_preserves_payload_type" \
    'HasTy Γ \(\.kraw v m\) \(\.tknow T\) emptyE' "$KRAW_RULE"
  check_absent "t_kraw_must_not_pin_real" \
    '\.tknow[[:space:]]+\.treal' "$KRAW_RULE"
fi

if [[ "$RUN_POSITIVE_CONTROLS" -eq 1 && "$NOT_RUN" -eq 0 ]]; then
  CHECKER_MUTANT="scripts/ci/fixtures/epistemic_measure_correspondence/checker_fixed_f64.sio"
  MODEL_MUTANT="scripts/ci/fixtures/epistemic_measure_correspondence/model_fixed_treal.lean"

  TOTAL=$((TOTAL + 1))
  if "$0" --control-child --checker "$CHECKER_MUTANT" --model "$MODEL_SOURCE" \
      --artifact "$WORK/checker-mutant.json" > "$WORK/checker-mutant.log" 2>&1; then
    FAILED=$((FAILED + 1))
    record_failure "positive_control_checker_fixed_f64_was_not_rejected"
  else
    PASSED=$((PASSED + 1))
    echo "[epistemic-correspondence] POSITIVE_CONTROL_FIRED: checker_fixed_f64 rejected"
    sed 's/^/[checker-control] /' "$WORK/checker-mutant.log"
  fi

  TOTAL=$((TOTAL + 1))
  if "$0" --control-child --checker "$CHECKER_SOURCE" --model "$MODEL_MUTANT" \
      --artifact "$WORK/model-mutant.json" > "$WORK/model-mutant.log" 2>&1; then
    FAILED=$((FAILED + 1))
    record_failure "positive_control_model_fixed_treal_was_not_rejected"
  else
    PASSED=$((PASSED + 1))
    echo "[epistemic-correspondence] POSITIVE_CONTROL_FIRED: model_fixed_treal rejected"
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
  "schema": "sounio.epistemic-measure-correspondence-gate.v1",
  "status": "$STATUS",
  "checker_source": "$CHECKER_SOURCE",
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

echo "epistemic_measure_correspondence_gate: status=$STATUS total=$TOTAL passed=$PASSED failed=$FAILED not_run=$NOT_RUN artifact=$ARTIFACT"
if [[ "$STATUS" != "pass" ]]; then
  exit 1
fi
