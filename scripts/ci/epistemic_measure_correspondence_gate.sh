#!/usr/bin/env bash
# Guard the payload-type correspondence between Madaros measure/Knowledge
# typing and the value-carrying Lean epistemic calculus.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 1

CHECKER_SOURCE="self-hosted/check/check.sio"
MODEL_SOURCE="formal/lean4/EpistemicEffectsV2.lean"
CONSUMER_SOURCE="formal/lean4/EpistemicEffectsV2_measure_nat.lean"
V1_MUTANT="scripts/ci/fixtures/epistemic_measure_correspondence/v1_imports_measure_nat.lean"
KVALUE_CONSUMER_SOURCE="formal/lean4/EpistemicEffectsV2_kvalue_nat.lean"
KVALUE_V1_MUTANT="scripts/ci/fixtures/epistemic_measure_correspondence/v1_imports_kvalue_nat.lean"
INKRAW_CONSUMER_SOURCE="formal/lean4/EpistemicEffectsV2_invkraw_nat.lean"
INKRAW_V1_MUTANT="scripts/ci/fixtures/epistemic_measure_correspondence/v1_imports_invkraw_nat.lean"
INKRAW_MG_CONSUMER_SOURCE="formal/lean4/EpistemicEffectsV2_invkraw_mg.lean"
INKRAW_MG_V1_MUTANT="scripts/ci/fixtures/epistemic_measure_correspondence/v1_imports_invkraw_mg.lean"
ARTIFACT="${TMPDIR:-/tmp}/epistemic_measure_correspondence_gate.v1.json"
RUN_POSITIVE_CONTROLS=1
LEAN_CONSUME=0
LEAN_CONSUME_KVALUE=0
LEAN_CONSUME_INKRAW=0
LEAN_CONSUME_INKRAW_MG=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checker) CHECKER_SOURCE="${2:?missing path after --checker}"; shift 2 ;;
    --model) MODEL_SOURCE="${2:?missing path after --model}"; shift 2 ;;
    --artifact) ARTIFACT="${2:?missing path after --artifact}"; shift 2 ;;
    --control-child) RUN_POSITIVE_CONTROLS=0; shift ;;
    --lean-consume) LEAN_CONSUME=1; shift ;;
    --lean-consume-kvalue) LEAN_CONSUME_KVALUE=1; shift ;;
    --lean-consume-invkraw) LEAN_CONSUME_INKRAW=1; shift ;;
    --lean-consume-invkraw-mg) LEAN_CONSUME_INKRAW_MG=1; shift ;;
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
    'ty_knowledge\(v_ty,[[:space:]]*confidence\)' "$KNOWLEDGE_BODY"
  check_grep "knowledge_ctor_preserves_literal_confidence" \
    'let confidence = checker_enforce_knowledge_confidence_inplace\(c, e\.args, e\.span\)' "$KNOWLEDGE_BODY"
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

run_lean_consume() {
  local lean_dir="$ROOT_DIR/formal/lean4"
  if ! command -v lake >/dev/null 2>&1; then
    TOTAL=$((TOTAL + 1)); NOT_RUN=$((NOT_RUN + 1))
    record_failure "lake_missing"
    return
  fi

  check_grep "consumer_imports_v2" \
    '^import EpistemicEffectsV2$' "$CONSUMER_SOURCE"
  check_absent "consumer_must_not_import_v1_directly" \
    '^import EpistemicEffects$' "$CONSUMER_SOURCE"
  check_grep "consumer_names_the_inverted_witness" \
    '^theorem measure_nat_reduct_stays_know_nat$' "$CONSUMER_SOURCE"
  check_grep "v1_mutant_imports_v1" \
    '^import EpistemicEffects$' "$V1_MUTANT"
  check_absent "v1_mutant_must_not_import_v2" \
    '^import EpistemicEffectsV2$' "$V1_MUTANT"
  check_grep "v1_mutant_attempts_the_same_theorem" \
    '^theorem measure_nat_reduct_stays_know_nat$' "$V1_MUTANT"

  if ! (cd "$lean_dir" && lake build EpistemicEffects) \
      >"$WORK/lake-v1.log" 2>&1; then
    TOTAL=$((TOTAL + 1)); FAILED=$((FAILED + 1))
    record_failure "lake_build_v1_failed"
    sed 's/^/[lake-v1] /' "$WORK/lake-v1.log" >&2
    return
  fi

  # Arm 3 FIRST. If this mutant elaborates, arm 2 is measuring mention.
  TOTAL=$((TOTAL + 1))
  if (cd "$lean_dir" && lake env lean "$ROOT_DIR/$V1_MUTANT") \
      >"$WORK/v1-mutant.log" 2>&1; then
    FAILED=$((FAILED + 1))
    record_failure "positive_control_v1_import_measure_nat_was_not_rejected"
    sed 's/^/[v1-mutant] /' "$WORK/v1-mutant.log" >&2
    return
  fi
  PASSED=$((PASSED + 1))
  echo "[epistemic-correspondence] POSITIVE_CONTROL_FIRED: v1_imports_measure_nat rejected"
  sed 's/^/[v1-mutant] /' "$WORK/v1-mutant.log"

  TOTAL=$((TOTAL + 1))
  if ! (cd "$lean_dir" && lake build EpistemicEffectsV2_measure_nat) \
      >"$WORK/lake-consumer.log" 2>&1; then
    FAILED=$((FAILED + 1))
    record_failure "v2_measure_nat_consumer_failed_to_build"
    sed 's/^/[lake-consumer] /' "$WORK/lake-consumer.log" >&2
    return
  fi
  PASSED=$((PASSED + 1))
  echo "[epistemic-correspondence] V2_CONSUMED: EpistemicEffectsV2_measure_nat built"
}

run_lean_consume_kvalue() {
  local lean_dir="$ROOT_DIR/formal/lean4"
  if ! command -v lake >/dev/null 2>&1; then
    TOTAL=$((TOTAL + 1)); NOT_RUN=$((NOT_RUN + 1))
    record_failure "lake_missing"
    return
  fi

  check_grep "kvalue_consumer_imports_v2" \
    '^import EpistemicEffectsV2$' "$KVALUE_CONSUMER_SOURCE"
  check_absent "kvalue_consumer_must_not_import_v1_directly" \
    '^import EpistemicEffects$' "$KVALUE_CONSUMER_SOURCE"
  check_grep "kvalue_consumer_cites_preservation" \
    'preservation \(kvalue_nat_typed' "$KVALUE_CONSUMER_SOURCE"
  check_grep "kvalue_consumer_names_the_unwrap_witness" \
    '^theorem kvalue_nat_reduct_stays_nat$' "$KVALUE_CONSUMER_SOURCE"
  check_grep "kvalue_v1_mutant_imports_v1" \
    '^import EpistemicEffects$' "$KVALUE_V1_MUTANT"
  check_absent "kvalue_v1_mutant_must_not_import_v2" \
    '^import EpistemicEffectsV2$' "$KVALUE_V1_MUTANT"
  check_grep "kvalue_v1_mutant_attempts_the_same_theorem" \
    '^theorem kvalue_nat_reduct_stays_nat$' "$KVALUE_V1_MUTANT"

  if ! (cd "$lean_dir" && lake build EpistemicEffects) \
      >"$WORK/lake-v1-kvalue.log" 2>&1; then
    TOTAL=$((TOTAL + 1)); FAILED=$((FAILED + 1))
    record_failure "lake_build_v1_failed"
    sed 's/^/[lake-v1] /' "$WORK/lake-v1-kvalue.log" >&2
    return
  fi

  # Arm 3 FIRST. If this mutant elaborates, arm 2 is measuring mention.
  TOTAL=$((TOTAL + 1))
  if (cd "$lean_dir" && lake env lean "$ROOT_DIR/$KVALUE_V1_MUTANT") \
      >"$WORK/v1-kvalue-mutant.log" 2>&1; then
    FAILED=$((FAILED + 1))
    record_failure "positive_control_v1_import_kvalue_nat_was_not_rejected"
    sed 's/^/[v1-kvalue-mutant] /' "$WORK/v1-kvalue-mutant.log" >&2
    return
  fi
  PASSED=$((PASSED + 1))
  echo "[epistemic-correspondence] POSITIVE_CONTROL_FIRED: v1_imports_kvalue_nat rejected"
  sed 's/^/[v1-kvalue-mutant] /' "$WORK/v1-kvalue-mutant.log"

  TOTAL=$((TOTAL + 1))
  if ! (cd "$lean_dir" && lake build EpistemicEffectsV2_kvalue_nat) \
      >"$WORK/lake-kvalue-consumer.log" 2>&1; then
    FAILED=$((FAILED + 1))
    record_failure "v2_kvalue_nat_consumer_failed_to_build"
    sed 's/^/[lake-kvalue-consumer] /' "$WORK/lake-kvalue-consumer.log" >&2
    return
  fi
  PASSED=$((PASSED + 1))
  echo "[epistemic-correspondence] V2_CONSUMED: EpistemicEffectsV2_kvalue_nat built"
}

run_lean_consume_invkraw() {
  local lean_dir="$ROOT_DIR/formal/lean4"
  if ! command -v lake >/dev/null 2>&1; then
    TOTAL=$((TOTAL + 1)); NOT_RUN=$((NOT_RUN + 1))
    record_failure "lake_missing"
    return
  fi

  check_grep "invkraw_consumer_imports_v2" \
    '^import EpistemicEffectsV2$' "$INKRAW_CONSUMER_SOURCE"
  check_absent "invkraw_consumer_must_not_import_v1_directly" \
    '^import EpistemicEffects$' "$INKRAW_CONSUMER_SOURCE"
  check_grep "invkraw_consumer_cites_invKraw" \
    'invKraw hk rfl' "$INKRAW_CONSUMER_SOURCE"
  check_grep "invkraw_consumer_names_the_propagation_witness" \
    '^theorem kraw_nat_inverts_and_is_usable$' "$INKRAW_CONSUMER_SOURCE"
  check_grep "invkraw_v1_mutant_imports_v1" \
    '^import EpistemicEffects$' "$INKRAW_V1_MUTANT"
  check_absent "invkraw_v1_mutant_must_not_import_v2" \
    '^import EpistemicEffectsV2$' "$INKRAW_V1_MUTANT"
  check_grep "invkraw_v1_mutant_attempts_the_same_theorem" \
    '^theorem kraw_nat_inverts_and_is_usable$' "$INKRAW_V1_MUTANT"

  if ! (cd "$lean_dir" && lake build EpistemicEffects) \
      >"$WORK/lake-v1-invkraw.log" 2>&1; then
    TOTAL=$((TOTAL + 1)); FAILED=$((FAILED + 1))
    record_failure "lake_build_v1_failed"
    sed 's/^/[lake-v1] /' "$WORK/lake-v1-invkraw.log" >&2
    return
  fi

  # Arm 3 FIRST. If this mutant elaborates, arm 2 is measuring mention.
  TOTAL=$((TOTAL + 1))
  if (cd "$lean_dir" && lake env lean "$ROOT_DIR/$INKRAW_V1_MUTANT") \
      >"$WORK/v1-invkraw-mutant.log" 2>&1; then
    FAILED=$((FAILED + 1))
    record_failure "positive_control_v1_import_invkraw_nat_was_not_rejected"
    sed 's/^/[v1-invkraw-mutant] /' "$WORK/v1-invkraw-mutant.log" >&2
    return
  fi
  PASSED=$((PASSED + 1))
  echo "[epistemic-correspondence] POSITIVE_CONTROL_FIRED: v1_imports_invkraw_nat rejected"
  sed 's/^/[v1-invkraw-mutant] /' "$WORK/v1-invkraw-mutant.log"

  TOTAL=$((TOTAL + 1))
  if ! (cd "$lean_dir" && lake build EpistemicEffectsV2_invkraw_nat) \
      >"$WORK/lake-invkraw-consumer.log" 2>&1; then
    FAILED=$((FAILED + 1))
    record_failure "v2_invkraw_nat_consumer_failed_to_build"
    sed 's/^/[lake-invkraw-consumer] /' "$WORK/lake-invkraw-consumer.log" >&2
    return
  fi
  PASSED=$((PASSED + 1))
  echo "[epistemic-correspondence] V2_CONSUMED: EpistemicEffectsV2_invkraw_nat built"
}

run_lean_consume_invkraw_mg() {
  local lean_dir="$ROOT_DIR/formal/lean4"
  if ! command -v lake >/dev/null 2>&1; then
    TOTAL=$((TOTAL + 1)); NOT_RUN=$((NOT_RUN + 1))
    record_failure "lake_missing"
    return
  fi

  check_grep "invkraw_mg_consumer_imports_v2" \
    '^import EpistemicEffectsV2$' "$INKRAW_MG_CONSUMER_SOURCE"
  check_absent "invkraw_mg_consumer_must_not_import_v1_directly" \
    '^import EpistemicEffects$' "$INKRAW_MG_CONSUMER_SOURCE"
  check_grep "invkraw_mg_consumer_cites_invKraw" \
    'invKraw hk rfl' "$INKRAW_MG_CONSUMER_SOURCE"
  check_grep "invkraw_mg_consumer_names_the_propagation_witness" \
    '^theorem kraw_mg_inverts_and_is_usable$' "$INKRAW_MG_CONSUMER_SOURCE"
  check_grep "invkraw_mg_consumer_uses_tmg" \
    '\.tknow \.tmg' "$INKRAW_MG_CONSUMER_SOURCE"
  check_grep "invkraw_mg_v1_mutant_imports_v1" \
    '^import EpistemicEffects$' "$INKRAW_MG_V1_MUTANT"
  check_absent "invkraw_mg_v1_mutant_must_not_import_v2" \
    '^import EpistemicEffectsV2$' "$INKRAW_MG_V1_MUTANT"
  check_grep "invkraw_mg_v1_mutant_attempts_the_same_theorem" \
    '^theorem kraw_mg_inverts_and_is_usable$' "$INKRAW_MG_V1_MUTANT"
  check_grep "invkraw_mg_v1_mutant_uses_tmg" \
    '\.tknow \.tmg' "$INKRAW_MG_V1_MUTANT"

  if ! (cd "$lean_dir" && lake build EpistemicEffects) \
      >"$WORK/lake-v1-invkraw-mg.log" 2>&1; then
    TOTAL=$((TOTAL + 1)); FAILED=$((FAILED + 1))
    record_failure "lake_build_v1_failed"
    sed 's/^/[lake-v1] /' "$WORK/lake-v1-invkraw-mg.log" >&2
    return
  fi

  # Arm 3 FIRST. If this mutant elaborates, arm 2 is measuring mention.
  TOTAL=$((TOTAL + 1))
  if (cd "$lean_dir" && lake env lean "$ROOT_DIR/$INKRAW_MG_V1_MUTANT") \
      >"$WORK/v1-invkraw-mg-mutant.log" 2>&1; then
    FAILED=$((FAILED + 1))
    record_failure "positive_control_v1_import_invkraw_mg_was_not_rejected"
    sed 's/^/[v1-invkraw-mg-mutant] /' "$WORK/v1-invkraw-mg-mutant.log" >&2
    return
  fi
  PASSED=$((PASSED + 1))
  echo "[epistemic-correspondence] POSITIVE_CONTROL_FIRED: v1_imports_invkraw_mg rejected"
  sed 's/^/[v1-invkraw-mg-mutant] /' "$WORK/v1-invkraw-mg-mutant.log"

  TOTAL=$((TOTAL + 1))
  if ! (cd "$lean_dir" && lake build EpistemicEffectsV2_invkraw_mg) \
      >"$WORK/lake-invkraw-mg-consumer.log" 2>&1; then
    FAILED=$((FAILED + 1))
    record_failure "v2_invkraw_mg_consumer_failed_to_build"
    sed 's/^/[lake-invkraw-mg-consumer] /' "$WORK/lake-invkraw-mg-consumer.log" >&2
    return
  fi
  PASSED=$((PASSED + 1))
  echo "[epistemic-correspondence] V2_CONSUMED: EpistemicEffectsV2_invkraw_mg built"
}

if [[ "$LEAN_CONSUME" -eq 1 ]]; then
  if [[ ! -f "$CONSUMER_SOURCE" ]]; then
    TOTAL=$((TOTAL + 1)); NOT_RUN=$((NOT_RUN + 1))
    record_failure "consumer_source_missing:$CONSUMER_SOURCE"
  fi
  if [[ ! -f "$V1_MUTANT" ]]; then
    TOTAL=$((TOTAL + 1)); NOT_RUN=$((NOT_RUN + 1))
    record_failure "v1_mutant_missing:$V1_MUTANT"
  fi
  if [[ "$NOT_RUN" -eq 0 ]]; then
    run_lean_consume
  fi
fi

if [[ "$LEAN_CONSUME_KVALUE" -eq 1 && "$LEAN_CONSUME" -eq 0 ]]; then
  if [[ ! -f "$KVALUE_CONSUMER_SOURCE" ]]; then
    TOTAL=$((TOTAL + 1)); NOT_RUN=$((NOT_RUN + 1))
    record_failure "kvalue_consumer_source_missing:$KVALUE_CONSUMER_SOURCE"
  fi
  if [[ ! -f "$KVALUE_V1_MUTANT" ]]; then
    TOTAL=$((TOTAL + 1)); NOT_RUN=$((NOT_RUN + 1))
    record_failure "kvalue_v1_mutant_missing:$KVALUE_V1_MUTANT"
  fi
  if [[ "$NOT_RUN" -eq 0 ]]; then
    run_lean_consume_kvalue
  fi
fi

if [[ "$LEAN_CONSUME_INKRAW" -eq 1 && "$LEAN_CONSUME" -eq 0 && "$LEAN_CONSUME_KVALUE" -eq 0 && "$LEAN_CONSUME_INKRAW_MG" -eq 0 ]]; then
  if [[ ! -f "$INKRAW_CONSUMER_SOURCE" ]]; then
    TOTAL=$((TOTAL + 1)); NOT_RUN=$((NOT_RUN + 1))
    record_failure "invkraw_consumer_source_missing:$INKRAW_CONSUMER_SOURCE"
  fi
  if [[ ! -f "$INKRAW_V1_MUTANT" ]]; then
    TOTAL=$((TOTAL + 1)); NOT_RUN=$((NOT_RUN + 1))
    record_failure "invkraw_v1_mutant_missing:$INKRAW_V1_MUTANT"
  fi
  if [[ "$NOT_RUN" -eq 0 ]]; then
    run_lean_consume_invkraw
  fi
fi

if [[ "$LEAN_CONSUME_INKRAW_MG" -eq 1 && "$LEAN_CONSUME" -eq 0 && "$LEAN_CONSUME_KVALUE" -eq 0 && "$LEAN_CONSUME_INKRAW" -eq 0 ]]; then
  if [[ ! -f "$INKRAW_MG_CONSUMER_SOURCE" ]]; then
    TOTAL=$((TOTAL + 1)); NOT_RUN=$((NOT_RUN + 1))
    record_failure "invkraw_mg_consumer_source_missing:$INKRAW_MG_CONSUMER_SOURCE"
  fi
  if [[ ! -f "$INKRAW_MG_V1_MUTANT" ]]; then
    TOTAL=$((TOTAL + 1)); NOT_RUN=$((NOT_RUN + 1))
    record_failure "invkraw_mg_v1_mutant_missing:$INKRAW_MG_V1_MUTANT"
  fi
  if [[ "$NOT_RUN" -eq 0 ]]; then
    run_lean_consume_invkraw_mg
  fi
fi

if [[ "$RUN_POSITIVE_CONTROLS" -eq 1 && "$NOT_RUN" -eq 0 && "$LEAN_CONSUME" -eq 0 && "$LEAN_CONSUME_KVALUE" -eq 0 && "$LEAN_CONSUME_INKRAW" -eq 0 && "$LEAN_CONSUME_INKRAW_MG" -eq 0 ]]; then
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
