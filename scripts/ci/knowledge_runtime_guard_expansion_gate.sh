#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

EXPANDER="scripts/ontology/expand_knowledge_runtime_guards.sh"
CURRENT_SOUC="$TMP_DIR/sounio-current-source"
BUILD_LOG="$TMP_DIR/build-current-source.log"
POSITIVE="tests/frontend/knowledge_runtime_guard_positive.sio"
NEGATIVE="tests/frontend/knowledge_runtime_guard_reject.sio"
MULTI_POSITIVE="tests/frontend/knowledge_runtime_guard_multi_positive.sio"
MULTI_NEGATIVE="tests/frontend/knowledge_runtime_guard_multi_reject.sio"
UPPER_POSITIVE="tests/frontend/knowledge_runtime_guard_upper_positive.sio"
UPPER_NEGATIVE="tests/frontend/knowledge_runtime_guard_upper_reject.sio"
EQ_POSITIVE="tests/frontend/knowledge_runtime_guard_eq_positive.sio"
EQ_NEGATIVE="tests/frontend/knowledge_runtime_guard_eq_reject.sio"
UNIT_POSITIVE="tests/frontend/knowledge_runtime_guard_unit_positive.sio"
UNIT_NEGATIVE="tests/frontend/knowledge_runtime_guard_unit_reject.sio"
INTERNAL_LABEL_POSITIVE="tests/frontend/knowledge_runtime_guard_internal_label_positive.sio"
INTERNAL_LABEL_NEGATIVE="tests/frontend/knowledge_runtime_guard_internal_label_reject.sio"

CACHE_DIR="$TMP_DIR/knowledge-runtime-guard-cache"
POSITIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_positive.expanded.sio"
NEGATIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_reject.expanded.sio"
MULTI_POSITIVE_EXPANDED_FIRST="$TMP_DIR/knowledge_runtime_guard_multi_positive.first.expanded.sio"
MULTI_POSITIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_multi_positive.expanded.sio"
MULTI_NEGATIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_multi_reject.expanded.sio"
UPPER_POSITIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_upper_positive.expanded.sio"
UPPER_NEGATIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_upper_reject.expanded.sio"
EQ_POSITIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_eq_positive.expanded.sio"
EQ_NEGATIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_eq_reject.expanded.sio"
UNIT_POSITIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_unit_positive.expanded.sio"
UNIT_NEGATIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_unit_reject.expanded.sio"
INTERNAL_LABEL_POSITIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_internal_label_positive.expanded.sio"
INTERNAL_LABEL_NEGATIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_internal_label_reject.expanded.sio"
MULTI_POSITIVE_CACHE_MISS_LOG="$TMP_DIR/multi-positive-cache-miss.log"
MULTI_POSITIVE_CACHE_HIT_LOG="$TMP_DIR/multi-positive-cache-hit.log"

bash "$EXPANDER" "$POSITIVE" "$POSITIVE_EXPANDED"
bash "$EXPANDER" "$NEGATIVE" "$NEGATIVE_EXPANDED"
bash "$EXPANDER" --cache-dir "$CACHE_DIR" "$MULTI_POSITIVE" "$MULTI_POSITIVE_EXPANDED_FIRST" 2>"$MULTI_POSITIVE_CACHE_MISS_LOG"
bash "$EXPANDER" --cache-dir "$CACHE_DIR" "$MULTI_POSITIVE" "$MULTI_POSITIVE_EXPANDED" 2>"$MULTI_POSITIVE_CACHE_HIT_LOG"
bash "$EXPANDER" "$MULTI_NEGATIVE" "$MULTI_NEGATIVE_EXPANDED"
bash "$EXPANDER" "$UPPER_POSITIVE" "$UPPER_POSITIVE_EXPANDED"
bash "$EXPANDER" "$UPPER_NEGATIVE" "$UPPER_NEGATIVE_EXPANDED"
bash "$EXPANDER" "$EQ_POSITIVE" "$EQ_POSITIVE_EXPANDED"
bash "$EXPANDER" "$EQ_NEGATIVE" "$EQ_NEGATIVE_EXPANDED"
bash "$EXPANDER" "$UNIT_POSITIVE" "$UNIT_POSITIVE_EXPANDED"
bash "$EXPANDER" "$UNIT_NEGATIVE" "$UNIT_NEGATIVE_EXPANDED"
bash "$EXPANDER" "$INTERNAL_LABEL_POSITIVE" "$INTERNAL_LABEL_POSITIVE_EXPANDED"
bash "$EXPANDER" "$INTERNAL_LABEL_NEGATIVE" "$INTERNAL_LABEL_NEGATIVE_EXPANDED"

if grep -q 'knowledge-runtime-guard cache MISS' "$MULTI_POSITIVE_CACHE_MISS_LOG" &&
   grep -q 'knowledge-runtime-guard cache HIT' "$MULTI_POSITIVE_CACHE_HIT_LOG" &&
   cmp -s "$MULTI_POSITIVE_EXPANDED_FIRST" "$MULTI_POSITIVE_EXPANDED"; then
  printf 'PASS  Knowledge runtime guard expansion populated and reused deterministic .guardcache\n'
else
  printf 'FAIL  Knowledge runtime guard expansion did not prove deterministic .guardcache reuse\n' >&2
  printf '%s\n' '--- cache miss log ---' >&2
  cat "$MULTI_POSITIVE_CACHE_MISS_LOG" >&2
  printf '%s\n' '--- cache hit log ---' >&2
  cat "$MULTI_POSITIVE_CACHE_HIT_LOG" >&2
  exit 1
fi

if grep -q '__sounio_knowledge_guard_PatientRuntimeGuardPositive_age_ge_18' "$POSITIVE_EXPANDED" &&
   grep -q 'assert(value.age >= 18)' "$POSITIVE_EXPANDED"; then
  printf 'PASS  %s expanded with a generated Knowledge<T> runtime guard helper\n' "$POSITIVE"
else
  printf 'FAIL  %s did not expand to a generated Knowledge<T> runtime guard helper\n' "$POSITIVE" >&2
  cat "$POSITIVE_EXPANDED" >&2
  exit 1
fi

POSITIVE_LOG="$TMP_DIR/positive.log"
if bin/souc run "$POSITIVE_EXPANDED" >"$POSITIVE_LOG" 2>&1; then
  printf 'PASS  %s accepted dynamic value satisfying generated runtime guard\n' "$POSITIVE"
else
  printf 'FAIL  %s should pass the generated Knowledge<T> runtime guard\n' "$POSITIVE" >&2
  cat "$POSITIVE_LOG" >&2
  exit 1
fi

NEGATIVE_LOG="$TMP_DIR/negative.log"
if bin/souc run "$NEGATIVE_EXPANDED" >"$NEGATIVE_LOG" 2>&1; then
  printf 'FAIL  %s should fail the generated Knowledge<T> runtime guard\n' "$NEGATIVE" >&2
  cat "$NEGATIVE_LOG" >&2
  exit 1
else
  printf 'PASS  %s failed the generated Knowledge<T> runtime guard\n' "$NEGATIVE"
fi

if grep -q '__sounio_knowledge_guard_PatientRuntimeGuardMultiPositive_age_ge_18_glucose_ge_126' "$MULTI_POSITIVE_EXPANDED" &&
   grep -q 'assert(value.age >= 18)' "$MULTI_POSITIVE_EXPANDED" &&
   grep -q 'assert(value.glucose >= 126)' "$MULTI_POSITIVE_EXPANDED"; then
  printf 'PASS  %s expanded with conjunctive Knowledge<T> runtime guards\n' "$MULTI_POSITIVE"
else
  printf 'FAIL  %s did not expand to conjunctive Knowledge<T> runtime guards\n' "$MULTI_POSITIVE" >&2
  cat "$MULTI_POSITIVE_EXPANDED" >&2
  exit 1
fi

MULTI_POSITIVE_LOG="$TMP_DIR/multi-positive.log"
if bin/souc run "$MULTI_POSITIVE_EXPANDED" >"$MULTI_POSITIVE_LOG" 2>&1; then
  printf 'PASS  %s accepted dynamic value satisfying conjunctive generated runtime guards\n' "$MULTI_POSITIVE"
else
  printf 'FAIL  %s should pass conjunctive generated Knowledge<T> runtime guards\n' "$MULTI_POSITIVE" >&2
  cat "$MULTI_POSITIVE_LOG" >&2
  exit 1
fi

MULTI_NEGATIVE_LOG="$TMP_DIR/multi-negative.log"
if bin/souc run "$MULTI_NEGATIVE_EXPANDED" >"$MULTI_NEGATIVE_LOG" 2>&1; then
  printf 'FAIL  %s should fail one conjunctive generated Knowledge<T> runtime guard\n' "$MULTI_NEGATIVE" >&2
  cat "$MULTI_NEGATIVE_LOG" >&2
  exit 1
else
  printf 'PASS  %s failed one conjunctive generated Knowledge<T> runtime guard\n' "$MULTI_NEGATIVE"
fi

if grep -q '__sounio_knowledge_guard_ScoreRuntimeGuardUpperPositive_score_le_100' "$UPPER_POSITIVE_EXPANDED" &&
   grep -q 'assert(value.score <= 100)' "$UPPER_POSITIVE_EXPANDED"; then
  printf 'PASS  %s expanded with an upper-bound Knowledge<T> runtime guard\n' "$UPPER_POSITIVE"
else
  printf 'FAIL  %s did not expand to an upper-bound Knowledge<T> runtime guard\n' "$UPPER_POSITIVE" >&2
  cat "$UPPER_POSITIVE_EXPANDED" >&2
  exit 1
fi

UPPER_POSITIVE_LOG="$TMP_DIR/upper-positive.log"
if bin/souc run "$UPPER_POSITIVE_EXPANDED" >"$UPPER_POSITIVE_LOG" 2>&1; then
  printf 'PASS  %s accepted dynamic value satisfying generated upper-bound runtime guard\n' "$UPPER_POSITIVE"
else
  printf 'FAIL  %s should pass the generated upper-bound Knowledge<T> runtime guard\n' "$UPPER_POSITIVE" >&2
  cat "$UPPER_POSITIVE_LOG" >&2
  exit 1
fi

UPPER_NEGATIVE_LOG="$TMP_DIR/upper-negative.log"
if bin/souc run "$UPPER_NEGATIVE_EXPANDED" >"$UPPER_NEGATIVE_LOG" 2>&1; then
  printf 'FAIL  %s should fail the generated upper-bound Knowledge<T> runtime guard\n' "$UPPER_NEGATIVE" >&2
  cat "$UPPER_NEGATIVE_LOG" >&2
  exit 1
else
  printf 'PASS  %s failed the generated upper-bound Knowledge<T> runtime guard\n' "$UPPER_NEGATIVE"
fi

if grep -q '__sounio_knowledge_guard_AssayRuntimeGuardEqPositive_repeats_eq_3' "$EQ_POSITIVE_EXPANDED" &&
   grep -q 'assert(value.repeats == 3)' "$EQ_POSITIVE_EXPANDED"; then
  printf 'PASS  %s expanded with an equality Knowledge<T> runtime guard\n' "$EQ_POSITIVE"
else
  printf 'FAIL  %s did not expand to an equality Knowledge<T> runtime guard\n' "$EQ_POSITIVE" >&2
  cat "$EQ_POSITIVE_EXPANDED" >&2
  exit 1
fi

EQ_POSITIVE_LOG="$TMP_DIR/eq-positive.log"
if bin/souc run "$EQ_POSITIVE_EXPANDED" >"$EQ_POSITIVE_LOG" 2>&1; then
  printf 'PASS  %s accepted dynamic value satisfying generated equality runtime guard\n' "$EQ_POSITIVE"
else
  printf 'FAIL  %s should pass the generated equality Knowledge<T> runtime guard\n' "$EQ_POSITIVE" >&2
  cat "$EQ_POSITIVE_LOG" >&2
  exit 1
fi

EQ_NEGATIVE_LOG="$TMP_DIR/eq-negative.log"
if bin/souc run "$EQ_NEGATIVE_EXPANDED" >"$EQ_NEGATIVE_LOG" 2>&1; then
  printf 'FAIL  %s should fail the generated equality Knowledge<T> runtime guard\n' "$EQ_NEGATIVE" >&2
  cat "$EQ_NEGATIVE_LOG" >&2
  exit 1
else
  printf 'PASS  %s failed the generated equality Knowledge<T> runtime guard\n' "$EQ_NEGATIVE"
fi

if grep -q '__sounio_knowledge_guard_DoseRuntimeGuardUnitPositive_amount_ge_500_mg' "$UNIT_POSITIVE_EXPANDED" &&
   grep -q 'let __sounio_threshold_amount: mg = 500.0' "$UNIT_POSITIVE_EXPANDED" &&
   grep -q 'assert(__sounio_ratio_amount >= 1.0)' "$UNIT_POSITIVE_EXPANDED"; then
  printf 'PASS  %s expanded with a unit-suffixed Knowledge<T> runtime guard\n' "$UNIT_POSITIVE"
else
  printf 'FAIL  %s did not expand to a unit-suffixed Knowledge<T> runtime guard\n' "$UNIT_POSITIVE" >&2
  cat "$UNIT_POSITIVE_EXPANDED" >&2
  exit 1
fi

UNIT_POSITIVE_LOG="$TMP_DIR/unit-positive.log"
if bin/souc run "$UNIT_POSITIVE_EXPANDED" >"$UNIT_POSITIVE_LOG" 2>&1; then
  printf 'PASS  %s accepted dynamic unit value satisfying generated runtime guard\n' "$UNIT_POSITIVE"
else
  printf 'FAIL  %s should pass the generated unit-suffixed Knowledge<T> runtime guard\n' "$UNIT_POSITIVE" >&2
  cat "$UNIT_POSITIVE_LOG" >&2
  exit 1
fi

UNIT_NEGATIVE_LOG="$TMP_DIR/unit-negative.log"
if bin/souc run "$UNIT_NEGATIVE_EXPANDED" >"$UNIT_NEGATIVE_LOG" 2>&1; then
  printf 'FAIL  %s should fail the generated unit-suffixed Knowledge<T> runtime guard\n' "$UNIT_NEGATIVE" >&2
  cat "$UNIT_NEGATIVE_LOG" >&2
  exit 1
else
  printf 'PASS  %s failed the generated unit-suffixed Knowledge<T> runtime guard\n' "$UNIT_NEGATIVE"
fi

if grep -q '__sounio_knowledge_guard_PatientRuntimeGuardInternalLabelPositive_glucose_ge_126_0_mg_dL' "$INTERNAL_LABEL_POSITIVE_EXPANDED" &&
   grep -q 'let __sounio_threshold_glucose: mg_dL = 126.0' "$INTERNAL_LABEL_POSITIVE_EXPANDED" &&
   grep -q 'assert(__sounio_ratio_glucose >= 1.0)' "$INTERNAL_LABEL_POSITIVE_EXPANDED"; then
  printf 'PASS  %s expanded with an internal-label unit Knowledge<T> runtime guard\n' "$INTERNAL_LABEL_POSITIVE"
else
  printf 'FAIL  %s did not expand to an internal-label unit Knowledge<T> runtime guard\n' "$INTERNAL_LABEL_POSITIVE" >&2
  cat "$INTERNAL_LABEL_POSITIVE_EXPANDED" >&2
  exit 1
fi

if bin/souc-linux-x86_64 self-hosted/compiler/lean_single.sio "$CURRENT_SOUC" >"$BUILD_LOG" 2>&1; then
  chmod +x "$CURRENT_SOUC"
  printf 'PASS  built current-source lean_single compiler for internal-label runtime guard gate\n'
else
  printf 'FAIL  could not build current-source lean_single compiler for internal-label runtime guard gate\n' >&2
  cat "$BUILD_LOG" >&2
  exit 1
fi

INTERNAL_LABEL_POSITIVE_LOG="$TMP_DIR/internal-label-positive.log"
if SOUNIO_SOUC_BIN="$CURRENT_SOUC" bin/souc run "$INTERNAL_LABEL_POSITIVE_EXPANDED" >"$INTERNAL_LABEL_POSITIVE_LOG" 2>&1; then
  printf 'PASS  %s accepted dynamic internal-label unit value satisfying generated runtime guard\n' "$INTERNAL_LABEL_POSITIVE"
else
  printf 'FAIL  %s should pass the generated internal-label unit Knowledge<T> runtime guard\n' "$INTERNAL_LABEL_POSITIVE" >&2
  cat "$INTERNAL_LABEL_POSITIVE_LOG" >&2
  exit 1
fi

INTERNAL_LABEL_NEGATIVE_LOG="$TMP_DIR/internal-label-negative.log"
if SOUNIO_SOUC_BIN="$CURRENT_SOUC" bin/souc run "$INTERNAL_LABEL_NEGATIVE_EXPANDED" >"$INTERNAL_LABEL_NEGATIVE_LOG" 2>&1; then
  printf 'FAIL  %s should fail the generated internal-label unit Knowledge<T> runtime guard\n' "$INTERNAL_LABEL_NEGATIVE" >&2
  cat "$INTERNAL_LABEL_NEGATIVE_LOG" >&2
  exit 1
else
  printf 'PASS  %s failed the generated internal-label unit Knowledge<T> runtime guard\n' "$INTERNAL_LABEL_NEGATIVE"
fi

cat <<'MSG'
Knowledge runtime guard expansion gate passed.
This proves the first pre-native executable guard bridge for dynamic
Knowledge<T where {...}> numeric constraints: generated helpers use
Sounio assert(...) at runtime, satisfying dynamic evidence runs successfully,
violating dynamic evidence fails at execution time, and multiple numeric
lower-bound constraints are enforced conjunctively. It now also proves an
upper-bound slice with `<=`, so the bridge is no longer restricted to lower
bound checks only, and an equality slice with `==`. Unit-suffixed numeric thresholds are
also guarded at runtime after the unit-typed field has been materialized by the
existing compiler. A current-source slice also covers internal validation-data
unit labels such as mg_dL without making clinical, conversion, UCUM, LOINC,
ChEBI, dosing, or regulatory-exchange claims. Native backend guard/trap
lowering remains future work.
MSG
