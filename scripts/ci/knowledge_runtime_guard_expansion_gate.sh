#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

EXPANDER="scripts/ontology/expand_knowledge_runtime_guards.sh"
OBLIGATION_PROBE="self-hosted/compiler/k2_knowledge_runtime_obligation_probe.sio"
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
CALL_POSITIVE="tests/frontend/knowledge_runtime_guard_call_positive.sio"
CALL_NEGATIVE="tests/frontend/knowledge_runtime_guard_call_reject.sio"
ASSIGN_POSITIVE="tests/frontend/knowledge_runtime_guard_assign_positive.sio"
ASSIGN_NEGATIVE="tests/frontend/knowledge_runtime_guard_assign_reject.sio"
DIRECTIVE_POSITIVE="tests/frontend/knowledge_runtime_guard_directive_positive.sio"
DIRECTIVE_NEGATIVE="tests/frontend/knowledge_runtime_guard_directive_reject.sio"

CACHE_DIR="$TMP_DIR/knowledge-runtime-guard-cache"
POSITIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_positive.expanded.sio"
NEGATIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_reject.expanded.sio"
POSITIVE_NATIVE_BIN="$TMP_DIR/knowledge_runtime_guard_positive.native"
NEGATIVE_NATIVE_BIN="$TMP_DIR/knowledge_runtime_guard_reject.native"
MULTI_POSITIVE_EXPANDED_FIRST="$TMP_DIR/knowledge_runtime_guard_multi_positive.first.expanded.sio"
MULTI_POSITIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_multi_positive.expanded.sio"
MULTI_POSITIVE_MANIFEST_FIRST="$TMP_DIR/knowledge_runtime_guard_multi_positive.first.guardmanifest.tsv"
MULTI_POSITIVE_MANIFEST="$TMP_DIR/knowledge_runtime_guard_multi_positive.guardmanifest.tsv"
MULTI_NEGATIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_multi_reject.expanded.sio"
UPPER_POSITIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_upper_positive.expanded.sio"
UPPER_NEGATIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_upper_reject.expanded.sio"
EQ_POSITIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_eq_positive.expanded.sio"
EQ_NEGATIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_eq_reject.expanded.sio"
UNIT_POSITIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_unit_positive.expanded.sio"
UNIT_POSITIVE_MANIFEST="$TMP_DIR/knowledge_runtime_guard_unit_positive.guardmanifest.tsv"
UNIT_NEGATIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_unit_reject.expanded.sio"
INTERNAL_LABEL_POSITIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_internal_label_positive.expanded.sio"
INTERNAL_LABEL_POSITIVE_MANIFEST="$TMP_DIR/knowledge_runtime_guard_internal_label_positive.guardmanifest.tsv"
INTERNAL_LABEL_NEGATIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_internal_label_reject.expanded.sio"
CALL_POSITIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_call_positive.expanded.sio"
CALL_NEGATIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_call_reject.expanded.sio"
ASSIGN_POSITIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_assign_positive.expanded.sio"
ASSIGN_NEGATIVE_EXPANDED="$TMP_DIR/knowledge_runtime_guard_assign_reject.expanded.sio"
MULTI_POSITIVE_CACHE_MISS_LOG="$TMP_DIR/multi-positive-cache-miss.log"
MULTI_POSITIVE_CACHE_HIT_LOG="$TMP_DIR/multi-positive-cache-hit.log"

bash "$EXPANDER" "$POSITIVE" "$POSITIVE_EXPANDED"
bash "$EXPANDER" "$NEGATIVE" "$NEGATIVE_EXPANDED"
bash "$EXPANDER" --cache-dir "$CACHE_DIR" --manifest "$MULTI_POSITIVE_MANIFEST_FIRST" "$MULTI_POSITIVE" "$MULTI_POSITIVE_EXPANDED_FIRST" 2>"$MULTI_POSITIVE_CACHE_MISS_LOG"
bash "$EXPANDER" --cache-dir "$CACHE_DIR" --manifest "$MULTI_POSITIVE_MANIFEST" "$MULTI_POSITIVE" "$MULTI_POSITIVE_EXPANDED" 2>"$MULTI_POSITIVE_CACHE_HIT_LOG"
bash "$EXPANDER" "$MULTI_NEGATIVE" "$MULTI_NEGATIVE_EXPANDED"
bash "$EXPANDER" "$UPPER_POSITIVE" "$UPPER_POSITIVE_EXPANDED"
bash "$EXPANDER" "$UPPER_NEGATIVE" "$UPPER_NEGATIVE_EXPANDED"
bash "$EXPANDER" "$EQ_POSITIVE" "$EQ_POSITIVE_EXPANDED"
bash "$EXPANDER" "$EQ_NEGATIVE" "$EQ_NEGATIVE_EXPANDED"
bash "$EXPANDER" --manifest "$UNIT_POSITIVE_MANIFEST" "$UNIT_POSITIVE" "$UNIT_POSITIVE_EXPANDED"
bash "$EXPANDER" "$UNIT_NEGATIVE" "$UNIT_NEGATIVE_EXPANDED"
bash "$EXPANDER" --manifest "$INTERNAL_LABEL_POSITIVE_MANIFEST" "$INTERNAL_LABEL_POSITIVE" "$INTERNAL_LABEL_POSITIVE_EXPANDED"
bash "$EXPANDER" "$INTERNAL_LABEL_NEGATIVE" "$INTERNAL_LABEL_NEGATIVE_EXPANDED"
bash "$EXPANDER" "$CALL_POSITIVE" "$CALL_POSITIVE_EXPANDED"
bash "$EXPANDER" "$CALL_NEGATIVE" "$CALL_NEGATIVE_EXPANDED"
bash "$EXPANDER" "$ASSIGN_POSITIVE" "$ASSIGN_POSITIVE_EXPANDED"
bash "$EXPANDER" "$ASSIGN_NEGATIVE" "$ASSIGN_NEGATIVE_EXPANDED"

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

if grep -q '^# Sounio Knowledge runtime guard diagnostics manifest v1$' "$MULTI_POSITIVE_MANIFEST" &&
   grep -q '^type	field	op	threshold	unit	constraint$' "$MULTI_POSITIVE_MANIFEST" &&
   awk -F '\t' '$1=="PatientRuntimeGuardMultiPositive" && $2=="age" && $3==">=" && $4=="18" && $5=="" && $6=="PatientRuntimeGuardMultiPositive.age >= 18" { found=1 } END { exit(found ? 0 : 1) }' "$MULTI_POSITIVE_MANIFEST" &&
   awk -F '\t' '$1=="PatientRuntimeGuardMultiPositive" && $2=="glucose" && $3==">=" && $4=="126" && $5=="" && $6=="PatientRuntimeGuardMultiPositive.glucose >= 126" { found=1 } END { exit(found ? 0 : 1) }' "$MULTI_POSITIVE_MANIFEST" &&
   awk -F '\t' '$1=="DoseRuntimeGuardUnitPositive" && $2=="amount" && $3==">=" && $4=="500" && $5=="mg" && $6=="DoseRuntimeGuardUnitPositive.amount >= 500 <mg>" { found=1 } END { exit(found ? 0 : 1) }' "$UNIT_POSITIVE_MANIFEST" &&
   awk -F '\t' '$1=="PatientRuntimeGuardInternalLabelPositive" && $2=="glucose" && $3==">=" && $4=="126.0" && $5=="mg_dL" && $6=="PatientRuntimeGuardInternalLabelPositive.glucose >= 126.0 <mg_dL>" { found=1 } END { exit(found ? 0 : 1) }' "$INTERNAL_LABEL_POSITIVE_MANIFEST"; then
  printf 'PASS  Knowledge runtime guard expansion wrote auditable guard diagnostic manifests\n'
else
  printf 'FAIL  Knowledge runtime guard expansion did not write the expected guard diagnostic manifests\n' >&2
  printf '%s\n' '--- multi manifest ---' >&2
  cat "$MULTI_POSITIVE_MANIFEST" >&2
  printf '%s\n' '--- unit manifest ---' >&2
  cat "$UNIT_POSITIVE_MANIFEST" >&2
  printf '%s\n' '--- internal-label manifest ---' >&2
  cat "$INTERNAL_LABEL_POSITIVE_MANIFEST" >&2
  exit 1
fi

assert_obligation_drained() {
  local label="$1"
  local source_file="$2"
  local expanded_file="$3"
  local source_log="$TMP_DIR/${label}-source-obligations.log"
  local expanded_log="$TMP_DIR/${label}-expanded-obligations.log"
  if bin/souc run "$OBLIGATION_PROBE" -- "$source_file" >"$source_log" 2>&1 &&
     bin/souc run "$OBLIGATION_PROBE" -- "$expanded_file" >"$expanded_log" 2>&1 &&
     grep -q 'knowledge_runtime_obligation_verdict=0' "$source_log" &&
     grep -Eq 'knowledge_runtime_obligation_count=[1-9][0-9]*' "$source_log" &&
     grep -q 'knowledge_runtime_obligation_verdict=0' "$expanded_log" &&
     grep -q 'knowledge_runtime_obligation_count=0' "$expanded_log"; then
    return 0
  fi
  printf 'FAIL  Knowledge runtime guard expansion did not drain checker-visible obligations for %s\n' "$label" >&2
  printf '%s\n' "--- $label source obligations ---" >&2
  cat "$source_log" >&2
  printf '%s\n' "--- $label expanded obligations ---" >&2
  cat "$expanded_log" >&2
  exit 1
}

assert_obligation_drained "return-positive" "$POSITIVE" "$POSITIVE_EXPANDED"
assert_obligation_drained "return-reject" "$NEGATIVE" "$NEGATIVE_EXPANDED"
assert_obligation_drained "call-positive" "$CALL_POSITIVE" "$CALL_POSITIVE_EXPANDED"
assert_obligation_drained "call-reject" "$CALL_NEGATIVE" "$CALL_NEGATIVE_EXPANDED"
assert_obligation_drained "assign-positive" "$ASSIGN_POSITIVE" "$ASSIGN_POSITIVE_EXPANDED"
assert_obligation_drained "assign-reject" "$ASSIGN_NEGATIVE" "$ASSIGN_NEGATIVE_EXPANDED"
assert_obligation_drained "multi-positive" "$MULTI_POSITIVE" "$MULTI_POSITIVE_EXPANDED"
assert_obligation_drained "multi-reject" "$MULTI_NEGATIVE" "$MULTI_NEGATIVE_EXPANDED"
assert_obligation_drained "upper-positive" "$UPPER_POSITIVE" "$UPPER_POSITIVE_EXPANDED"
assert_obligation_drained "upper-reject" "$UPPER_NEGATIVE" "$UPPER_NEGATIVE_EXPANDED"
assert_obligation_drained "equality-positive" "$EQ_POSITIVE" "$EQ_POSITIVE_EXPANDED"
assert_obligation_drained "equality-reject" "$EQ_NEGATIVE" "$EQ_NEGATIVE_EXPANDED"
assert_obligation_drained "unit-positive" "$UNIT_POSITIVE" "$UNIT_POSITIVE_EXPANDED"
assert_obligation_drained "unit-reject" "$UNIT_NEGATIVE" "$UNIT_NEGATIVE_EXPANDED"
assert_obligation_drained "internal-label-positive" "$INTERNAL_LABEL_POSITIVE" "$INTERNAL_LABEL_POSITIVE_EXPANDED"
assert_obligation_drained "internal-label-reject" "$INTERNAL_LABEL_NEGATIVE" "$INTERNAL_LABEL_NEGATIVE_EXPANDED"
printf 'PASS  Knowledge runtime guard expansion consumes checker-visible runtime obligations across dynamic sites and supported guard families\n'

if grep -q '__sounio_knowledge_guard_PatientRuntimeGuardPositive_age_ge_18' "$POSITIVE_EXPANDED" &&
   grep -q 'Knowledge runtime guard diagnostic: PatientRuntimeGuardPositive.age >= 18' "$POSITIVE_EXPANDED" &&
   grep -q 'assert(value.age >= 18)' "$POSITIVE_EXPANDED"; then
  printf 'PASS  %s expanded with a generated Knowledge<T> runtime guard helper and diagnostic\n' "$POSITIVE"
else
  printf 'FAIL  %s did not expand to a generated Knowledge<T> runtime guard helper with diagnostic\n' "$POSITIVE" >&2
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

POSITIVE_NATIVE_BUILD_LOG="$TMP_DIR/positive-native-build.log"
POSITIVE_NATIVE_LOG="$TMP_DIR/positive-native.log"
if bin/souc compile "$POSITIVE_EXPANDED" -o "$POSITIVE_NATIVE_BIN" >"$POSITIVE_NATIVE_BUILD_LOG" 2>&1 &&
   "$POSITIVE_NATIVE_BIN" >"$POSITIVE_NATIVE_LOG" 2>&1; then
  printf 'PASS  %s lowered the generated Knowledge<T> runtime guard through native assert/trap path\n' "$POSITIVE"
else
  printf 'FAIL  %s did not compile/run through the native generated guard path\n' "$POSITIVE" >&2
  printf '%s\n' '--- native build log ---' >&2
  cat "$POSITIVE_NATIVE_BUILD_LOG" >&2
  printf '%s\n' '--- native run log ---' >&2
  cat "$POSITIVE_NATIVE_LOG" >&2
  exit 1
fi

NEGATIVE_NATIVE_BUILD_LOG="$TMP_DIR/negative-native-build.log"
NEGATIVE_NATIVE_LOG="$TMP_DIR/negative-native.log"
if ! bin/souc compile "$NEGATIVE_EXPANDED" -o "$NEGATIVE_NATIVE_BIN" >"$NEGATIVE_NATIVE_BUILD_LOG" 2>&1; then
  printf 'FAIL  %s did not compile the native generated guard reject witness\n' "$NEGATIVE" >&2
  cat "$NEGATIVE_NATIVE_BUILD_LOG" >&2
  exit 1
fi
set +e
"$NEGATIVE_NATIVE_BIN" >"$NEGATIVE_NATIVE_LOG" 2>&1
NEGATIVE_NATIVE_RC=$?
set -e
if [[ "$NEGATIVE_NATIVE_RC" -eq 1 ]]; then
  printf 'PASS  %s trapped through native assert path with exit code 1\n' "$NEGATIVE"
else
  printf 'FAIL  %s should trap through native assert path with exit code 1, got %s\n' "$NEGATIVE" "$NEGATIVE_NATIVE_RC" >&2
  cat "$NEGATIVE_NATIVE_LOG" >&2
  exit 1
fi

if grep -q '__sounio_knowledge_guard_PatientRuntimeGuardCallPositive_age_ge_18' "$CALL_POSITIVE_EXPANDED" &&
   grep -q 'assert(value.age >= 18)' "$CALL_POSITIVE_EXPANDED" &&
   grep -q '__sounio_knowledge_guard_PatientRuntimeGuardAssignPositive_age_ge_18' "$ASSIGN_POSITIVE_EXPANDED" &&
   grep -q 'assert(value.age >= 18)' "$ASSIGN_POSITIVE_EXPANDED"; then
  printf 'PASS  call-boundary and assignment witnesses expanded with generated Knowledge<T> runtime guards\n'
else
  printf 'FAIL  call-boundary or assignment witnesses did not expand to generated Knowledge<T> runtime guards\n' >&2
  printf '%s\n' '--- call positive expanded ---' >&2
  cat "$CALL_POSITIVE_EXPANDED" >&2
  printf '%s\n' '--- assignment positive expanded ---' >&2
  cat "$ASSIGN_POSITIVE_EXPANDED" >&2
  exit 1
fi

CALL_POSITIVE_LOG="$TMP_DIR/call-positive.log"
if bin/souc run "$CALL_POSITIVE_EXPANDED" >"$CALL_POSITIVE_LOG" 2>&1; then
  printf 'PASS  %s accepted call-boundary dynamic value satisfying generated runtime guard\n' "$CALL_POSITIVE"
else
  printf 'FAIL  %s should pass the generated call-boundary Knowledge<T> runtime guard\n' "$CALL_POSITIVE" >&2
  cat "$CALL_POSITIVE_LOG" >&2
  exit 1
fi

CALL_NEGATIVE_LOG="$TMP_DIR/call-negative.log"
if bin/souc run "$CALL_NEGATIVE_EXPANDED" >"$CALL_NEGATIVE_LOG" 2>&1; then
  printf 'FAIL  %s should fail the generated call-boundary Knowledge<T> runtime guard\n' "$CALL_NEGATIVE" >&2
  cat "$CALL_NEGATIVE_LOG" >&2
  exit 1
else
  printf 'PASS  %s failed the generated call-boundary Knowledge<T> runtime guard\n' "$CALL_NEGATIVE"
fi

ASSIGN_POSITIVE_LOG="$TMP_DIR/assign-positive.log"
if bin/souc run "$ASSIGN_POSITIVE_EXPANDED" >"$ASSIGN_POSITIVE_LOG" 2>&1; then
  printf 'PASS  %s accepted assignment dynamic value satisfying generated runtime guard\n' "$ASSIGN_POSITIVE"
else
  printf 'FAIL  %s should pass the generated assignment Knowledge<T> runtime guard\n' "$ASSIGN_POSITIVE" >&2
  cat "$ASSIGN_POSITIVE_LOG" >&2
  exit 1
fi

ASSIGN_NEGATIVE_LOG="$TMP_DIR/assign-negative.log"
if bin/souc run "$ASSIGN_NEGATIVE_EXPANDED" >"$ASSIGN_NEGATIVE_LOG" 2>&1; then
  printf 'FAIL  %s should fail the generated assignment Knowledge<T> runtime guard\n' "$ASSIGN_NEGATIVE" >&2
  cat "$ASSIGN_NEGATIVE_LOG" >&2
  exit 1
else
  printf 'PASS  %s failed the generated assignment Knowledge<T> runtime guard\n' "$ASSIGN_NEGATIVE"
fi

assert_native_guard_pair() {
  local label="$1"
  local positive_source="$2"
  local positive_expanded="$3"
  local negative_source="$4"
  local negative_expanded="$5"
  local positive_bin="$TMP_DIR/${label}-positive.native"
  local negative_bin="$TMP_DIR/${label}-negative.native"
  local positive_build_log="$TMP_DIR/${label}-positive-native-build.log"
  local negative_build_log="$TMP_DIR/${label}-negative-native-build.log"
  local positive_run_log="$TMP_DIR/${label}-positive-native.log"
  local negative_run_log="$TMP_DIR/${label}-negative-native.log"

  if ! bin/souc compile "$positive_expanded" -o "$positive_bin" >"$positive_build_log" 2>&1; then
    printf 'FAIL  %s did not compile native positive generated guard witness %s\n' "$label" "$positive_source" >&2
    cat "$positive_build_log" >&2
    exit 1
  fi
  if "$positive_bin" >"$positive_run_log" 2>&1; then
    printf 'PASS  %s positive generated guard witness lowered through native assert/trap path\n' "$label"
  else
    printf 'FAIL  %s positive native generated guard witness should exit 0\n' "$label" >&2
    cat "$positive_run_log" >&2
    exit 1
  fi

  if ! bin/souc compile "$negative_expanded" -o "$negative_bin" >"$negative_build_log" 2>&1; then
    printf 'FAIL  %s did not compile native reject generated guard witness %s\n' "$label" "$negative_source" >&2
    cat "$negative_build_log" >&2
    exit 1
  fi
  set +e
  "$negative_bin" >"$negative_run_log" 2>&1
  local negative_rc=$?
  set -e
  if [[ "$negative_rc" -eq 1 ]]; then
    printf 'PASS  %s reject generated guard witness trapped through native assert path with exit code 1\n' "$label"
  else
    printf 'FAIL  %s reject native generated guard witness should exit 1, got %s\n' "$label" "$negative_rc" >&2
    cat "$negative_run_log" >&2
    exit 1
  fi
}

assert_native_guard_pair "call-boundary" "$CALL_POSITIVE" "$CALL_POSITIVE_EXPANDED" "$CALL_NEGATIVE" "$CALL_NEGATIVE_EXPANDED"
assert_native_guard_pair "assignment" "$ASSIGN_POSITIVE" "$ASSIGN_POSITIVE_EXPANDED" "$ASSIGN_NEGATIVE" "$ASSIGN_NEGATIVE_EXPANDED"

assert_launcher_guard_pair() {
  local label="$1"
  local positive_source="$2"
  local negative_source="$3"
  local positive_run_log="$TMP_DIR/${label}-launcher-positive-run.log"
  local negative_run_log="$TMP_DIR/${label}-launcher-negative-run.log"
  local positive_bin="$TMP_DIR/${label}-launcher-positive.native"
  local negative_bin="$TMP_DIR/${label}-launcher-negative.native"
  local positive_compile_log="$TMP_DIR/${label}-launcher-positive-compile.log"
  local negative_compile_log="$TMP_DIR/${label}-launcher-negative-compile.log"
  local positive_compile_run_log="$TMP_DIR/${label}-launcher-positive-compile-run.log"
  local negative_compile_run_log="$TMP_DIR/${label}-launcher-negative-compile-run.log"

  if SOUNIO_KNOWLEDGE_RUNTIME_GUARDS=1 bin/souc run "$positive_source" >"$positive_run_log" 2>&1; then
    printf 'PASS  %s launcher opt-in run accepted positive generated guard witness\n' "$label"
  else
    printf 'FAIL  %s launcher opt-in run should accept positive generated guard witness\n' "$label" >&2
    cat "$positive_run_log" >&2
    exit 1
  fi

  set +e
  SOUNIO_KNOWLEDGE_RUNTIME_GUARDS=1 bin/souc run "$negative_source" >"$negative_run_log" 2>&1
  local negative_run_rc=$?
  set -e
  if [[ "$negative_run_rc" -eq 1 ]]; then
    printf 'PASS  %s launcher opt-in run trapped reject generated guard witness with exit code 1\n' "$label"
  else
    printf 'FAIL  %s launcher opt-in run should trap reject generated guard witness with exit code 1, got %s\n' "$label" "$negative_run_rc" >&2
    cat "$negative_run_log" >&2
    exit 1
  fi

  if ! SOUNIO_KNOWLEDGE_RUNTIME_GUARDS=1 bin/souc compile "$positive_source" -o "$positive_bin" >"$positive_compile_log" 2>&1; then
    printf 'FAIL  %s launcher opt-in compile should build positive generated guard witness\n' "$label" >&2
    cat "$positive_compile_log" >&2
    exit 1
  fi
  if "$positive_bin" >"$positive_compile_run_log" 2>&1; then
    printf 'PASS  %s launcher opt-in compile produced positive native guard witness\n' "$label"
  else
    printf 'FAIL  %s launcher opt-in compiled positive guard witness should exit 0\n' "$label" >&2
    cat "$positive_compile_run_log" >&2
    exit 1
  fi

  if ! SOUNIO_KNOWLEDGE_RUNTIME_GUARDS=1 bin/souc compile "$negative_source" -o "$negative_bin" >"$negative_compile_log" 2>&1; then
    printf 'FAIL  %s launcher opt-in compile should build reject generated guard witness\n' "$label" >&2
    cat "$negative_compile_log" >&2
    exit 1
  fi
  set +e
  "$negative_bin" >"$negative_compile_run_log" 2>&1
  local negative_compile_run_rc=$?
  set -e
  if [[ "$negative_compile_run_rc" -eq 1 ]]; then
    printf 'PASS  %s launcher opt-in compile produced reject native guard witness trapping with exit code 1\n' "$label"
  else
    printf 'FAIL  %s launcher opt-in compiled reject guard witness should exit 1, got %s\n' "$label" "$negative_compile_run_rc" >&2
    cat "$negative_compile_run_log" >&2
    exit 1
  fi
}

assert_launcher_guard_pair "call-boundary" "$CALL_POSITIVE" "$CALL_NEGATIVE"

assert_directive_launcher_guard_pair() {
  local label="$1"
  local positive_source="$2"
  local negative_source="$3"
  local positive_run_log="$TMP_DIR/${label}-directive-positive-run.log"
  local negative_run_log="$TMP_DIR/${label}-directive-negative-run.log"
  local positive_bin="$TMP_DIR/${label}-directive-positive.native"
  local negative_bin="$TMP_DIR/${label}-directive-negative.native"
  local positive_compile_log="$TMP_DIR/${label}-directive-positive-compile.log"
  local negative_compile_log="$TMP_DIR/${label}-directive-negative-compile.log"
  local positive_compile_run_log="$TMP_DIR/${label}-directive-positive-compile-run.log"
  local negative_compile_run_log="$TMP_DIR/${label}-directive-negative-compile-run.log"

  if ! grep -Eq '^[[:space:]]*//@[[:space:]]*knowledge-runtime-guards([[:space:]]|$)' "$positive_source" ||
     ! grep -Eq '^[[:space:]]*//@[[:space:]]*knowledge-runtime-guards([[:space:]]|$)' "$negative_source"; then
    printf 'FAIL  %s directive launcher witnesses must declare //@ knowledge-runtime-guards\n' "$label" >&2
    exit 1
  fi

  if bin/souc run "$positive_source" >"$positive_run_log" 2>&1; then
    printf 'PASS  %s directive run accepted positive generated guard witness\n' "$label"
  else
    printf 'FAIL  %s directive run should accept positive generated guard witness\n' "$label" >&2
    cat "$positive_run_log" >&2
    exit 1
  fi

  set +e
  bin/souc run "$negative_source" >"$negative_run_log" 2>&1
  local negative_run_rc=$?
  set -e
  if [[ "$negative_run_rc" -eq 1 ]]; then
    printf 'PASS  %s directive run trapped reject generated guard witness with exit code 1\n' "$label"
  else
    printf 'FAIL  %s directive run should trap reject generated guard witness with exit code 1, got %s\n' "$label" "$negative_run_rc" >&2
    cat "$negative_run_log" >&2
    exit 1
  fi

  if ! bin/souc compile "$positive_source" -o "$positive_bin" >"$positive_compile_log" 2>&1; then
    printf 'FAIL  %s directive compile should build positive generated guard witness\n' "$label" >&2
    cat "$positive_compile_log" >&2
    exit 1
  fi
  if "$positive_bin" >"$positive_compile_run_log" 2>&1; then
    printf 'PASS  %s directive compile produced positive native guard witness\n' "$label"
  else
    printf 'FAIL  %s directive compiled positive guard witness should exit 0\n' "$label" >&2
    cat "$positive_compile_run_log" >&2
    exit 1
  fi

  if ! bin/souc compile "$negative_source" -o "$negative_bin" >"$negative_compile_log" 2>&1; then
    printf 'FAIL  %s directive compile should build reject generated guard witness\n' "$label" >&2
    cat "$negative_compile_log" >&2
    exit 1
  fi
  set +e
  "$negative_bin" >"$negative_compile_run_log" 2>&1
  local negative_compile_run_rc=$?
  set -e
  if [[ "$negative_compile_run_rc" -eq 1 ]]; then
    printf 'PASS  %s directive compile produced reject native guard witness trapping with exit code 1\n' "$label"
  else
    printf 'FAIL  %s directive compiled reject guard witness should exit 1, got %s\n' "$label" "$negative_compile_run_rc" >&2
    cat "$negative_compile_run_log" >&2
    exit 1
  fi
}

assert_directive_launcher_guard_pair "source-directive" "$DIRECTIVE_POSITIVE" "$DIRECTIVE_NEGATIVE"

if grep -q '__sounio_knowledge_guard_PatientRuntimeGuardMultiPositive_age_ge_18_glucose_ge_126' "$MULTI_POSITIVE_EXPANDED" &&
   grep -q 'Knowledge runtime guard diagnostic: PatientRuntimeGuardMultiPositive.age >= 18' "$MULTI_POSITIVE_EXPANDED" &&
   grep -q 'Knowledge runtime guard diagnostic: PatientRuntimeGuardMultiPositive.glucose >= 126' "$MULTI_POSITIVE_EXPANDED" &&
   grep -q 'assert(value.age >= 18)' "$MULTI_POSITIVE_EXPANDED" &&
   grep -q 'assert(value.glucose >= 126)' "$MULTI_POSITIVE_EXPANDED"; then
  printf 'PASS  %s expanded with conjunctive Knowledge<T> runtime guards and diagnostics\n' "$MULTI_POSITIVE"
else
  printf 'FAIL  %s did not expand to conjunctive Knowledge<T> runtime guards with diagnostics\n' "$MULTI_POSITIVE" >&2
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
assert_native_guard_pair "conjunctive-lower-bound" "$MULTI_POSITIVE" "$MULTI_POSITIVE_EXPANDED" "$MULTI_NEGATIVE" "$MULTI_NEGATIVE_EXPANDED"

if grep -q '__sounio_knowledge_guard_ScoreRuntimeGuardUpperPositive_score_le_100' "$UPPER_POSITIVE_EXPANDED" &&
   grep -q 'Knowledge runtime guard diagnostic: ScoreRuntimeGuardUpperPositive.score <= 100' "$UPPER_POSITIVE_EXPANDED" &&
   grep -q 'assert(value.score <= 100)' "$UPPER_POSITIVE_EXPANDED"; then
  printf 'PASS  %s expanded with an upper-bound Knowledge<T> runtime guard and diagnostic\n' "$UPPER_POSITIVE"
else
  printf 'FAIL  %s did not expand to an upper-bound Knowledge<T> runtime guard with diagnostic\n' "$UPPER_POSITIVE" >&2
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
assert_native_guard_pair "upper-bound" "$UPPER_POSITIVE" "$UPPER_POSITIVE_EXPANDED" "$UPPER_NEGATIVE" "$UPPER_NEGATIVE_EXPANDED"

if grep -q '__sounio_knowledge_guard_AssayRuntimeGuardEqPositive_repeats_eq_3' "$EQ_POSITIVE_EXPANDED" &&
   grep -q 'Knowledge runtime guard diagnostic: AssayRuntimeGuardEqPositive.repeats == 3' "$EQ_POSITIVE_EXPANDED" &&
   grep -q 'assert(value.repeats == 3)' "$EQ_POSITIVE_EXPANDED"; then
  printf 'PASS  %s expanded with an equality Knowledge<T> runtime guard and diagnostic\n' "$EQ_POSITIVE"
else
  printf 'FAIL  %s did not expand to an equality Knowledge<T> runtime guard with diagnostic\n' "$EQ_POSITIVE" >&2
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
assert_native_guard_pair "equality" "$EQ_POSITIVE" "$EQ_POSITIVE_EXPANDED" "$EQ_NEGATIVE" "$EQ_NEGATIVE_EXPANDED"

if grep -q '__sounio_knowledge_guard_DoseRuntimeGuardUnitPositive_amount_ge_500_mg' "$UNIT_POSITIVE_EXPANDED" &&
   grep -q 'Knowledge runtime guard diagnostic: DoseRuntimeGuardUnitPositive.amount >= 500 <mg>' "$UNIT_POSITIVE_EXPANDED" &&
   grep -q 'let __sounio_threshold_amount: mg = 500.0' "$UNIT_POSITIVE_EXPANDED" &&
   grep -q 'assert(__sounio_ratio_amount >= 1.0)' "$UNIT_POSITIVE_EXPANDED"; then
  printf 'PASS  %s expanded with a unit-suffixed Knowledge<T> runtime guard and diagnostic\n' "$UNIT_POSITIVE"
else
  printf 'FAIL  %s did not expand to a unit-suffixed Knowledge<T> runtime guard with diagnostic\n' "$UNIT_POSITIVE" >&2
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
assert_native_guard_pair "unit-suffixed" "$UNIT_POSITIVE" "$UNIT_POSITIVE_EXPANDED" "$UNIT_NEGATIVE" "$UNIT_NEGATIVE_EXPANDED"

if grep -q '__sounio_knowledge_guard_PatientRuntimeGuardInternalLabelPositive_glucose_ge_126_0_mg_dL' "$INTERNAL_LABEL_POSITIVE_EXPANDED" &&
   grep -q 'Knowledge runtime guard diagnostic: PatientRuntimeGuardInternalLabelPositive.glucose >= 126.0 <mg_dL>' "$INTERNAL_LABEL_POSITIVE_EXPANDED" &&
   grep -q 'let __sounio_threshold_glucose: mg_dL = 126.0' "$INTERNAL_LABEL_POSITIVE_EXPANDED" &&
   grep -q 'assert(__sounio_ratio_glucose >= 1.0)' "$INTERNAL_LABEL_POSITIVE_EXPANDED"; then
  printf 'PASS  %s expanded with an internal-label unit Knowledge<T> runtime guard and diagnostic\n' "$INTERNAL_LABEL_POSITIVE"
else
  printf 'FAIL  %s did not expand to an internal-label unit Knowledge<T> runtime guard with diagnostic\n' "$INTERNAL_LABEL_POSITIVE" >&2
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
assert_native_guard_pair "internal-label-unit" "$INTERNAL_LABEL_POSITIVE" "$INTERNAL_LABEL_POSITIVE_EXPANDED" "$INTERNAL_LABEL_NEGATIVE" "$INTERNAL_LABEL_NEGATIVE_EXPANDED"

cat <<'MSG'
Knowledge runtime guard expansion gate passed.
This proves the first pre-native executable guard bridge for dynamic
Knowledge<T where {...}> numeric constraints: generated helpers use
Sounio assert(...) at runtime, satisfying dynamic evidence runs successfully,
violating dynamic evidence fails at execution time, and multiple numeric
lower-bound constraints are enforced conjunctively. The gate now also proves the
bridge from checker-visible runtime obligations to generated guards across the
supported dynamic guard surface: return, call-boundary, and assignment
lower-bound sites, conjunctive lower bounds, upper bounds, equality,
unit-suffixed thresholds, and internal validation-data unit labels. The
original sources report nonzero runtime obligations, while their expanded
sources report zero obligations after guard insertion and proof-context
lowering. Those pairs compile through the default native backend after
pre-native expansion, where reject witnesses exit through the existing native
assert/trap path with code 1. This is native execution coverage for generated
guard assertions, not direct native lowering of checker-visible proof
obligations. The opt-in launcher bridge also proves
SOUNIO_KNOWLEDGE_RUNTIME_GUARDS=1 bin/souc run/compile applies this pre-native
expansion before invoking the usual selected compiler path; source files can
also request the same bridge with //@ knowledge-runtime-guards. It now also proves an
upper-bound slice with `<=`, so the bridge is no longer restricted to lower
bound checks only, and an equality slice with `==`. Unit-suffixed numeric thresholds are
also guarded at runtime after the unit-typed field has been materialized by the
existing compiler. A current-source slice also covers internal validation-data
unit labels such as mg_dL without making clinical, conversion, UCUM, LOINC,
ChEBI, dosing, or regulatory-exchange claims. The expanded source now preserves
stable guard diagnostics for representative lower-bound, conjunctive,
upper-bound, equality, named-unit, and internal-label constraints, and can emit
small deterministic structured guard diagnostic TSV manifests for audit. Direct
native backend proof-obligation guard/trap lowering remains future work.
MSG
