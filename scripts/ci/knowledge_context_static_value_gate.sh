#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

PROBE="self-hosted/compiler/k2_knowledge_static_value_probe.sio"
POSITIVE="tests/frontend/knowledge_static_value_positive.sio"
NEGATIVE_LITERAL="tests/frontend/knowledge_static_value_reject_literal.sio"
DYNAMIC_GAP="tests/frontend/knowledge_static_value_dynamic_gap.sio"
LOCAL_POSITIVE="tests/frontend/knowledge_static_value_local_positive.sio"
LOCAL_NEGATIVE_LITERAL="tests/frontend/knowledge_static_value_local_reject_literal.sio"
VAR_NEGATIVE_LITERAL="tests/frontend/knowledge_static_value_var_reject_literal.sio"
CALL_POSITIVE="tests/frontend/knowledge_static_value_call_positive.sio"
CALL_NEGATIVE_LITERAL="tests/frontend/knowledge_static_value_call_reject_literal.sio"
CALL_DYNAMIC_GAP="tests/frontend/knowledge_static_value_call_dynamic_gap.sio"
ASSIGN_POSITIVE="tests/frontend/knowledge_static_value_assign_positive.sio"
ASSIGN_NEGATIVE_LITERAL="tests/frontend/knowledge_static_value_assign_reject_literal.sio"
ASSIGN_DYNAMIC_GAP="tests/frontend/knowledge_static_value_assign_dynamic_gap.sio"
UNIT_POSITIVE="tests/frontend/knowledge_static_value_unit_positive.sio"
UNIT_NEGATIVE_LITERAL="tests/frontend/knowledge_static_value_unit_reject_literal.sio"
UNIT_DYNAMIC_GAP="tests/frontend/knowledge_static_value_unit_dynamic_gap.sio"
GENERATOR_POSITIVE="tests/frontend/knowledge_static_value_generator_positive.sio"
GENERATOR_NEGATIVE_LITERAL="tests/frontend/knowledge_static_value_generator_reject_literal.sio"
GENERATOR_DYNAMIC_GAP="tests/frontend/knowledge_static_value_generator_dynamic_gap.sio"
GENERATOR_ARG_POSITIVE="tests/frontend/knowledge_static_value_generator_arg_positive.sio"
GENERATOR_ARG_NEGATIVE_LITERAL="tests/frontend/knowledge_static_value_generator_arg_reject_literal.sio"
GENERATOR_ARG_DYNAMIC_GAP="tests/frontend/knowledge_static_value_generator_arg_dynamic_gap.sio"
GENERATOR_EXPR_POSITIVE="tests/frontend/knowledge_static_value_generator_expr_positive.sio"
GENERATOR_EXPR_NEGATIVE_LITERAL="tests/frontend/knowledge_static_value_generator_expr_reject_literal.sio"
GENERATOR_EXPR_DYNAMIC_GAP="tests/frontend/knowledge_static_value_generator_expr_dynamic_gap.sio"
GENERATOR_LOCAL_POSITIVE="tests/frontend/knowledge_static_value_generator_local_positive.sio"
GENERATOR_LOCAL_NEGATIVE_LITERAL="tests/frontend/knowledge_static_value_generator_local_reject_literal.sio"
GENERATOR_LOCAL_DYNAMIC_GAP="tests/frontend/knowledge_static_value_generator_local_dynamic_gap.sio"
GENERATOR_ASSIGN_POSITIVE="tests/frontend/knowledge_static_value_generator_assign_positive.sio"
GENERATOR_ASSIGN_NEGATIVE_LITERAL="tests/frontend/knowledge_static_value_generator_assign_reject_literal.sio"
GENERATOR_ASSIGN_DYNAMIC_GAP="tests/frontend/knowledge_static_value_generator_assign_dynamic_gap.sio"
GENERATOR_COMPOUND_POSITIVE="tests/frontend/knowledge_static_value_generator_compound_positive.sio"
GENERATOR_COMPOUND_NEGATIVE_LITERAL="tests/frontend/knowledge_static_value_generator_compound_reject_literal.sio"
GENERATOR_COMPOUND_DYNAMIC_GAP="tests/frontend/knowledge_static_value_generator_compound_dynamic_gap.sio"
GENERATOR_HELPER_POSITIVE="tests/frontend/knowledge_static_value_generator_helper_positive.sio"
GENERATOR_HELPER_NEGATIVE_LITERAL="tests/frontend/knowledge_static_value_generator_helper_reject_literal.sio"
GENERATOR_HELPER_DYNAMIC_GAP="tests/frontend/knowledge_static_value_generator_helper_dynamic_gap.sio"
GENERATOR_HELPER_LOCAL_POSITIVE="tests/frontend/knowledge_static_value_generator_helper_local_positive.sio"
GENERATOR_HELPER_LOCAL_NEGATIVE_LITERAL="tests/frontend/knowledge_static_value_generator_helper_local_reject_literal.sio"
GENERATOR_HELPER_LOCAL_DYNAMIC_GAP="tests/frontend/knowledge_static_value_generator_helper_local_dynamic_gap.sio"
GENERATOR_HELPER_ASSIGN_POSITIVE="tests/frontend/knowledge_static_value_generator_helper_assign_positive.sio"
GENERATOR_HELPER_ASSIGN_NEGATIVE_LITERAL="tests/frontend/knowledge_static_value_generator_helper_assign_reject_literal.sio"
GENERATOR_HELPER_ASSIGN_DYNAMIC_GAP="tests/frontend/knowledge_static_value_generator_helper_assign_dynamic_gap.sio"
GENERATOR_BRANCH_POSITIVE="tests/frontend/knowledge_static_value_generator_branch_positive.sio"
GENERATOR_BRANCH_NEGATIVE_LITERAL="tests/frontend/knowledge_static_value_generator_branch_reject_literal.sio"
GENERATOR_BRANCH_DYNAMIC_GAP="tests/frontend/knowledge_static_value_generator_branch_dynamic_gap.sio"
GENERATOR_BRANCH_FLOW_POSITIVE="tests/frontend/knowledge_static_value_generator_branch_flow_positive.sio"
GENERATOR_BRANCH_FLOW_NEGATIVE_LITERAL="tests/frontend/knowledge_static_value_generator_branch_flow_reject_literal.sio"
GENERATOR_BRANCH_FLOW_DYNAMIC_GAP="tests/frontend/knowledge_static_value_generator_branch_flow_dynamic_gap.sio"
GENERATOR_BRANCH_COND_POSITIVE="tests/frontend/knowledge_static_value_generator_branch_cond_positive.sio"
GENERATOR_BRANCH_COND_NEGATIVE_LITERAL="tests/frontend/knowledge_static_value_generator_branch_cond_reject_literal.sio"
GENERATOR_BRANCH_COND_DYNAMIC_GAP="tests/frontend/knowledge_static_value_generator_branch_cond_dynamic_gap.sio"
GENERATOR_BRANCH_RETURN_COND_POSITIVE="tests/frontend/knowledge_static_value_generator_branch_return_cond_positive.sio"
GENERATOR_BRANCH_RETURN_COND_NEGATIVE_LITERAL="tests/frontend/knowledge_static_value_generator_branch_return_cond_reject_literal.sio"
GENERATOR_BRANCH_RETURN_COND_DYNAMIC_GAP="tests/frontend/knowledge_static_value_generator_branch_return_cond_dynamic_gap.sio"
GENERATOR_HELPER_BRANCH_COND_POSITIVE="tests/frontend/knowledge_static_value_generator_helper_branch_cond_positive.sio"
GENERATOR_HELPER_BRANCH_COND_NEGATIVE_LITERAL="tests/frontend/knowledge_static_value_generator_helper_branch_cond_reject_literal.sio"
GENERATOR_HELPER_BRANCH_COND_DYNAMIC_GAP="tests/frontend/knowledge_static_value_generator_helper_branch_cond_dynamic_gap.sio"
GENERATOR_HELPER_BRANCH_FLOW_POSITIVE="tests/frontend/knowledge_static_value_generator_helper_branch_flow_positive.sio"
GENERATOR_HELPER_BRANCH_FLOW_NEGATIVE_LITERAL="tests/frontend/knowledge_static_value_generator_helper_branch_flow_reject_literal.sio"
GENERATOR_HELPER_BRANCH_FLOW_DYNAMIC_GAP="tests/frontend/knowledge_static_value_generator_helper_branch_flow_dynamic_gap.sio"
GENERATOR_NESTED_HELPER_POSITIVE="tests/frontend/knowledge_static_value_generator_nested_helper_positive.sio"
GENERATOR_NESTED_HELPER_NEGATIVE_LITERAL="tests/frontend/knowledge_static_value_generator_nested_helper_reject_literal.sio"
GENERATOR_NESTED_HELPER_DYNAMIC_GAP="tests/frontend/knowledge_static_value_generator_nested_helper_dynamic_gap.sio"

POSITIVE_LOG="$TMP_DIR/knowledge-static-value-positive.log"
if bin/souc run "$PROBE" -- "$POSITIVE" >"$POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$POSITIVE_LOG"; then
  printf 'PASS  %s accepted literal Knowledge<T> initializer satisfying proof context\n' "$POSITIVE"
else
  printf 'FAIL  %s did not accept the satisfying literal Knowledge<T> initializer\n' "$POSITIVE" >&2
  cat "$POSITIVE_LOG" >&2
  exit 1
fi

NEGATIVE_LITERAL_LOG="$TMP_DIR/knowledge-static-value-reject-literal.log"
if bin/souc run "$PROBE" -- "$NEGATIVE_LITERAL" >"$NEGATIVE_LITERAL_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=1' "$NEGATIVE_LITERAL_LOG"; then
  printf 'PASS  %s rejected literal Knowledge<T> initializer violating proof context\n' "$NEGATIVE_LITERAL"
else
  printf 'FAIL  %s did not reject the violating literal Knowledge<T> initializer\n' "$NEGATIVE_LITERAL" >&2
  cat "$NEGATIVE_LITERAL_LOG" >&2
  exit 1
fi

DYNAMIC_GAP_LOG="$TMP_DIR/knowledge-static-value-dynamic-gap.log"
if bin/souc run "$PROBE" -- "$DYNAMIC_GAP" >"$DYNAMIC_GAP_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$DYNAMIC_GAP_LOG"; then
  printf 'PASS  %s classified dynamic value as runtime-check gap, not static rejection\n' "$DYNAMIC_GAP"
else
  printf 'FAIL  %s did not preserve the dynamic/runtime gap classification\n' "$DYNAMIC_GAP" >&2
  cat "$DYNAMIC_GAP_LOG" >&2
  exit 1
fi

LOCAL_POSITIVE_LOG="$TMP_DIR/knowledge-static-value-local-positive.log"
if bin/souc run "$PROBE" -- "$LOCAL_POSITIVE" >"$LOCAL_POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$LOCAL_POSITIVE_LOG"; then
  printf 'PASS  %s accepted local literal Knowledge<T> binding satisfying proof context\n' "$LOCAL_POSITIVE"
else
  printf 'FAIL  %s did not accept the satisfying local literal Knowledge<T> binding\n' "$LOCAL_POSITIVE" >&2
  cat "$LOCAL_POSITIVE_LOG" >&2
  exit 1
fi

LOCAL_NEGATIVE_LITERAL_LOG="$TMP_DIR/knowledge-static-value-local-reject-literal.log"
if bin/souc run "$PROBE" -- "$LOCAL_NEGATIVE_LITERAL" >"$LOCAL_NEGATIVE_LITERAL_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=1' "$LOCAL_NEGATIVE_LITERAL_LOG"; then
  printf 'PASS  %s rejected local literal Knowledge<T> binding violating proof context\n' "$LOCAL_NEGATIVE_LITERAL"
else
  printf 'FAIL  %s did not reject the violating local literal Knowledge<T> binding\n' "$LOCAL_NEGATIVE_LITERAL" >&2
  cat "$LOCAL_NEGATIVE_LITERAL_LOG" >&2
  exit 1
fi

VAR_NEGATIVE_LITERAL_LOG="$TMP_DIR/knowledge-static-value-var-reject-literal.log"
if bin/souc run "$PROBE" -- "$VAR_NEGATIVE_LITERAL" >"$VAR_NEGATIVE_LITERAL_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=1' "$VAR_NEGATIVE_LITERAL_LOG"; then
  printf 'PASS  %s rejected var literal Knowledge<T> binding violating proof context\n' "$VAR_NEGATIVE_LITERAL"
else
  printf 'FAIL  %s did not reject the violating var literal Knowledge<T> binding\n' "$VAR_NEGATIVE_LITERAL" >&2
  cat "$VAR_NEGATIVE_LITERAL_LOG" >&2
  exit 1
fi

CALL_POSITIVE_LOG="$TMP_DIR/knowledge-static-value-call-positive.log"
if bin/souc run "$PROBE" -- "$CALL_POSITIVE" >"$CALL_POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$CALL_POSITIVE_LOG"; then
  printf 'PASS  %s accepted literal Knowledge<T> call argument satisfying proof context\n' "$CALL_POSITIVE"
else
  printf 'FAIL  %s did not accept the satisfying literal Knowledge<T> call argument\n' "$CALL_POSITIVE" >&2
  cat "$CALL_POSITIVE_LOG" >&2
  exit 1
fi

CALL_NEGATIVE_LITERAL_LOG="$TMP_DIR/knowledge-static-value-call-reject-literal.log"
if bin/souc run "$PROBE" -- "$CALL_NEGATIVE_LITERAL" >"$CALL_NEGATIVE_LITERAL_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=1' "$CALL_NEGATIVE_LITERAL_LOG"; then
  printf 'PASS  %s rejected literal Knowledge<T> call argument violating proof context\n' "$CALL_NEGATIVE_LITERAL"
else
  printf 'FAIL  %s did not reject the violating literal Knowledge<T> call argument\n' "$CALL_NEGATIVE_LITERAL" >&2
  cat "$CALL_NEGATIVE_LITERAL_LOG" >&2
  exit 1
fi

CALL_DYNAMIC_GAP_LOG="$TMP_DIR/knowledge-static-value-call-dynamic-gap.log"
if bin/souc run "$PROBE" -- "$CALL_DYNAMIC_GAP" >"$CALL_DYNAMIC_GAP_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$CALL_DYNAMIC_GAP_LOG"; then
  printf 'PASS  %s classified dynamic Knowledge<T> call argument as runtime-check gap\n' "$CALL_DYNAMIC_GAP"
else
  printf 'FAIL  %s did not preserve the dynamic call-argument gap classification\n' "$CALL_DYNAMIC_GAP" >&2
  cat "$CALL_DYNAMIC_GAP_LOG" >&2
  exit 1
fi

ASSIGN_POSITIVE_LOG="$TMP_DIR/knowledge-static-value-assign-positive.log"
if bin/souc run "$PROBE" -- "$ASSIGN_POSITIVE" >"$ASSIGN_POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$ASSIGN_POSITIVE_LOG"; then
  printf 'PASS  %s accepted literal Knowledge<T> assignment satisfying proof context\n' "$ASSIGN_POSITIVE"
else
  printf 'FAIL  %s did not accept the satisfying literal Knowledge<T> assignment\n' "$ASSIGN_POSITIVE" >&2
  cat "$ASSIGN_POSITIVE_LOG" >&2
  exit 1
fi

ASSIGN_NEGATIVE_LITERAL_LOG="$TMP_DIR/knowledge-static-value-assign-reject-literal.log"
if bin/souc run "$PROBE" -- "$ASSIGN_NEGATIVE_LITERAL" >"$ASSIGN_NEGATIVE_LITERAL_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=1' "$ASSIGN_NEGATIVE_LITERAL_LOG"; then
  printf 'PASS  %s rejected literal Knowledge<T> assignment violating proof context\n' "$ASSIGN_NEGATIVE_LITERAL"
else
  printf 'FAIL  %s did not reject the violating literal Knowledge<T> assignment\n' "$ASSIGN_NEGATIVE_LITERAL" >&2
  cat "$ASSIGN_NEGATIVE_LITERAL_LOG" >&2
  exit 1
fi

ASSIGN_DYNAMIC_GAP_LOG="$TMP_DIR/knowledge-static-value-assign-dynamic-gap.log"
if bin/souc run "$PROBE" -- "$ASSIGN_DYNAMIC_GAP" >"$ASSIGN_DYNAMIC_GAP_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$ASSIGN_DYNAMIC_GAP_LOG"; then
  printf 'PASS  %s classified dynamic Knowledge<T> assignment as runtime-check gap\n' "$ASSIGN_DYNAMIC_GAP"
else
  printf 'FAIL  %s did not preserve the dynamic assignment gap classification\n' "$ASSIGN_DYNAMIC_GAP" >&2
  cat "$ASSIGN_DYNAMIC_GAP_LOG" >&2
  exit 1
fi

UNIT_POSITIVE_LOG="$TMP_DIR/knowledge-static-value-unit-positive.log"
if bin/souc run "$PROBE" -- "$UNIT_POSITIVE" >"$UNIT_POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$UNIT_POSITIVE_LOG"; then
  printf 'PASS  %s accepted unit-suffixed literal Knowledge<T> value satisfying proof context\n' "$UNIT_POSITIVE"
else
  printf 'FAIL  %s did not accept the satisfying unit-suffixed literal Knowledge<T> value\n' "$UNIT_POSITIVE" >&2
  cat "$UNIT_POSITIVE_LOG" >&2
  exit 1
fi

UNIT_NEGATIVE_LITERAL_LOG="$TMP_DIR/knowledge-static-value-unit-reject-literal.log"
if bin/souc run "$PROBE" -- "$UNIT_NEGATIVE_LITERAL" >"$UNIT_NEGATIVE_LITERAL_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=1' "$UNIT_NEGATIVE_LITERAL_LOG"; then
  printf 'PASS  %s rejected unit-suffixed literal Knowledge<T> value violating proof context\n' "$UNIT_NEGATIVE_LITERAL"
else
  printf 'FAIL  %s did not reject the violating unit-suffixed literal Knowledge<T> value\n' "$UNIT_NEGATIVE_LITERAL" >&2
  cat "$UNIT_NEGATIVE_LITERAL_LOG" >&2
  exit 1
fi

UNIT_DYNAMIC_GAP_LOG="$TMP_DIR/knowledge-static-value-unit-dynamic-gap.log"
if bin/souc run "$PROBE" -- "$UNIT_DYNAMIC_GAP" >"$UNIT_DYNAMIC_GAP_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$UNIT_DYNAMIC_GAP_LOG"; then
  printf 'PASS  %s classified dynamic unit-suffixed Knowledge<T> value as runtime-check gap\n' "$UNIT_DYNAMIC_GAP"
else
  printf 'FAIL  %s did not preserve the dynamic unit-suffixed gap classification\n' "$UNIT_DYNAMIC_GAP" >&2
  cat "$UNIT_DYNAMIC_GAP_LOG" >&2
  exit 1
fi

GENERATOR_POSITIVE_LOG="$TMP_DIR/knowledge-static-value-generator-positive.log"
if bin/souc run "$PROBE" -- "$GENERATOR_POSITIVE" >"$GENERATOR_POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_POSITIVE_LOG"; then
  printf 'PASS  %s accepted simple generated Knowledge<T> value satisfying call-site proof context\n' "$GENERATOR_POSITIVE"
else
  printf 'FAIL  %s did not accept the satisfying simple generated Knowledge<T> value\n' "$GENERATOR_POSITIVE" >&2
  cat "$GENERATOR_POSITIVE_LOG" >&2
  exit 1
fi

GENERATOR_NEGATIVE_LITERAL_LOG="$TMP_DIR/knowledge-static-value-generator-reject-literal.log"
if bin/souc run "$PROBE" -- "$GENERATOR_NEGATIVE_LITERAL" >"$GENERATOR_NEGATIVE_LITERAL_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=1' "$GENERATOR_NEGATIVE_LITERAL_LOG"; then
  printf 'PASS  %s rejected simple generated Knowledge<T> value violating call-site proof context\n' "$GENERATOR_NEGATIVE_LITERAL"
else
  printf 'FAIL  %s did not reject the violating simple generated Knowledge<T> value\n' "$GENERATOR_NEGATIVE_LITERAL" >&2
  cat "$GENERATOR_NEGATIVE_LITERAL_LOG" >&2
  exit 1
fi

GENERATOR_DYNAMIC_GAP_LOG="$TMP_DIR/knowledge-static-value-generator-dynamic-gap.log"
if bin/souc run "$PROBE" -- "$GENERATOR_DYNAMIC_GAP" >"$GENERATOR_DYNAMIC_GAP_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_DYNAMIC_GAP_LOG"; then
  printf 'PASS  %s classified dynamic generated Knowledge<T> value as runtime-check gap\n' "$GENERATOR_DYNAMIC_GAP"
else
  printf 'FAIL  %s did not preserve the dynamic generated-value gap classification\n' "$GENERATOR_DYNAMIC_GAP" >&2
  cat "$GENERATOR_DYNAMIC_GAP_LOG" >&2
  exit 1
fi

GENERATOR_ARG_POSITIVE_LOG="$TMP_DIR/knowledge-static-value-generator-arg-positive.log"
if bin/souc run "$PROBE" -- "$GENERATOR_ARG_POSITIVE" >"$GENERATOR_ARG_POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_ARG_POSITIVE_LOG"; then
  printf 'PASS  %s accepted simple generated Knowledge<T> value with literal argument satisfying call-site proof context\n' "$GENERATOR_ARG_POSITIVE"
else
  printf 'FAIL  %s did not accept the satisfying simple generated Knowledge<T> value with literal argument\n' "$GENERATOR_ARG_POSITIVE" >&2
  cat "$GENERATOR_ARG_POSITIVE_LOG" >&2
  exit 1
fi

GENERATOR_ARG_NEGATIVE_LITERAL_LOG="$TMP_DIR/knowledge-static-value-generator-arg-reject-literal.log"
if bin/souc run "$PROBE" -- "$GENERATOR_ARG_NEGATIVE_LITERAL" >"$GENERATOR_ARG_NEGATIVE_LITERAL_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=1' "$GENERATOR_ARG_NEGATIVE_LITERAL_LOG"; then
  printf 'PASS  %s rejected simple generated Knowledge<T> value with literal argument violating call-site proof context\n' "$GENERATOR_ARG_NEGATIVE_LITERAL"
else
  printf 'FAIL  %s did not reject the violating simple generated Knowledge<T> value with literal argument\n' "$GENERATOR_ARG_NEGATIVE_LITERAL" >&2
  cat "$GENERATOR_ARG_NEGATIVE_LITERAL_LOG" >&2
  exit 1
fi

GENERATOR_ARG_DYNAMIC_GAP_LOG="$TMP_DIR/knowledge-static-value-generator-arg-dynamic-gap.log"
if bin/souc run "$PROBE" -- "$GENERATOR_ARG_DYNAMIC_GAP" >"$GENERATOR_ARG_DYNAMIC_GAP_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_ARG_DYNAMIC_GAP_LOG"; then
  printf 'PASS  %s classified dynamic generated Knowledge<T> argument as runtime-check gap\n' "$GENERATOR_ARG_DYNAMIC_GAP"
else
  printf 'FAIL  %s did not preserve the dynamic generated-argument gap classification\n' "$GENERATOR_ARG_DYNAMIC_GAP" >&2
  cat "$GENERATOR_ARG_DYNAMIC_GAP_LOG" >&2
  exit 1
fi

GENERATOR_EXPR_POSITIVE_LOG="$TMP_DIR/knowledge-static-value-generator-expr-positive.log"
if bin/souc run "$PROBE" -- "$GENERATOR_EXPR_POSITIVE" >"$GENERATOR_EXPR_POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_EXPR_POSITIVE_LOG"; then
  printf 'PASS  %s accepted simple generated Knowledge<T> value with const-evaluable expression satisfying call-site proof context\n' "$GENERATOR_EXPR_POSITIVE"
else
  printf 'FAIL  %s did not accept the satisfying simple generated Knowledge<T> value with const-evaluable expression\n' "$GENERATOR_EXPR_POSITIVE" >&2
  cat "$GENERATOR_EXPR_POSITIVE_LOG" >&2
  exit 1
fi

GENERATOR_EXPR_NEGATIVE_LITERAL_LOG="$TMP_DIR/knowledge-static-value-generator-expr-reject-literal.log"
if bin/souc run "$PROBE" -- "$GENERATOR_EXPR_NEGATIVE_LITERAL" >"$GENERATOR_EXPR_NEGATIVE_LITERAL_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=1' "$GENERATOR_EXPR_NEGATIVE_LITERAL_LOG"; then
  printf 'PASS  %s rejected simple generated Knowledge<T> value with const-evaluable expression violating call-site proof context\n' "$GENERATOR_EXPR_NEGATIVE_LITERAL"
else
  printf 'FAIL  %s did not reject the violating simple generated Knowledge<T> value with const-evaluable expression\n' "$GENERATOR_EXPR_NEGATIVE_LITERAL" >&2
  cat "$GENERATOR_EXPR_NEGATIVE_LITERAL_LOG" >&2
  exit 1
fi

GENERATOR_EXPR_DYNAMIC_GAP_LOG="$TMP_DIR/knowledge-static-value-generator-expr-dynamic-gap.log"
if bin/souc run "$PROBE" -- "$GENERATOR_EXPR_DYNAMIC_GAP" >"$GENERATOR_EXPR_DYNAMIC_GAP_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_EXPR_DYNAMIC_GAP_LOG"; then
  printf 'PASS  %s classified dynamic generated Knowledge<T> expression as runtime-check gap\n' "$GENERATOR_EXPR_DYNAMIC_GAP"
else
  printf 'FAIL  %s did not preserve the dynamic generated-expression gap classification\n' "$GENERATOR_EXPR_DYNAMIC_GAP" >&2
  cat "$GENERATOR_EXPR_DYNAMIC_GAP_LOG" >&2
  exit 1
fi

GENERATOR_LOCAL_POSITIVE_LOG="$TMP_DIR/knowledge-static-value-generator-local-positive.log"
if bin/souc run "$PROBE" -- "$GENERATOR_LOCAL_POSITIVE" >"$GENERATOR_LOCAL_POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_LOCAL_POSITIVE_LOG"; then
  printf 'PASS  %s accepted simple generated Knowledge<T> value through local const binding satisfying call-site proof context\n' "$GENERATOR_LOCAL_POSITIVE"
else
  printf 'FAIL  %s did not accept the satisfying simple generated Knowledge<T> value through local const binding\n' "$GENERATOR_LOCAL_POSITIVE" >&2
  cat "$GENERATOR_LOCAL_POSITIVE_LOG" >&2
  exit 1
fi

GENERATOR_LOCAL_NEGATIVE_LITERAL_LOG="$TMP_DIR/knowledge-static-value-generator-local-reject-literal.log"
if bin/souc run "$PROBE" -- "$GENERATOR_LOCAL_NEGATIVE_LITERAL" >"$GENERATOR_LOCAL_NEGATIVE_LITERAL_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=1' "$GENERATOR_LOCAL_NEGATIVE_LITERAL_LOG"; then
  printf 'PASS  %s rejected simple generated Knowledge<T> value through local const binding violating call-site proof context\n' "$GENERATOR_LOCAL_NEGATIVE_LITERAL"
else
  printf 'FAIL  %s did not reject the violating simple generated Knowledge<T> value through local const binding\n' "$GENERATOR_LOCAL_NEGATIVE_LITERAL" >&2
  cat "$GENERATOR_LOCAL_NEGATIVE_LITERAL_LOG" >&2
  exit 1
fi

GENERATOR_LOCAL_DYNAMIC_GAP_LOG="$TMP_DIR/knowledge-static-value-generator-local-dynamic-gap.log"
if bin/souc run "$PROBE" -- "$GENERATOR_LOCAL_DYNAMIC_GAP" >"$GENERATOR_LOCAL_DYNAMIC_GAP_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_LOCAL_DYNAMIC_GAP_LOG"; then
  printf 'PASS  %s classified dynamic generated Knowledge<T> local binding as runtime-check gap\n' "$GENERATOR_LOCAL_DYNAMIC_GAP"
else
  printf 'FAIL  %s did not preserve the dynamic generated-local gap classification\n' "$GENERATOR_LOCAL_DYNAMIC_GAP" >&2
  cat "$GENERATOR_LOCAL_DYNAMIC_GAP_LOG" >&2
  exit 1
fi

GENERATOR_ASSIGN_POSITIVE_LOG="$TMP_DIR/knowledge-static-value-generator-assign-positive.log"
if bin/souc run "$PROBE" -- "$GENERATOR_ASSIGN_POSITIVE" >"$GENERATOR_ASSIGN_POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_ASSIGN_POSITIVE_LOG"; then
  printf 'PASS  %s accepted simple generated Knowledge<T> value through local assignment satisfying call-site proof context\n' "$GENERATOR_ASSIGN_POSITIVE"
else
  printf 'FAIL  %s did not accept the satisfying simple generated Knowledge<T> value through local assignment\n' "$GENERATOR_ASSIGN_POSITIVE" >&2
  cat "$GENERATOR_ASSIGN_POSITIVE_LOG" >&2
  exit 1
fi

GENERATOR_ASSIGN_NEGATIVE_LITERAL_LOG="$TMP_DIR/knowledge-static-value-generator-assign-reject-literal.log"
if bin/souc run "$PROBE" -- "$GENERATOR_ASSIGN_NEGATIVE_LITERAL" >"$GENERATOR_ASSIGN_NEGATIVE_LITERAL_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=1' "$GENERATOR_ASSIGN_NEGATIVE_LITERAL_LOG"; then
  printf 'PASS  %s rejected simple generated Knowledge<T> value through local assignment violating call-site proof context\n' "$GENERATOR_ASSIGN_NEGATIVE_LITERAL"
else
  printf 'FAIL  %s did not reject the violating simple generated Knowledge<T> value through local assignment\n' "$GENERATOR_ASSIGN_NEGATIVE_LITERAL" >&2
  cat "$GENERATOR_ASSIGN_NEGATIVE_LITERAL_LOG" >&2
  exit 1
fi

GENERATOR_ASSIGN_DYNAMIC_GAP_LOG="$TMP_DIR/knowledge-static-value-generator-assign-dynamic-gap.log"
if bin/souc run "$PROBE" -- "$GENERATOR_ASSIGN_DYNAMIC_GAP" >"$GENERATOR_ASSIGN_DYNAMIC_GAP_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_ASSIGN_DYNAMIC_GAP_LOG"; then
  printf 'PASS  %s classified dynamic generated Knowledge<T> assignment as runtime-check gap\n' "$GENERATOR_ASSIGN_DYNAMIC_GAP"
else
  printf 'FAIL  %s did not preserve the dynamic generated-assignment gap classification\n' "$GENERATOR_ASSIGN_DYNAMIC_GAP" >&2
  cat "$GENERATOR_ASSIGN_DYNAMIC_GAP_LOG" >&2
  exit 1
fi

GENERATOR_COMPOUND_POSITIVE_LOG="$TMP_DIR/knowledge-static-value-generator-compound-positive.log"
if bin/souc run "$PROBE" -- "$GENERATOR_COMPOUND_POSITIVE" >"$GENERATOR_COMPOUND_POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_COMPOUND_POSITIVE_LOG"; then
  printf 'PASS  %s accepted simple generated Knowledge<T> value through compound assignment satisfying call-site proof context\n' "$GENERATOR_COMPOUND_POSITIVE"
else
  printf 'FAIL  %s did not accept the satisfying simple generated Knowledge<T> value through compound assignment\n' "$GENERATOR_COMPOUND_POSITIVE" >&2
  cat "$GENERATOR_COMPOUND_POSITIVE_LOG" >&2
  exit 1
fi

GENERATOR_COMPOUND_NEGATIVE_LITERAL_LOG="$TMP_DIR/knowledge-static-value-generator-compound-reject-literal.log"
if bin/souc run "$PROBE" -- "$GENERATOR_COMPOUND_NEGATIVE_LITERAL" >"$GENERATOR_COMPOUND_NEGATIVE_LITERAL_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=1' "$GENERATOR_COMPOUND_NEGATIVE_LITERAL_LOG"; then
  printf 'PASS  %s rejected simple generated Knowledge<T> value through compound assignment violating call-site proof context\n' "$GENERATOR_COMPOUND_NEGATIVE_LITERAL"
else
  printf 'FAIL  %s did not reject the violating simple generated Knowledge<T> value through compound assignment\n' "$GENERATOR_COMPOUND_NEGATIVE_LITERAL" >&2
  cat "$GENERATOR_COMPOUND_NEGATIVE_LITERAL_LOG" >&2
  exit 1
fi

GENERATOR_COMPOUND_DYNAMIC_GAP_LOG="$TMP_DIR/knowledge-static-value-generator-compound-dynamic-gap.log"
if bin/souc run "$PROBE" -- "$GENERATOR_COMPOUND_DYNAMIC_GAP" >"$GENERATOR_COMPOUND_DYNAMIC_GAP_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_COMPOUND_DYNAMIC_GAP_LOG"; then
  printf 'PASS  %s classified dynamic generated Knowledge<T> compound assignment as runtime-check gap\n' "$GENERATOR_COMPOUND_DYNAMIC_GAP"
else
  printf 'FAIL  %s did not preserve the dynamic generated-compound-assignment gap classification\n' "$GENERATOR_COMPOUND_DYNAMIC_GAP" >&2
  cat "$GENERATOR_COMPOUND_DYNAMIC_GAP_LOG" >&2
  exit 1
fi

GENERATOR_HELPER_POSITIVE_LOG="$TMP_DIR/knowledge-static-value-generator-helper-positive.log"
if bin/souc run "$PROBE" -- "$GENERATOR_HELPER_POSITIVE" >"$GENERATOR_HELPER_POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_HELPER_POSITIVE_LOG"; then
  printf 'PASS  %s accepted simple generated Knowledge<T> value through helper call satisfying call-site proof context\n' "$GENERATOR_HELPER_POSITIVE"
else
  printf 'FAIL  %s did not accept the satisfying simple generated Knowledge<T> value through helper call\n' "$GENERATOR_HELPER_POSITIVE" >&2
  cat "$GENERATOR_HELPER_POSITIVE_LOG" >&2
  exit 1
fi

GENERATOR_HELPER_NEGATIVE_LITERAL_LOG="$TMP_DIR/knowledge-static-value-generator-helper-reject-literal.log"
if bin/souc run "$PROBE" -- "$GENERATOR_HELPER_NEGATIVE_LITERAL" >"$GENERATOR_HELPER_NEGATIVE_LITERAL_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=1' "$GENERATOR_HELPER_NEGATIVE_LITERAL_LOG"; then
  printf 'PASS  %s rejected simple generated Knowledge<T> value through helper call violating call-site proof context\n' "$GENERATOR_HELPER_NEGATIVE_LITERAL"
else
  printf 'FAIL  %s did not reject the violating simple generated Knowledge<T> value through helper call\n' "$GENERATOR_HELPER_NEGATIVE_LITERAL" >&2
  cat "$GENERATOR_HELPER_NEGATIVE_LITERAL_LOG" >&2
  exit 1
fi

GENERATOR_HELPER_DYNAMIC_GAP_LOG="$TMP_DIR/knowledge-static-value-generator-helper-dynamic-gap.log"
if bin/souc run "$PROBE" -- "$GENERATOR_HELPER_DYNAMIC_GAP" >"$GENERATOR_HELPER_DYNAMIC_GAP_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_HELPER_DYNAMIC_GAP_LOG"; then
  printf 'PASS  %s classified dynamic generated Knowledge<T> helper call as runtime-check gap\n' "$GENERATOR_HELPER_DYNAMIC_GAP"
else
  printf 'FAIL  %s did not preserve the dynamic generated-helper gap classification\n' "$GENERATOR_HELPER_DYNAMIC_GAP" >&2
  cat "$GENERATOR_HELPER_DYNAMIC_GAP_LOG" >&2
  exit 1
fi

GENERATOR_HELPER_LOCAL_POSITIVE_LOG="$TMP_DIR/knowledge-static-value-generator-helper-local-positive.log"
if bin/souc run "$PROBE" -- "$GENERATOR_HELPER_LOCAL_POSITIVE" >"$GENERATOR_HELPER_LOCAL_POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_HELPER_LOCAL_POSITIVE_LOG"; then
  printf 'PASS  %s accepted simple generated Knowledge<T> value through helper-local binding satisfying call-site proof context\n' "$GENERATOR_HELPER_LOCAL_POSITIVE"
else
  printf 'FAIL  %s did not accept the satisfying simple generated Knowledge<T> value through helper-local binding\n' "$GENERATOR_HELPER_LOCAL_POSITIVE" >&2
  cat "$GENERATOR_HELPER_LOCAL_POSITIVE_LOG" >&2
  exit 1
fi

GENERATOR_HELPER_LOCAL_NEGATIVE_LITERAL_LOG="$TMP_DIR/knowledge-static-value-generator-helper-local-reject-literal.log"
if bin/souc run "$PROBE" -- "$GENERATOR_HELPER_LOCAL_NEGATIVE_LITERAL" >"$GENERATOR_HELPER_LOCAL_NEGATIVE_LITERAL_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=1' "$GENERATOR_HELPER_LOCAL_NEGATIVE_LITERAL_LOG"; then
  printf 'PASS  %s rejected simple generated Knowledge<T> value through helper-local binding violating call-site proof context\n' "$GENERATOR_HELPER_LOCAL_NEGATIVE_LITERAL"
else
  printf 'FAIL  %s did not reject the violating simple generated Knowledge<T> value through helper-local binding\n' "$GENERATOR_HELPER_LOCAL_NEGATIVE_LITERAL" >&2
  cat "$GENERATOR_HELPER_LOCAL_NEGATIVE_LITERAL_LOG" >&2
  exit 1
fi

GENERATOR_HELPER_LOCAL_DYNAMIC_GAP_LOG="$TMP_DIR/knowledge-static-value-generator-helper-local-dynamic-gap.log"
if bin/souc run "$PROBE" -- "$GENERATOR_HELPER_LOCAL_DYNAMIC_GAP" >"$GENERATOR_HELPER_LOCAL_DYNAMIC_GAP_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_HELPER_LOCAL_DYNAMIC_GAP_LOG"; then
  printf 'PASS  %s classified dynamic generated Knowledge<T> helper-local binding as runtime-check gap\n' "$GENERATOR_HELPER_LOCAL_DYNAMIC_GAP"
else
  printf 'FAIL  %s did not preserve the dynamic generated-helper-local gap classification\n' "$GENERATOR_HELPER_LOCAL_DYNAMIC_GAP" >&2
  cat "$GENERATOR_HELPER_LOCAL_DYNAMIC_GAP_LOG" >&2
  exit 1
fi

GENERATOR_HELPER_ASSIGN_POSITIVE_LOG="$TMP_DIR/knowledge-static-value-generator-helper-assign-positive.log"
if bin/souc run "$PROBE" -- "$GENERATOR_HELPER_ASSIGN_POSITIVE" >"$GENERATOR_HELPER_ASSIGN_POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_HELPER_ASSIGN_POSITIVE_LOG"; then
  printf 'PASS  %s accepted simple generated Knowledge<T> value through helper-local assignment satisfying call-site proof context\n' "$GENERATOR_HELPER_ASSIGN_POSITIVE"
else
  printf 'FAIL  %s did not accept the satisfying simple generated Knowledge<T> value through helper-local assignment\n' "$GENERATOR_HELPER_ASSIGN_POSITIVE" >&2
  cat "$GENERATOR_HELPER_ASSIGN_POSITIVE_LOG" >&2
  exit 1
fi

GENERATOR_HELPER_ASSIGN_NEGATIVE_LITERAL_LOG="$TMP_DIR/knowledge-static-value-generator-helper-assign-reject-literal.log"
if bin/souc run "$PROBE" -- "$GENERATOR_HELPER_ASSIGN_NEGATIVE_LITERAL" >"$GENERATOR_HELPER_ASSIGN_NEGATIVE_LITERAL_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=1' "$GENERATOR_HELPER_ASSIGN_NEGATIVE_LITERAL_LOG"; then
  printf 'PASS  %s rejected simple generated Knowledge<T> value through helper-local assignment violating call-site proof context\n' "$GENERATOR_HELPER_ASSIGN_NEGATIVE_LITERAL"
else
  printf 'FAIL  %s did not reject the violating simple generated Knowledge<T> value through helper-local assignment\n' "$GENERATOR_HELPER_ASSIGN_NEGATIVE_LITERAL" >&2
  cat "$GENERATOR_HELPER_ASSIGN_NEGATIVE_LITERAL_LOG" >&2
  exit 1
fi

GENERATOR_HELPER_ASSIGN_DYNAMIC_GAP_LOG="$TMP_DIR/knowledge-static-value-generator-helper-assign-dynamic-gap.log"
if bin/souc run "$PROBE" -- "$GENERATOR_HELPER_ASSIGN_DYNAMIC_GAP" >"$GENERATOR_HELPER_ASSIGN_DYNAMIC_GAP_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_HELPER_ASSIGN_DYNAMIC_GAP_LOG"; then
  printf 'PASS  %s classified dynamic generated Knowledge<T> helper-local assignment as runtime-check gap\n' "$GENERATOR_HELPER_ASSIGN_DYNAMIC_GAP"
else
  printf 'FAIL  %s did not preserve the dynamic generated-helper-assignment gap classification\n' "$GENERATOR_HELPER_ASSIGN_DYNAMIC_GAP" >&2
  cat "$GENERATOR_HELPER_ASSIGN_DYNAMIC_GAP_LOG" >&2
  exit 1
fi

GENERATOR_BRANCH_POSITIVE_LOG="$TMP_DIR/knowledge-static-value-generator-branch-positive.log"
if bin/souc run "$PROBE" -- "$GENERATOR_BRANCH_POSITIVE" >"$GENERATOR_BRANCH_POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_BRANCH_POSITIVE_LOG"; then
  printf 'PASS  %s accepted simple generated Knowledge<T> value through static branch satisfying call-site proof context\n' "$GENERATOR_BRANCH_POSITIVE"
else
  printf 'FAIL  %s did not accept the satisfying simple generated Knowledge<T> value through static branch\n' "$GENERATOR_BRANCH_POSITIVE" >&2
  cat "$GENERATOR_BRANCH_POSITIVE_LOG" >&2
  exit 1
fi

GENERATOR_BRANCH_NEGATIVE_LITERAL_LOG="$TMP_DIR/knowledge-static-value-generator-branch-reject-literal.log"
if bin/souc run "$PROBE" -- "$GENERATOR_BRANCH_NEGATIVE_LITERAL" >"$GENERATOR_BRANCH_NEGATIVE_LITERAL_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=1' "$GENERATOR_BRANCH_NEGATIVE_LITERAL_LOG"; then
  printf 'PASS  %s rejected simple generated Knowledge<T> value through static branch violating call-site proof context\n' "$GENERATOR_BRANCH_NEGATIVE_LITERAL"
else
  printf 'FAIL  %s did not reject the violating simple generated Knowledge<T> value through static branch\n' "$GENERATOR_BRANCH_NEGATIVE_LITERAL" >&2
  cat "$GENERATOR_BRANCH_NEGATIVE_LITERAL_LOG" >&2
  exit 1
fi

GENERATOR_BRANCH_DYNAMIC_GAP_LOG="$TMP_DIR/knowledge-static-value-generator-branch-dynamic-gap.log"
if bin/souc run "$PROBE" -- "$GENERATOR_BRANCH_DYNAMIC_GAP" >"$GENERATOR_BRANCH_DYNAMIC_GAP_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_BRANCH_DYNAMIC_GAP_LOG"; then
  printf 'PASS  %s classified dynamic generated Knowledge<T> branch as runtime-check gap\n' "$GENERATOR_BRANCH_DYNAMIC_GAP"
else
  printf 'FAIL  %s did not preserve the dynamic generated-branch gap classification\n' "$GENERATOR_BRANCH_DYNAMIC_GAP" >&2
  cat "$GENERATOR_BRANCH_DYNAMIC_GAP_LOG" >&2
  exit 1
fi

GENERATOR_BRANCH_FLOW_POSITIVE_LOG="$TMP_DIR/knowledge-static-value-generator-branch-flow-positive.log"
if bin/souc run "$PROBE" -- "$GENERATOR_BRANCH_FLOW_POSITIVE" >"$GENERATOR_BRANCH_FLOW_POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_BRANCH_FLOW_POSITIVE_LOG"; then
  printf 'PASS  %s accepted simple generated Knowledge<T> value through static branch local flow satisfying call-site proof context\n' "$GENERATOR_BRANCH_FLOW_POSITIVE"
else
  printf 'FAIL  %s did not accept the satisfying simple generated Knowledge<T> value through static branch local flow\n' "$GENERATOR_BRANCH_FLOW_POSITIVE" >&2
  cat "$GENERATOR_BRANCH_FLOW_POSITIVE_LOG" >&2
  exit 1
fi

GENERATOR_BRANCH_FLOW_NEGATIVE_LITERAL_LOG="$TMP_DIR/knowledge-static-value-generator-branch-flow-reject-literal.log"
if bin/souc run "$PROBE" -- "$GENERATOR_BRANCH_FLOW_NEGATIVE_LITERAL" >"$GENERATOR_BRANCH_FLOW_NEGATIVE_LITERAL_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=1' "$GENERATOR_BRANCH_FLOW_NEGATIVE_LITERAL_LOG"; then
  printf 'PASS  %s rejected simple generated Knowledge<T> value through static branch local flow violating call-site proof context\n' "$GENERATOR_BRANCH_FLOW_NEGATIVE_LITERAL"
else
  printf 'FAIL  %s did not reject the violating simple generated Knowledge<T> value through static branch local flow\n' "$GENERATOR_BRANCH_FLOW_NEGATIVE_LITERAL" >&2
  cat "$GENERATOR_BRANCH_FLOW_NEGATIVE_LITERAL_LOG" >&2
  exit 1
fi

GENERATOR_BRANCH_FLOW_DYNAMIC_GAP_LOG="$TMP_DIR/knowledge-static-value-generator-branch-flow-dynamic-gap.log"
if bin/souc run "$PROBE" -- "$GENERATOR_BRANCH_FLOW_DYNAMIC_GAP" >"$GENERATOR_BRANCH_FLOW_DYNAMIC_GAP_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_BRANCH_FLOW_DYNAMIC_GAP_LOG"; then
  printf 'PASS  %s classified dynamic generated Knowledge<T> branch local flow as runtime-check gap\n' "$GENERATOR_BRANCH_FLOW_DYNAMIC_GAP"
else
  printf 'FAIL  %s did not preserve the dynamic generated-branch-local-flow gap classification\n' "$GENERATOR_BRANCH_FLOW_DYNAMIC_GAP" >&2
  cat "$GENERATOR_BRANCH_FLOW_DYNAMIC_GAP_LOG" >&2
  exit 1
fi

GENERATOR_BRANCH_COND_POSITIVE_LOG="$TMP_DIR/knowledge-static-value-generator-branch-cond-positive.log"
if bin/souc run "$PROBE" -- "$GENERATOR_BRANCH_COND_POSITIVE" >"$GENERATOR_BRANCH_COND_POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_BRANCH_COND_POSITIVE_LOG"; then
  printf 'PASS  %s accepted simple generated Knowledge<T> value through const-evaluable branch condition satisfying call-site proof context\n' "$GENERATOR_BRANCH_COND_POSITIVE"
else
  printf 'FAIL  %s did not accept the satisfying simple generated Knowledge<T> value through const-evaluable branch condition\n' "$GENERATOR_BRANCH_COND_POSITIVE" >&2
  cat "$GENERATOR_BRANCH_COND_POSITIVE_LOG" >&2
  exit 1
fi

GENERATOR_BRANCH_COND_NEGATIVE_LITERAL_LOG="$TMP_DIR/knowledge-static-value-generator-branch-cond-reject-literal.log"
if bin/souc run "$PROBE" -- "$GENERATOR_BRANCH_COND_NEGATIVE_LITERAL" >"$GENERATOR_BRANCH_COND_NEGATIVE_LITERAL_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=1' "$GENERATOR_BRANCH_COND_NEGATIVE_LITERAL_LOG"; then
  printf 'PASS  %s rejected simple generated Knowledge<T> value through const-evaluable branch condition violating call-site proof context\n' "$GENERATOR_BRANCH_COND_NEGATIVE_LITERAL"
else
  printf 'FAIL  %s did not reject the violating simple generated Knowledge<T> value through const-evaluable branch condition\n' "$GENERATOR_BRANCH_COND_NEGATIVE_LITERAL" >&2
  cat "$GENERATOR_BRANCH_COND_NEGATIVE_LITERAL_LOG" >&2
  exit 1
fi

GENERATOR_BRANCH_COND_DYNAMIC_GAP_LOG="$TMP_DIR/knowledge-static-value-generator-branch-cond-dynamic-gap.log"
if bin/souc run "$PROBE" -- "$GENERATOR_BRANCH_COND_DYNAMIC_GAP" >"$GENERATOR_BRANCH_COND_DYNAMIC_GAP_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_BRANCH_COND_DYNAMIC_GAP_LOG"; then
  printf 'PASS  %s classified dynamic generated Knowledge<T> branch condition as runtime-check gap\n' "$GENERATOR_BRANCH_COND_DYNAMIC_GAP"
else
  printf 'FAIL  %s did not preserve the dynamic generated-branch-condition gap classification\n' "$GENERATOR_BRANCH_COND_DYNAMIC_GAP" >&2
  cat "$GENERATOR_BRANCH_COND_DYNAMIC_GAP_LOG" >&2
  exit 1
fi

GENERATOR_BRANCH_RETURN_COND_POSITIVE_LOG="$TMP_DIR/knowledge-static-value-generator-branch-return-cond-positive.log"
if bin/souc run "$PROBE" -- "$GENERATOR_BRANCH_RETURN_COND_POSITIVE" >"$GENERATOR_BRANCH_RETURN_COND_POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_BRANCH_RETURN_COND_POSITIVE_LOG"; then
  printf 'PASS  %s accepted direct-return generated Knowledge<T> value through const-evaluable branch condition satisfying call-site proof context\n' "$GENERATOR_BRANCH_RETURN_COND_POSITIVE"
else
  printf 'FAIL  %s did not accept the satisfying direct-return generated Knowledge<T> value through const-evaluable branch condition\n' "$GENERATOR_BRANCH_RETURN_COND_POSITIVE" >&2
  cat "$GENERATOR_BRANCH_RETURN_COND_POSITIVE_LOG" >&2
  exit 1
fi

GENERATOR_BRANCH_RETURN_COND_NEGATIVE_LITERAL_LOG="$TMP_DIR/knowledge-static-value-generator-branch-return-cond-reject-literal.log"
if bin/souc run "$PROBE" -- "$GENERATOR_BRANCH_RETURN_COND_NEGATIVE_LITERAL" >"$GENERATOR_BRANCH_RETURN_COND_NEGATIVE_LITERAL_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=1' "$GENERATOR_BRANCH_RETURN_COND_NEGATIVE_LITERAL_LOG"; then
  printf 'PASS  %s rejected direct-return generated Knowledge<T> value through const-evaluable branch condition violating call-site proof context\n' "$GENERATOR_BRANCH_RETURN_COND_NEGATIVE_LITERAL"
else
  printf 'FAIL  %s did not reject the violating direct-return generated Knowledge<T> value through const-evaluable branch condition\n' "$GENERATOR_BRANCH_RETURN_COND_NEGATIVE_LITERAL" >&2
  cat "$GENERATOR_BRANCH_RETURN_COND_NEGATIVE_LITERAL_LOG" >&2
  exit 1
fi

GENERATOR_BRANCH_RETURN_COND_DYNAMIC_GAP_LOG="$TMP_DIR/knowledge-static-value-generator-branch-return-cond-dynamic-gap.log"
if bin/souc run "$PROBE" -- "$GENERATOR_BRANCH_RETURN_COND_DYNAMIC_GAP" >"$GENERATOR_BRANCH_RETURN_COND_DYNAMIC_GAP_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_BRANCH_RETURN_COND_DYNAMIC_GAP_LOG"; then
  printf 'PASS  %s classified dynamic direct-return generated Knowledge<T> branch condition as runtime-check gap\n' "$GENERATOR_BRANCH_RETURN_COND_DYNAMIC_GAP"
else
  printf 'FAIL  %s did not preserve the dynamic direct-return generated-branch-condition gap classification\n' "$GENERATOR_BRANCH_RETURN_COND_DYNAMIC_GAP" >&2
  cat "$GENERATOR_BRANCH_RETURN_COND_DYNAMIC_GAP_LOG" >&2
  exit 1
fi

GENERATOR_HELPER_BRANCH_COND_POSITIVE_LOG="$TMP_DIR/knowledge-static-value-generator-helper-branch-cond-positive.log"
if bin/souc run "$PROBE" -- "$GENERATOR_HELPER_BRANCH_COND_POSITIVE" >"$GENERATOR_HELPER_BRANCH_COND_POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_HELPER_BRANCH_COND_POSITIVE_LOG"; then
  printf 'PASS  %s accepted generated Knowledge<T> value through helper const-evaluable branch condition satisfying call-site proof context\n' "$GENERATOR_HELPER_BRANCH_COND_POSITIVE"
else
  printf 'FAIL  %s did not accept the satisfying generated Knowledge<T> value through helper const-evaluable branch condition\n' "$GENERATOR_HELPER_BRANCH_COND_POSITIVE" >&2
  cat "$GENERATOR_HELPER_BRANCH_COND_POSITIVE_LOG" >&2
  exit 1
fi

GENERATOR_HELPER_BRANCH_COND_NEGATIVE_LITERAL_LOG="$TMP_DIR/knowledge-static-value-generator-helper-branch-cond-reject-literal.log"
if bin/souc run "$PROBE" -- "$GENERATOR_HELPER_BRANCH_COND_NEGATIVE_LITERAL" >"$GENERATOR_HELPER_BRANCH_COND_NEGATIVE_LITERAL_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=1' "$GENERATOR_HELPER_BRANCH_COND_NEGATIVE_LITERAL_LOG"; then
  printf 'PASS  %s rejected generated Knowledge<T> value through helper const-evaluable branch condition violating call-site proof context\n' "$GENERATOR_HELPER_BRANCH_COND_NEGATIVE_LITERAL"
else
  printf 'FAIL  %s did not reject the violating generated Knowledge<T> value through helper const-evaluable branch condition\n' "$GENERATOR_HELPER_BRANCH_COND_NEGATIVE_LITERAL" >&2
  cat "$GENERATOR_HELPER_BRANCH_COND_NEGATIVE_LITERAL_LOG" >&2
  exit 1
fi

GENERATOR_HELPER_BRANCH_COND_DYNAMIC_GAP_LOG="$TMP_DIR/knowledge-static-value-generator-helper-branch-cond-dynamic-gap.log"
if bin/souc run "$PROBE" -- "$GENERATOR_HELPER_BRANCH_COND_DYNAMIC_GAP" >"$GENERATOR_HELPER_BRANCH_COND_DYNAMIC_GAP_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_HELPER_BRANCH_COND_DYNAMIC_GAP_LOG"; then
  printf 'PASS  %s classified dynamic helper branch condition as runtime-check gap\n' "$GENERATOR_HELPER_BRANCH_COND_DYNAMIC_GAP"
else
  printf 'FAIL  %s did not preserve the dynamic helper branch-condition gap classification\n' "$GENERATOR_HELPER_BRANCH_COND_DYNAMIC_GAP" >&2
  cat "$GENERATOR_HELPER_BRANCH_COND_DYNAMIC_GAP_LOG" >&2
  exit 1
fi

GENERATOR_HELPER_BRANCH_FLOW_POSITIVE_LOG="$TMP_DIR/knowledge-static-value-generator-helper-branch-flow-positive.log"
if bin/souc run "$PROBE" -- "$GENERATOR_HELPER_BRANCH_FLOW_POSITIVE" >"$GENERATOR_HELPER_BRANCH_FLOW_POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_HELPER_BRANCH_FLOW_POSITIVE_LOG"; then
  printf 'PASS  %s accepted generated Knowledge<T> value through helper-local branch flow satisfying call-site proof context\n' "$GENERATOR_HELPER_BRANCH_FLOW_POSITIVE"
else
  printf 'FAIL  %s did not accept the satisfying generated Knowledge<T> value through helper-local branch flow\n' "$GENERATOR_HELPER_BRANCH_FLOW_POSITIVE" >&2
  cat "$GENERATOR_HELPER_BRANCH_FLOW_POSITIVE_LOG" >&2
  exit 1
fi

GENERATOR_HELPER_BRANCH_FLOW_NEGATIVE_LITERAL_LOG="$TMP_DIR/knowledge-static-value-generator-helper-branch-flow-reject-literal.log"
if bin/souc run "$PROBE" -- "$GENERATOR_HELPER_BRANCH_FLOW_NEGATIVE_LITERAL" >"$GENERATOR_HELPER_BRANCH_FLOW_NEGATIVE_LITERAL_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=1' "$GENERATOR_HELPER_BRANCH_FLOW_NEGATIVE_LITERAL_LOG"; then
  printf 'PASS  %s rejected generated Knowledge<T> value through helper-local branch flow violating call-site proof context\n' "$GENERATOR_HELPER_BRANCH_FLOW_NEGATIVE_LITERAL"
else
  printf 'FAIL  %s did not reject the violating generated Knowledge<T> value through helper-local branch flow\n' "$GENERATOR_HELPER_BRANCH_FLOW_NEGATIVE_LITERAL" >&2
  cat "$GENERATOR_HELPER_BRANCH_FLOW_NEGATIVE_LITERAL_LOG" >&2
  exit 1
fi

GENERATOR_HELPER_BRANCH_FLOW_DYNAMIC_GAP_LOG="$TMP_DIR/knowledge-static-value-generator-helper-branch-flow-dynamic-gap.log"
if bin/souc run "$PROBE" -- "$GENERATOR_HELPER_BRANCH_FLOW_DYNAMIC_GAP" >"$GENERATOR_HELPER_BRANCH_FLOW_DYNAMIC_GAP_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_HELPER_BRANCH_FLOW_DYNAMIC_GAP_LOG"; then
  printf 'PASS  %s classified dynamic helper-local branch flow as runtime-check gap\n' "$GENERATOR_HELPER_BRANCH_FLOW_DYNAMIC_GAP"
else
  printf 'FAIL  %s did not preserve the dynamic helper-local branch-flow gap classification\n' "$GENERATOR_HELPER_BRANCH_FLOW_DYNAMIC_GAP" >&2
  cat "$GENERATOR_HELPER_BRANCH_FLOW_DYNAMIC_GAP_LOG" >&2
  exit 1
fi

GENERATOR_NESTED_HELPER_POSITIVE_LOG="$TMP_DIR/knowledge-static-value-generator-nested-helper-positive.log"
if bin/souc run "$PROBE" -- "$GENERATOR_NESTED_HELPER_POSITIVE" >"$GENERATOR_NESTED_HELPER_POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_NESTED_HELPER_POSITIVE_LOG"; then
  printf 'PASS  %s accepted generated Knowledge<T> value through nested helper call satisfying call-site proof context\n' "$GENERATOR_NESTED_HELPER_POSITIVE"
else
  printf 'FAIL  %s did not accept the satisfying generated Knowledge<T> value through nested helper call\n' "$GENERATOR_NESTED_HELPER_POSITIVE" >&2
  cat "$GENERATOR_NESTED_HELPER_POSITIVE_LOG" >&2
  exit 1
fi

GENERATOR_NESTED_HELPER_NEGATIVE_LITERAL_LOG="$TMP_DIR/knowledge-static-value-generator-nested-helper-reject-literal.log"
if bin/souc run "$PROBE" -- "$GENERATOR_NESTED_HELPER_NEGATIVE_LITERAL" >"$GENERATOR_NESTED_HELPER_NEGATIVE_LITERAL_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=1' "$GENERATOR_NESTED_HELPER_NEGATIVE_LITERAL_LOG"; then
  printf 'PASS  %s rejected generated Knowledge<T> value through nested helper call violating call-site proof context\n' "$GENERATOR_NESTED_HELPER_NEGATIVE_LITERAL"
else
  printf 'FAIL  %s did not reject the violating generated Knowledge<T> value through nested helper call\n' "$GENERATOR_NESTED_HELPER_NEGATIVE_LITERAL" >&2
  cat "$GENERATOR_NESTED_HELPER_NEGATIVE_LITERAL_LOG" >&2
  exit 1
fi

GENERATOR_NESTED_HELPER_DYNAMIC_GAP_LOG="$TMP_DIR/knowledge-static-value-generator-nested-helper-dynamic-gap.log"
if bin/souc run "$PROBE" -- "$GENERATOR_NESTED_HELPER_DYNAMIC_GAP" >"$GENERATOR_NESTED_HELPER_DYNAMIC_GAP_LOG" 2>&1 &&
   grep -q 'knowledge_static_value_verdict=0' "$GENERATOR_NESTED_HELPER_DYNAMIC_GAP_LOG"; then
  printf 'PASS  %s classified dynamic nested helper call as runtime-check gap\n' "$GENERATOR_NESTED_HELPER_DYNAMIC_GAP"
else
  printf 'FAIL  %s did not preserve the dynamic nested-helper gap classification\n' "$GENERATOR_NESTED_HELPER_DYNAMIC_GAP" >&2
  cat "$GENERATOR_NESTED_HELPER_DYNAMIC_GAP_LOG" >&2
  exit 1
fi

cat <<'MSG'
Knowledge static value initializer gate passed.
This proves literal `Knowledge { value: Struct { ... } }` initializers are
checked against numeric proof-context thresholds for direct returns and local
let/var bindings, call arguments, and assignments when all relevant values are
compile-time literals, including unit-suffixed thresholds and simple generated
value constructors whose constrained fields are direct literals, direct
parameter references supplied by literal call arguments, or narrow
const-evaluable numeric expressions over those values, including caller
parameter threading across narrow nested helper calls, simple local
bindings plus simple `=` and compound assignments in the generated constructor
body and narrow helper calls with const-evaluable arguments and helper-local
bindings/assignments, plus literal-boolean branch selection in generated
constructors, literal-boolean branch local flow before return, and simple
const-evaluable numeric branch conditions over literals and literal-backed
parameters for branch local flow, direct-return branch selection, and narrow
helper branch selection plus helper-local branch flow, as well as narrow nested
helper calls with caller parameter threading. Nonliteral values and dynamic
branch conditions remain classified runtime-check gaps and are intentionally
not claimed as statically proven.
MSG
