#!/usr/bin/env bash
# knowledge_runtime_guard_native_lowering_gate.sh
#
# Validates the Sounio-native Knowledge<T where {...}> guard expander
# (self-hosted/compiler/knowledge_runtime_guard_expander.sio).
# Each test: expand via Sounio driver → compile to native → run → check exit code.
# positive cases must exit 0, reject cases must exit 1.
# Does NOT use scripts/ontology/expand_knowledge_runtime_guards.sh.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

EXPANDER="self-hosted/compiler/knowledge_runtime_guard_expander.sio"
SOUC="bin/souc"

pass=0
fail=0

run_case() {
    local label="$1"
    local src="$2"
    local expect_exit="$3"

    local expanded="$TMP_DIR/${label}.expanded.sio"
    local binary="$TMP_DIR/${label}.native"
    local expand_log="$TMP_DIR/${label}.expand.log"
    local compile_log="$TMP_DIR/${label}.compile.log"

    if ! "$SOUC" run "$EXPANDER" -- "$src" >"$expanded" 2>"$expand_log"; then
        printf 'FAIL  [%s]  expander error\n' "$label"
        cat "$expand_log" >&2
        fail=$((fail + 1))
        return
    fi

    if ! "$SOUC" compile "$expanded" -o "$binary" >"$compile_log" 2>&1; then
        printf 'FAIL  [%s]  compile error\n' "$label"
        cat "$compile_log" >&2
        fail=$((fail + 1))
        return
    fi

    actual_exit=0
    "$binary" || actual_exit=$?
    if [ "$actual_exit" -eq "$expect_exit" ]; then
        printf 'PASS  [%s]  exit=%d\n' "$label" "$actual_exit"
        pass=$((pass + 1))
    else
        printf 'FAIL  [%s]  expected exit=%d got exit=%d\n' "$label" "$expect_exit" "$actual_exit"
        fail=$((fail + 1))
    fi
}

# Return-site guards
run_case "positive"          tests/frontend/knowledge_runtime_guard_positive.sio          0
run_case "reject"            tests/frontend/knowledge_runtime_guard_reject.sio             1

# Multi-constraint (two conjunctive guards)
run_case "multi_positive"    tests/frontend/knowledge_runtime_guard_multi_positive.sio     0
run_case "multi_reject"      tests/frontend/knowledge_runtime_guard_multi_reject.sio       1

# Upper-bound guard
run_case "upper_positive"    tests/frontend/knowledge_runtime_guard_upper_positive.sio     0
run_case "upper_reject"      tests/frontend/knowledge_runtime_guard_upper_reject.sio       1

# Equality guard
run_case "eq_positive"       tests/frontend/knowledge_runtime_guard_eq_positive.sio        0
run_case "eq_reject"         tests/frontend/knowledge_runtime_guard_eq_reject.sio          1

# Assignment-site guards
run_case "assign_positive"   tests/frontend/knowledge_runtime_guard_assign_positive.sio    0
run_case "assign_reject"     tests/frontend/knowledge_runtime_guard_assign_reject.sio      1

# Call-boundary guards
run_case "call_positive"     tests/frontend/knowledge_runtime_guard_call_positive.sio      0
run_case "call_reject"       tests/frontend/knowledge_runtime_guard_call_reject.sio        1

# Directive-annotated source
run_case "directive_positive" tests/frontend/knowledge_runtime_guard_directive_positive.sio 0
run_case "directive_reject"   tests/frontend/knowledge_runtime_guard_directive_reject.sio   1

echo ""
printf 'knowledge_runtime_guard_native_lowering_gate: %d PASS  %d FAIL\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
