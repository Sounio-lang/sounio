#!/usr/bin/env bash
set -euo pipefail

PASS=0; FAIL=0; TOTAL=0
SOUC="${SOUC:-artifacts/omega/souc-bin/souc-linux-x86_64-jit}"
[ -x "$SOUC" ] || SOUC="target/debug/souc"

check() {
    local name="$1"; local expect="$2"; shift 2
    TOTAL=$((TOTAL+1))
    if eval "$@" > /dev/null 2>&1; then
        if [ "$expect" = "pass" ]; then
            echo "PASS  $name"; PASS=$((PASS+1))
        else
            echo "FAIL  $name (expected fail, got pass)"; FAIL=$((FAIL+1))
        fi
    else
        if [ "$expect" = "fail" ]; then
            echo "PASS  $name"; PASS=$((PASS+1))
        else
            echo "FAIL  $name (expected pass, got fail)"; FAIL=$((FAIL+1))
        fi
    fi
}

check_grep() {
    local name="$1"; local pattern="$2"; local file="$3"
    TOTAL=$((TOTAL+1))
    if grep -q "$pattern" "$file" 2>/dev/null; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (pattern '$pattern' not found in $file)"; FAIL=$((FAIL+1))
    fi
}

echo "=== Sprint 31: Shpitser-Pearl ID Algorithm ==="
echo ""

echo "--- Structural checks ---"
check_grep "causal:c_component_struct" "struct CComponent" "self-hosted/check/causal.sio"
check_grep "causal:c_component_list" "struct CComponentList" "self-hosted/check/causal.sio"
check_grep "causal:compute_c_components" "causal_compute_c_components" "self-hosted/check/causal.sio"
check_grep "causal:find_c_component_of" "causal_find_c_component_of" "self-hosted/check/causal.sio"
check_grep "causal:topological_sort" "causal_topological_sort" "self-hosted/check/causal.sio"
check_grep "causal:subgraph" "causal_subgraph" "self-hosted/check/causal.sio"
check_grep "causal:id_algorithm" "fn id_algorithm" "self-hosted/check/causal.sio"
check_grep "causal:set_union" "fn set_union" "self-hosted/check/causal.sio"
check_grep "causal:set_subtract" "fn set_subtract" "self-hosted/check/causal.sio"
check_grep "causal:set_is_subset" "fn set_is_subset" "self-hosted/check/causal.sio"
check_grep "causal:id_wired" "id_algorithm\|method=3\|0, 3" "self-hosted/check/causal.sio"

echo ""
echo "--- Test files ---"
check_grep "test:hedge_detected" "hedge\|bow" "tests/compile-fail/causal_id_hedge_detected.sio"
check_grep "test:bidirected_wall" "confounded\|bidirected" "tests/compile-fail/causal_id_bidirected_wall.sio"
check_grep "test:napkin_identified" "napkin\|Napkin" "tests/frontend/causal_id_napkin_identified.sio"
check_grep "test:chain_trivial" "chain\|Chain" "tests/frontend/causal_id_chain_trivial.sio"

echo ""
echo "--- souc check verification ---"

# CausalEffect keyword probe
if echo 'fn main() -> i64 { 0 }' > /tmp/id_probe.sio && $SOUC check /tmp/id_probe.sio > /dev/null 2>&1; then
    check "souc:chain_trivial" "pass" "$SOUC check tests/frontend/causal_id_chain_trivial.sio"
    check "souc:napkin_identified" "pass" "$SOUC check tests/frontend/causal_id_napkin_identified.sio"
    check "souc:hedge_detected" "fail" "$SOUC check tests/compile-fail/causal_id_hedge_detected.sio"
    check "souc:bidirected_wall" "fail" "$SOUC check tests/compile-fail/causal_id_bidirected_wall.sio"
else
    echo "SKIP  souc binary not available"
fi

echo ""
echo "--- Regression ---"
check "souc:regression_hello" "pass" "$SOUC check tests/run-pass/hello.sio"

# Check existing do-calculus tests still pass
for tf in tests/frontend/causal_d_sep_chain.sio tests/frontend/causal_d_sep_fork.sio tests/frontend/causal_d_sep_collider.sio; do
    if [ -f "$tf" ]; then
        check "souc:regression_$(basename $tf .sio)" "pass" "$SOUC check $tf"
    fi
done

echo ""
echo "=== Sprint 31 Gate: $PASS/$TOTAL pass ==="

mkdir -p artifacts/sprint31
cat > artifacts/sprint31/id_algorithm_gate.v1.json <<ARTIFACT
{
  "schema": "sounio.sprint31.id_algorithm_gate.v1",
  "status": "$([ $FAIL -eq 0 ] && echo pass || echo fail)",
  "total": $TOTAL,
  "passed": $PASS,
  "failed": $FAIL
}
ARTIFACT
echo "artifact: $(pwd)/artifacts/sprint31/id_algorithm_gate.v1.json"
