#!/usr/bin/env bash
# stdlib_evolution_gate.sh -- Validate stdlib evolution: docs + modules + tests
# Sprint 230: Making aspirational stubs REAL
set -eo pipefail

SOUC="${SOUC:-./bin/souc}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$(pwd)/stdlib}"

PASS=0
FAIL=0
TOTAL=0

check_file() {
    local label="$1"
    local file="$2"
    TOTAL=$((TOTAL+1))
    if $SOUC check "$file" >/dev/null 2>&1; then
        PASS=$((PASS+1))
    else
        FAIL=$((FAIL+1))
        echo "  FAIL (check): $label -- $file"
    fi
}

run_file() {
    local label="$1"
    local file="$2"
    TOTAL=$((TOTAL+1))
    if ! $SOUC check "$file" >/dev/null 2>&1; then
        FAIL=$((FAIL+1))
        echo "  FAIL (check): $label -- $file"
        return
    fi
    if timeout 30 $SOUC run "$file" >/dev/null 2>&1; then
        PASS=$((PASS+1))
    else
        FAIL=$((FAIL+1))
        echo "  FAIL (run): $label -- $file"
    fi
}

echo "=== STDLIB EVOLUTION GATE ==="
echo ""

# --- Phase 1: Core modules type-check ---
echo "Phase 1: Core module type-check"
check_file "csv/parser"          stdlib/csv/parser.sio
check_file "json/parser"         stdlib/json/parser.sio
check_file "signal/fft"          stdlib/signal/fft.sio
check_file "functional/calculus" stdlib/functional/calculus.sio
check_file "special/gamma"       stdlib/special/gamma.sio
check_file "special/beta"        stdlib/special/beta.sio
check_file "special/erf"         stdlib/special/erf.sio
check_file "special/igamma"      stdlib/special/igamma.sio
check_file "encoding/hex"        stdlib/encoding/hex.sio
check_file "test/helpers"        stdlib/test/helpers.sio
check_file "collections/vec"     stdlib/collections/vec.sio
check_file "core/option"         stdlib/core/option.sio
check_file "core/result"         stdlib/core/result.sio
check_file "prob/normal"         stdlib/prob/normal.sio
check_file "math/approx"         stdlib/math/approx.sio
check_file "iter/iterator"       stdlib/iter/iterator.sio
check_file "string/string"       stdlib/string/string.sio

# --- Phase 2: Run key tests ---
echo ""
echo "Phase 2: Run key tests"
run_file "test_gamma"            tests/stdlib/special/test_gamma.sio
run_file "test_beta"             tests/stdlib/special/test_beta.sio
run_file "test_erf"              tests/stdlib/special/test_erf.sio
run_file "test_igamma_bessel"    tests/stdlib/special/test_igamma_bessel.sio
run_file "test_fft"              tests/stdlib/signal/test_fft.sio
run_file "test_csv"              tests/stdlib/csv/test_csv.sio
run_file "test_json"             tests/stdlib/json/test_json.sio
run_file "test_calculus"         tests/stdlib/functional/test_calculus.sio
run_file "test_collections"      tests/stdlib/collections/test_collections_hm64_vec.sio
run_file "test_rng"              tests/stdlib/random/test_rng.sio
run_file "test_encoding"         tests/stdlib/encoding/test_encoding_e2e.sio
run_file "test_knowledge"        tests/stdlib/epistemic/test_knowledge_jit.sio
run_file "test_option"           tests/stdlib/core/test_option_e2e.sio
run_file "test_result"           tests/stdlib/core/test_result_e2e.sio

# --- Phase 3: LLM docs exist ---
echo ""
echo "Phase 3: LLM training docs"
for doc in docs/LLM_PROGRAMMING_GUIDE.md SOUNIO_QUICK_START.md SOUNIO_GOTCHAS.md SOUNIO_STYLE_GUIDE.md; do
    TOTAL=$((TOTAL+1))
    if [ -f "$doc" ] && [ -s "$doc" ]; then
        PASS=$((PASS+1))
    else
        FAIL=$((FAIL+1))
        echo "  FAIL (missing): $doc"
    fi
done

# --- Summary ---
echo ""
echo "=== RESULTS: TOTAL=$TOTAL PASS=$PASS FAIL=$FAIL ==="
if [ "$FAIL" -gt 0 ]; then
    echo "GATE: FAIL"
    exit 1
else
    echo "GATE: PASS"
    exit 0
fi
