#!/bin/bash
# Stdlib Verification Script - Check overnight expansion results

set -e

echo "============================================"
echo "Sounio Stdlib Expansion - Verification"
echo "============================================"
echo ""

cd "$(dirname "$0")/compiler"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS=0
FAIL=0
SKIP=0

check_module() {
    local module="$1"
    local path="../$module"

    if [ ! -f "$path" ]; then
        echo -e "${YELLOW}⊘ SKIP${NC} $module (not found)"
        ((SKIP++))
        return
    fi

    echo -n "Testing $module... "
    if cargo run --bin souc -- check "$path" 2>&1 | grep -q "All checks passed"; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASS++))
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((FAIL++))
    fi
}

echo "Phase 1: Collections & Search"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check_module "stdlib/collections/heap.sio"
check_module "stdlib/collections/bitset.sio"
check_module "stdlib/collections/btree.sio"
check_module "stdlib/collections/trie.sio"
check_module "stdlib/search/basic.sio"
check_module "stdlib/search/pattern.sio"
echo ""

echo "Phase 2: Algorithms"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check_module "stdlib/algo/sort.sio"
check_module "stdlib/algo/graph.sio"
echo ""

echo "Phase 3: Data Serialization"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check_module "stdlib/toml/mod.sio"
check_module "stdlib/yaml/mod.sio"
check_module "stdlib/msgpack/mod.sio"
echo ""

echo "Phase 5: Systems Primitives"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check_module "stdlib/sync/atomic.sio"
check_module "stdlib/sync/mutex.sio"
check_module "stdlib/sync/channel.sio"
check_module "stdlib/os/env.sio"
check_module "stdlib/os/process.sio"
echo ""

echo "Phase 6: Extensions"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check_module "stdlib/compress/zstd.sio"
echo ""

echo "Infrastructure"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check_module "stdlib/algo/mod.sio"
check_module "stdlib/sync/lib.sio"
check_module "stdlib/os/lib.sio"
check_module "stdlib/collections/lib.sio"
check_module "stdlib/search/lib.sio"
echo ""

TOTAL=$((PASS + FAIL + SKIP))
echo "============================================"
echo "Summary:"
echo "  Total modules: $TOTAL"
echo -e "  ${GREEN}Passed: $PASS${NC}"
echo -e "  ${RED}Failed: $FAIL${NC}"
echo -e "  ${YELLOW}Skipped: $SKIP${NC}"
echo ""

if [ $TOTAL -gt 0 ]; then
    SUCCESS_RATE=$((PASS * 100 / TOTAL))
    echo "Success rate: $SUCCESS_RATE%"
    echo ""

    if [ $SUCCESS_RATE -ge 95 ]; then
        echo -e "${GREEN}Excellent! Nearly perfect success.${NC}"
    elif [ $SUCCESS_RATE -ge 79 ]; then
        echo -e "${GREEN}Good! Most modules working.${NC}"
    elif [ $SUCCESS_RATE -ge 63 ]; then
        echo -e "${YELLOW}Acceptable. Some manual fixes needed.${NC}"
    else
        echo -e "${RED}Needs attention. Many modules need fixes.${NC}"
    fi
fi
echo "============================================"

# Test a few examples if they exist
echo ""
echo "Testing Example Programs:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

test_example() {
    local example="$1"
    local path="../$example"

    if [ ! -f "$path" ]; then
        return
    fi

    echo -n "Testing $example... "
    if cargo run --bin souc -- check "$path" 2>&1 | grep -q "All checks passed"; then
        echo -e "${GREEN}✓ PASS${NC}"
    else
        echo -e "${RED}✗ FAIL${NC}"
    fi
}

test_example "examples/collections/heap_demo.sio"
test_example "examples/collections/bitset_demo.sio"
test_example "examples/algo/sorting_demo.sio"
test_example "examples/algo/graph_demo.sio"
test_example "examples/serialization/toml_demo.sio"
test_example "examples/sync/atomic_demo.sio"

echo ""
echo "For detailed errors, run:"
echo "  cargo run --bin souc -- check ../stdlib/path/to/module.sio"
echo ""
echo "See /tmp/compilation_report.md for agent monitoring results"
echo "See WHEN_YOU_WAKE_UP.md for next steps"
