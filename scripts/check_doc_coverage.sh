#!/bin/bash
# Check documentation coverage and enforce for new/modified code
#
# This script validates that:
# 1. Public items in modified files have documentation
# 2. Documentation coverage meets minimum threshold
# 3. Cross-references are not broken

set -e

# Configuration
MIN_COVERAGE=${MIN_COVERAGE:-60}  # Minimum acceptable coverage percentage
STRICT=${STRICT:-false}            # Whether to fail on undocumented items

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "📚 Checking documentation coverage..."
echo "  Minimum coverage: ${MIN_COVERAGE}%"
echo "  Strict mode: ${STRICT}"
echo ""

# Run souniodoc check
cd compiler
if cargo run --bin souniodoc -- check --min-coverage "${MIN_COVERAGE}" > /tmp/doc_coverage.txt 2>&1; then
    echo -e "${GREEN}✓ Overall documentation coverage meets minimum threshold${NC}"
    cat /tmp/doc_coverage.txt
    COVERAGE_OK=true
else
    echo -e "${RED}✗ Overall documentation coverage below threshold${NC}"
    cat /tmp/doc_coverage.txt
    COVERAGE_OK=false
fi

# Get list of changed files in this PR/commit
if [ -n "$GITHUB_BASE_REF" ]; then
    # Running in GitHub Actions
    CHANGED_FILES=$(git diff --name-only "origin/$GITHUB_BASE_REF...HEAD" | grep '\.sio$' || true)
elif [ -n "$CI" ]; then
    # Running in CI but not GitHub Actions
    CHANGED_FILES=$(git diff --name-only HEAD~1 | grep '\.sio$' || true)
else
    # Running locally
    CHANGED_FILES=$(git diff --name-only HEAD | grep '\.sio$' || true)
    if [ -z "$CHANGED_FILES" ]; then
        # No staged changes, check unstaged
        CHANGED_FILES=$(git diff --name-only | grep '\.sio$' || true)
    fi
fi

if [ -z "$CHANGED_FILES" ]; then
    echo ""
    echo "ℹ️  No .sio files changed"
    if [ "$COVERAGE_OK" = "true" ]; then
        exit 0
    else
        if [ "$STRICT" = "true" ]; then
            exit 1
        else
            echo -e "${YELLOW}⚠  Coverage below threshold but passing in non-strict mode${NC}"
            exit 0
        fi
    fi
fi

echo ""
echo "📝 Checking documentation in changed files:"
echo "$CHANGED_FILES" | sed 's/^/  - /'
echo ""

# Check each changed file for undocumented public items
EXIT_CODE=0
UNDOCUMENTED_COUNT=0

for file in $CHANGED_FILES; do
    if [ ! -f "$file" ]; then
        continue
    fi

    echo "Checking $file..."

    # Look for public functions without doc comments
    # This is a simple heuristic - souniodoc does the real analysis

    # Extract public functions and check if they have doc comments above them
    while IFS= read -r line_num; do
        # Get the function line
        fn_line=$(sed -n "${line_num}p" "$file")

        # Check if there are doc comments (/// or //) in the 10 lines before
        start_line=$((line_num - 10))
        if [ $start_line -lt 1 ]; then
            start_line=1
        fi

        has_doc=false
        doc_lines=$(sed -n "${start_line},$((line_num - 1))p" "$file" | grep -E '^\s*(///|//)' || true)
        if [ -n "$doc_lines" ]; then
            has_doc=true
        fi

        if [ "$has_doc" = "false" ]; then
            fn_name=$(echo "$fn_line" | sed -n 's/.*fn \([a-zA-Z_][a-zA-Z0-9_]*\).*/\1/p')
            if [ -n "$fn_name" ]; then
                echo -e "  ${YELLOW}⚠  Undocumented public function: $fn_name at line $line_num${NC}"
                UNDOCUMENTED_COUNT=$((UNDOCUMENTED_COUNT + 1))
            fi
        fi
    done < <(grep -n '^pub fn ' "$file" | cut -d: -f1)

    # Check public structs
    while IFS= read -r line_num; do
        start_line=$((line_num - 10))
        if [ $start_line -lt 1 ]; then
            start_line=1
        fi

        has_doc=false
        doc_lines=$(sed -n "${start_line},$((line_num - 1))p" "$file" | grep -E '^\s*(///|//)' || true)
        if [ -n "$doc_lines" ]; then
            has_doc=true
        fi

        if [ "$has_doc" = "false" ]; then
            struct_line=$(sed -n "${line_num}p" "$file")
            struct_name=$(echo "$struct_line" | sed -n 's/.*struct \([a-zA-Z_][a-zA-Z0-9_]*\).*/\1/p')
            if [ -n "$struct_name" ]; then
                echo -e "  ${YELLOW}⚠  Undocumented public struct: $struct_name at line $line_num${NC}"
                UNDOCUMENTED_COUNT=$((UNDOCUMENTED_COUNT + 1))
            fi
        fi
    done < <(grep -n '^pub struct ' "$file" | cut -d: -f1)
done

echo ""
if [ $UNDOCUMENTED_COUNT -eq 0 ]; then
    echo -e "${GREEN}✓ All public items in changed files are documented${NC}"
else
    echo -e "${YELLOW}⚠  Found $UNDOCUMENTED_COUNT undocumented public items in changed files${NC}"

    if [ "$STRICT" = "true" ]; then
        echo -e "${RED}✗ Failing due to strict mode${NC}"
        EXIT_CODE=1
    else
        echo -e "${YELLOW}ℹ️  Passing in non-strict mode (set STRICT=true to fail)${NC}"
    fi
fi

echo ""
echo "Summary:"
echo "  - Overall coverage: $(grep -oP 'Coverage: \K[0-9]+' /tmp/doc_coverage.txt || echo 'N/A')%"
echo "  - Undocumented items in changed files: $UNDOCUMENTED_COUNT"
echo "  - Status: $([ $EXIT_CODE -eq 0 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${RED}FAIL${NC}")"

exit $EXIT_CODE
