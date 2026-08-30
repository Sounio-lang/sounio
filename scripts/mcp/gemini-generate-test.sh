#!/usr/bin/env bash
# gemini-generate-test.sh — Generate Sounio test cases using Gemini CLI
# Usage: ./scripts/mcp/gemini-generate-test.sh <function_name> <source_file> [test_type]
# Example: ./scripts/mcp/gemini-generate-test.sh "propagate_bmi" "stdlib/epistemic/gum.sio" unit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

if [[ $# -lt 2 ]]; then
    echo -e "${RED}Error: Insufficient arguments${NC}"
    echo "Usage: $0 <function_name> <source_file.sio> [test_type: unit|integration|compile-fail|fuzz]"
    exit 1
fi

FUNC_NAME="$1"
SOURCE_FILE="$2"
TEST_TYPE="${3:-unit}"

if [[ ! -f "$SOURCE_FILE" ]]; then
    echo -e "${RED}Error: Source file not found: $SOURCE_FILE${NC}"
    exit 1
fi

# Extract context from source file
FILE_CONTEXT=$(head -100 "$SOURCE_FILE" 2>/dev/null || cat "$SOURCE_FILE")

# Build test generation prompt
case "$TEST_TYPE" in
    unit)
        PROMPT="Generate a Sounio unit test for the function '${FUNC_NAME}'.

Source file context (first 100 lines):
\`\`\`sio
${FILE_CONTEXT}
\`\`\`

Requirements:
1. Test should be in tests/run-pass/ format
2. Include //@ run-pass annotation
3. Test edge cases, normal cases, and error conditions
4. Use appropriate effect declarations (with IO, Mut, Panic, Div as needed)
5. Follow Sounio syntax: no semicolons, use 'var' not 'let mut', '&!' not '&mut'
6. Include println() for observable output

Generate a complete, runnable test file."
        ;;
    integration)
        PROMPT="Generate a Sounio integration test for the function '${FUNC_NAME}'.

Source file context:
\`\`\`sio
${FILE_CONTEXT}
\`\`\`

Requirements:
1. Test integration with stdlib modules
2. Use realistic data if applicable (epistemic types, units, etc.)
3. Include proper effect propagation
4. Test cross-module interactions if relevant
5. Format: tests/run-pass/ style with annotations

Generate a complete integration test."
        ;;
    compile-fail)
        PROMPT="Generate a Sounio compile-fail test for the function '${FUNC_NAME}'.

Source file context:
\`\`\`sio
${FILE_CONTEXT}
\`\`\`

Requirements:
1. Test should fail to compile with appropriate error
2. Include //@ compile-fail annotation
3. Include //@ error-pattern: <expected error>
4. Test type safety, effect safety, or borrow checking
5. Demonstrate correct error detection

Generate a test that SHOULD fail compilation with clear error."
        ;;
    fuzz)
        PROMPT="Generate a Sounio fuzz/property-based test for the function '${FUNC_NAME}'.

Source file context:
\`\`\`sio
${FILE_CONTEXT}
\`\`\`

Requirements:
1. Generate random/property-based test cases
2. Test invariants that should always hold
3. Include property checks (e.g., round-trip, commutativity)
4. Use loops to test many random inputs
5. Include assertions

Generate a comprehensive fuzz-style test."
        ;;
esac

echo -e "${BLUE}Generating ${TEST_TYPE} test for: ${FUNC_NAME}${NC}"
echo -e "${BLUE}Source: ${SOURCE_FILE}${NC}"

# Generate test
# Use snap gemini with OAuth (Ultra subscription)
if [[ -n "${GEMINI_API_KEY:-}" ]]; then
    TEST_OUTPUT=$(/snap/bin/gemini --prompt "$PROMPT" --model gemini-1.5-pro)
else
    TEST_OUTPUT=$(GOOGLE_GENAI_USE_GCA=true /snap/bin/gemini --prompt "$PROMPT" --model gemini-1.5-pro)
fi

# Extract code block if present
if echo "$TEST_OUTPUT" | grep -q '```sio'; then
    TEST_CODE=$(echo "$TEST_OUTPUT" | awk '/```sio/{flag=1;next}/```/{flag=0}flag')
else
    TEST_CODE="$TEST_OUTPUT"
fi

# Output file naming
OUTPUT_FILE="tests/run-pass/${FUNC_NAME}_generated_test.sio"
if [[ "$TEST_TYPE" == "compile-fail" ]]; then
    OUTPUT_FILE="tests/compile-fail/${FUNC_NAME}_generated_test.sio"
fi

# Display generated test
echo ""
echo -e "${GREEN}Generated test:${NC}"
echo "---"
echo "$TEST_CODE"
echo "---"

# Ask user before saving
echo ""
read -p "Save to ${OUTPUT_FILE}? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p "$(dirname "$OUTPUT_FILE")"
    echo "$TEST_CODE" > "$OUTPUT_FILE"
    echo -e "${GREEN}Saved to: ${OUTPUT_FILE}${NC}"
    
    # Validate with souc if available
    if command -v souc &> /dev/null || [[ -n "${SOUC:-}" ]]; then
        SOUC_BIN="${SOUC:-souc}"
        echo -e "${BLUE}Validating with souc...${NC}"
        if $SOUC_BIN check "$OUTPUT_FILE" 2>&1; then
            echo -e "${GREEN}Validation passed!${NC}"
        else
            echo -e "${YELLOW}Validation found issues — review needed${NC}"
        fi
    fi
else
    echo -e "${YELLOW}Test not saved${NC}"
fi
