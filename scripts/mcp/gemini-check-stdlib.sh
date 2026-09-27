#!/usr/bin/env bash
# gemini-check-stdlib.sh — Batch analyze stdlib modules using Gemini Ultra
# Usage: ./scripts/mcp/gemini-check-stdlib.sh [module-pattern] [max-files]
# Example: ./scripts/mcp/gemini-check-stdlib.sh "epistemic/*.sio" 5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

PATTERN="${1:-*.sio}"
MAX_FILES="${2:-10}"

cd "$PROJECT_ROOT"

# Find files
FILES=$(find stdlib -name "$PATTERN" -type f 2>/dev/null | head -n "$MAX_FILES")

if [[ -z "$FILES" ]]; then
    echo -e "${RED}No files matching pattern: ${PATTERN}${NC}"
    exit 1
fi

FILE_COUNT=$(echo "$FILES" | wc -l)
echo -e "${BLUE}Found ${FILE_COUNT} files to analyze${NC}"
echo -e "${BLUE}Max files: ${MAX_FILES}${NC}"
echo ""

# Check Ultra quota
QUOTA_REMAINING=2000
GEMINI_BIN="/snap/bin/gemini"

if GOOGLE_GENAI_USE_GCA=true $GEMINI_BIN /auth 2>/dev/null | grep -q "2,000"; then
    ULTRA_INFO=$(GOOGLE_GENAI_USE_GCA=true $GEMINI_BIN /auth 2>/dev/null)
    echo -e "${GREEN}✓ Ultra subscription: 2,000 requests/day${NC}"
else
    echo -e "${YELLOW}⚠ Using default quota (authenticate with Ultra for 2,000/day)${NC}"
    QUOTA_REMAINING=1000
fi

if [[ "$FILE_COUNT" -gt "$QUOTA_REMAINING" ]]; then
    echo -e "${RED}Error: ${FILE_COUNT} files exceeds daily quota (${QUOTA_REMAINING})${NC}"
    echo "Reduce MAX_FILES or use --sample mode"
    exit 1
fi

# Analysis types
ANALYSIS_TYPES=("syntax" "effects" "types" "documentation")

# Process each file
COUNTER=0
for FILE in $FILES; do
    COUNTER=$((COUNTER + 1))
    ANALYSIS="${ANALYSIS_TYPES[$((COUNTER % 4))]}"
    
    echo -e "${CYAN}[${COUNTER}/${FILE_COUNT}]${NC} Analyzing: ${FILE} (${ANALYSIS})"
    
    # Skip if file is too large (>100KB)
    FILE_SIZE=$(stat -c%s "$FILE" 2>/dev/null || echo 0)
    if [[ "$FILE_SIZE" -gt 100000 ]]; then
        echo -e "${YELLOW}  Skipping (too large: ${FILE_SIZE} bytes)${NC}"
        continue
    fi
    
    # Create focused prompt
    case "$ANALYSIS" in
        syntax)
            PROMPT="Quick syntax check for Sounio code. List any syntax errors:
- Semicolons (should be none)
- Missing effect declarations
- Wrong mutability keywords
- Array syntax issues

File: ${FILE}"
            ;;
        effects)
            PROMPT="Check effect declarations in this Sounio code:
- Are all IO operations declared with 'with IO'?
- Are mutations declared with 'with Mut'?
- Is Panic declared for array access/as casts?
- Is Div declared for division/modulo?

File: ${FILE}"
            ;;
        types)
            PROMPT="Check type system usage:
- Proper reference types (& vs &!)
- Type annotations where needed
- Generic usage correctness
- Knowledge[T] epistemic type usage

File: ${FILE}"
            ;;
        documentation)
            PROMPT="Review code documentation:
- Are public functions documented?
- Are complex algorithms explained?
- Are epistemic assumptions noted?
- Suggest doc improvements

File: ${FILE}"
            ;;
    esac
    
    # Run Gemini analysis (brief mode)
    TEMP_PROMPT=$(mktemp)
    echo "$PROMPT" > "$TEMP_PROMPT"
    echo "" >> "$TEMP_PROMPT"
    echo "Code:" >> "$TEMP_PROMPT"
    head -50 "$FILE" >> "$TEMP_PROMPT"
    
    if [[ -n "${GEMINI_API_KEY:-}" ]]; then
        RESULT=$($GEMINI_BIN --prompt "$(cat "$TEMP_PROMPT")" --model gemini-2.5-flash 2>&1 || echo "ERROR")
    else
        RESULT=$(GOOGLE_GENAI_USE_GCA=true $GEMINI_BIN --prompt "$(cat "$TEMP_PROMPT")" --model gemini-2.5-flash 2>&1 || echo "ERROR")
    fi
    
    if echo "$RESULT" | grep -qi "error\|issue\|problem\|missing\|should be"; then
        echo -e "${YELLOW}  ⚠ Issues detected${NC}"
        echo "$RESULT" | head -5 | sed 's/^/    /'
    else
        echo -e "${GREEN}  ✓ Clean${NC}"
    fi
    
    rm -f "$TEMP_PROMPT"
    
    # Rate limit: max 60 RPM for Ultra
    sleep 1
done

echo ""
echo -e "${GREEN}Batch analysis complete${NC}"
echo -e "${BLUE}Quota used: ${COUNTER}/2000${NC}"
