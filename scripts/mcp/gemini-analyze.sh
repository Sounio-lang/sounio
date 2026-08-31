#!/usr/bin/env bash
# gemini-analyze.sh — Analyze Sounio code using Gemini CLI
# Usage: ./scripts/mcp/gemini-analyze.sh <file.sio> [context]
# Authentication: Set GEMINI_API_KEY environment variable, or use interactive login

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Gemini CLI path
GEMINI_CLI="/snap/bin/gemini"

# Check arguments
if [[ $# -lt 1 ]]; then
    echo -e "${RED}Error: No file specified${NC}"
    echo "Usage: $0 <file.sio> [context: architecture|types|effects|optimization]"
    exit 1
fi

TARGET_FILE="$1"
CONTEXT="${2:-general}"

# Validate file exists
if [[ ! -f "$TARGET_FILE" ]]; then
    echo -e "${RED}Error: File not found: $TARGET_FILE${NC}"
    exit 1
fi

# Check if gemini CLI exists
if [[ ! -f "$GEMINI_CLI" ]]; then
    echo -e "${RED}Error: Gemini CLI not found${NC}"
    echo "Install with: npm install -g @google/gemini-cli"
    exit 1
fi

# Check authentication
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
    echo -e "${YELLOW}Warning: GEMINI_API_KEY not set${NC}"
    echo "Get key from: https://aistudio.google.com/app/apikey"
    echo "Or run interactive mode for OAuth (Ultra subscription)"
    echo ""
    echo "Trying anyway... (may prompt for auth)"
    echo ""
fi

# Build context-specific prompt
case "$CONTEXT" in
    architecture)
        PROMPT="Analyze this Sounio code for architectural patterns. Focus on:
- Module structure and organization
- Separation of concerns
- Potential refactoring opportunities
- Bootstrap chain implications
- Self-hosting compatibility

File: ${TARGET_FILE}"
        MODEL="gemini-1.5-pro"
        ;;
    types)
        PROMPT="Analyze this Sounio code for type system correctness. Focus on:
- Type safety issues
- Missing effect declarations (with IO, Mut, Panic, Div)
- Reference type usage (& vs &!)
- Potential type checker failures
- Epistemic type propagation if applicable

File: ${TARGET_FILE}"
        MODEL="gemini-1.5-pro"
        ;;
    effects)
        PROMPT="Analyze this Sounio code for effect system compliance. Focus on:
- Missing 'with' clauses
- Effect propagation through call chains
- Algebraic effects best practices
- Unsafe operations without Panic
- IO operations not declared

File: ${TARGET_FILE}"
        MODEL="gemini-1.5-pro"
        ;;
    optimization)
        PROMPT="Analyze this Sounio code for optimization opportunities. Focus on:
- Performance bottlenecks
- Unnecessary allocations
- Loop optimizations
- Native codegen efficiency
- E-graph rewrite opportunities

File: ${TARGET_FILE}"
        MODEL="gemini-1.5-pro"
        ;;
    general|*)
        PROMPT="Analyze this Sounio code for general quality. Provide:
- Syntax correctness check
- Style guide compliance
- Potential bugs or issues
- Suggestions for improvement
- Bootstrap chain compatibility

File: ${TARGET_FILE}"
        MODEL="gemini-2.5-flash"
        ;;
esac

echo -e "${BLUE}Analyzing: ${TARGET_FILE}${NC}"
echo -e "${BLUE}Context: ${CONTEXT}${NC}"
echo -e "${BLUE}Model: ${MODEL}${NC}"
echo "---"

# Create full prompt with code
FULL_PROMPT="${PROMPT}

Code content:
\`\`\`sio
$(cat "$TARGET_FILE")
\`\`\`"

# Run Gemini analysis
if [[ -n "${GEMINI_API_KEY:-}" ]]; then
    # Use API Key mode
    "$GEMINI_CLI" -p "$FULL_PROMPT" -m "$MODEL" 2>&1
else
    # Use OAuth mode (Ultra subscription via snap credentials)
    GOOGLE_GENAI_USE_GCA=true "$GEMINI_CLI" -p "$FULL_PROMPT" -m "$MODEL" 2>&1 || {
        echo ""
        echo -e "${RED}Authentication failed. Please:${NC}"
        echo "1. Set GEMINI_API_KEY environment variable, OR"
        echo "2. Run 'gemini' interactively to refresh OAuth credentials"
        exit 1
    }
fi

echo ""
echo -e "${GREEN}Analysis complete${NC}"
