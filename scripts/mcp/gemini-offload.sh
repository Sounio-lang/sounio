#!/usr/bin/env bash
# gemini-offload.sh — Fan-out to Gemini CLI for Sounio-specific tasks
# Usage: ./scripts/mcp/gemini-offload.sh <prompt-file> [mode: review|scaffold|explain|optimize|math]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

GEMINI_CLI="/snap/bin/gemini"

if [[ $# -lt 1 ]]; then
    echo -e "${RED}Error: No prompt file specified${NC}"
    echo "Usage: $0 <prompt-file> [mode: review|scaffold|explain|optimize|math]"
    exit 1
fi

PROMPT_FILE="$1"
MODE="${2:-review}"

if [[ ! -f "$PROMPT_FILE" ]]; then
    echo -e "${RED}Error: Prompt file not found: $PROMPT_FILE${NC}"
    exit 1
fi

# Check authentication method
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
    echo -e "${YELLOW}Note: GEMINI_API_KEY not set, using OAuth (Ultra subscription)${NC}"
fi

# Read base prompt
BASE_PROMPT=$(cat "$PROMPT_FILE")

# Wrap with Sounio context
SONUIO_CONTEXT="You are analyzing code for Sounio, a self-hosted systems programming language for epistemic computing.

SOUNIO SYNTAX RULES (CRITICAL):
- NO semicolons at end of lines
- Use 'var' for mutable, 'let' for immutable (NOT 'let mut')
- Use '&!' for exclusive references (NOT '&mut')
- Functions MUST declare effects: 'fn foo() with IO, Mut, Panic { ... }'
- Effects: IO, Mut, Alloc, Panic, Div, Async, GPU, Prob, Observe
- NO closure literals — use named function references
- NO macros — use 'assert()' not 'assert!()'
- Struct fields end with comma, no semicolon after last field
- NO unary minus — use '0 - x'
- Bit shifts require u8: 'x >> 4u8'

Current project: Self-hosted compiler with 1.55M LOC, bootstrapped from C.

TASK:"

FULL_PROMPT="${SONUIO_CONTEXT}

${BASE_PROMPT}"

# Select model based on mode
case "$MODE" in
    review)
        MODEL="gemini-1.5-pro"
        ECHO_MODE="${CYAN}Code Review${NC}"
        ;;
    scaffold)
        MODEL="gemini-2.5-flash"
        ECHO_MODE="${CYAN}Scaffolding${NC}"
        ;;
    explain)
        MODEL="gemini-1.5-pro"
        ECHO_MODE="${CYAN}Deep Explanation${NC}"
        ;;
    optimize)
        MODEL="gemini-1.5-pro"
        ECHO_MODE="${CYAN}Optimization Analysis${NC}"
        ;;
    math)
        MODEL="gemini-1.5-pro"
        ECHO_MODE="${CYAN}Mathematical Reasoning${NC}"
        ;;
esac

echo -e "${BLUE}=== Gemini Offload ===${NC}"
echo -e "Mode: ${ECHO_MODE}"
echo -e "Model: ${MODEL}"
echo -e "Prompt: $(wc -c < "$PROMPT_FILE") bytes"
echo ""

# Execute with API key or OAuth
if [[ -n "${GEMINI_API_KEY:-}" ]]; then
    "$GEMINI_CLI" -p "$FULL_PROMPT" -m "$MODEL" 2>&1
else
    # Use OAuth mode (Ultra subscription via snap credentials)
    GOOGLE_GENAI_USE_GCA=true "$GEMINI_CLI" -p "$FULL_PROMPT" -m "$MODEL" 2>&1 || {
        echo -e "${RED}Failed. Set GEMINI_API_KEY or run 'gemini' interactively to refresh OAuth.${NC}"
        exit 1
    }
fi

echo ""
echo -e "${GREEN}✓ Gemini offload complete${NC}"
