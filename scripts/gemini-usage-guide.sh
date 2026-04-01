#!/bin/bash
# gemini-usage-guide.sh - Show how to use Gemini CLI integration scripts

cat << 'EOF'
=== Gemini CLI Integration for Sounio ===

Once you have authentication set up (run ./scripts/gemini-api-key-setup.sh),
here's how to use the integration scripts:

--- 1. Code Analysis (gemini-analyze.sh) ---

Analyze a Sounio source file:
  ./scripts/gemini-analyze.sh <file> [type]

Types: architecture, types, effects, optimization, general (default)

Examples:
  ./scripts/gemini-analyze.sh stdlib/epistemic/knowledge.sio architecture
  ./scripts/gemini-analyze.sh stdlib/effects/io.sio effects
  ./scripts/gemini-analyze.sh mycode.sio types

--- 2. LLM Offload (gemini-offload.sh) ---

Drop-in replacement for llm-offload.sh with Gemini Ultra:
  ./scripts/gemini-offload.sh <prompt-file> [task]

Tasks: review, scaffold, explain, optimize, math

Example:
  echo "Review this code for idiomatic Sounio patterns" > /tmp/prompt.txt
  ./scripts/gemini-offload.sh /tmp/prompt.txt review

--- 3. Test Generation (gemini-generate-test.sh) ---

Generate Sounio test cases:
  ./scripts/gemini-generate-test.sh <function> <source-file> [type]

Types: unit, integration, compile-fail, fuzz

Example:
  ./scripts/gemini-generate-test.sh "knowledge_map" stdlib/epistemic/knowledge.sio unit

--- 4. Stdlib Batch Check (gemini-check-stdlib.sh) ---

Batch analyze stdlib modules:
  ./scripts/gemini-check-stdlib.sh [pattern] [max-files]

Examples:
  ./scripts/gemini-check-stdlib.sh "stdlib/epistemic/*.sio" 3
  ./scripts/gemini-check-stdlib.sh "stdlib/effects/*.sio" 5

--- 5. Verify Setup (gemini-verify-setup.sh) ---

Check everything is working:
  ./scripts/gemini-verify-setup.sh

--- Authentication Required ---

All scripts require GEMINI_API_KEY to be set:
  export GEMINI_API_KEY='your-key-from-aistudio.google.com'

Or for OAuth (Ultra subscription):
  export GOOGLE_GENAI_USE_GCA=true
  # Then run Gemini CLI interactively once to refresh token

--- Tips ---

- Scripts default to gemini-2.5-flash for speed, use Pro for complex analysis
- All scripts include Sounio context (var, &!, with IO, GUM, etc.)
- 1,000 requests/day free with API key
- 2,000 requests/day with Ultra subscription (OAuth)

EOF
