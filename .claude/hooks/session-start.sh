#!/bin/bash
# .claude/hooks/session-start.sh
# Loads context at session start

set -e

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_DIR"

CONTEXT_FILE="$PROJECT_DIR/.claude/session-context.md"

cat > "$CONTEXT_FILE" << EOF
# Sounio Session Context

## Git Status
\`\`\`
$(git status --short 2>/dev/null || echo "N/A")
\`\`\`

## Branch
$(git branch --show-current 2>/dev/null || echo "N/A")

## Recent Commits
\`\`\`
$(git log --oneline -3 2>/dev/null || echo "N/A")
\`\`\`

## Recently Modified (last hour)
$(find . -name "*.rs" -o -name "*.sio" -mmin -60 2>/dev/null | head -10 || echo "None")

## Build Status
$(cd compiler && cargo check --quiet 2>&1 | tail -3 || echo "OK")

## Compiler Version
$(./target/release/souc --version 2>/dev/null || echo "Not built")
EOF

echo "Context loaded: .claude/session-context.md"

exit 0
