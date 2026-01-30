#!/bin/bash
# .claude/hooks/on-stop.sh
# Executed when Claude finishes a response

set -e

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_DIR"

# Check for modified files
MODIFIED=$(git diff --name-only --cached 2>/dev/null || git diff --name-only 2>/dev/null || echo "")

# Run quick tests if compiler changed
if echo "$MODIFIED" | grep -q "^compiler/"; then
    echo "Changes detected in compiler/. Running quick tests..."
    cd compiler && cargo test --lib --quiet 2>&1 | tail -5 || true
    cd ..
fi

# Validate .sio files if stdlib changed
if echo "$MODIFIED" | grep -q "^stdlib/"; then
    echo "Changes detected in stdlib/."
    for file in $(echo "$MODIFIED" | grep "\.sio$"); do
        if [[ -f "$file" ]]; then
            ./target/release/souc check "$file" 2>&1 || echo "Error in $file"
        fi
    done
fi

echo "✓ Claude Code task complete"

exit 0
