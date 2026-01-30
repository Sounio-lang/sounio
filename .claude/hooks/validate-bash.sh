#!/bin/bash
# .claude/hooks/validate-bash.sh
# Validates bash commands before execution

set -e

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Dangerous patterns
DANGEROUS_PATTERNS=(
    "rm -rf /"
    "rm -rf ~"
    "rm -rf \$HOME"
    "> /dev/sda"
    "mkfs"
    "dd if="
    ":(){:|:&};:"
)

for pattern in "${DANGEROUS_PATTERNS[@]}"; do
    if [[ "$COMMAND" == *"$pattern"* ]]; then
        echo "BLOCKED: Dangerous command detected: $pattern" >&2
        exit 2
    fi
done

# Block deletions in compiler/src without confirmation
if [[ "$COMMAND" == *"rm"* ]] && [[ "$COMMAND" == *"compiler/src"* ]]; then
    echo "BLOCKED: Deletion in compiler/src requires manual confirmation" >&2
    exit 2
fi

# Log command
echo "$(date '+%Y-%m-%d %H:%M:%S') | BASH | $COMMAND" >> "$PROJECT_DIR/.claude/logs/commands.log" 2>/dev/null || true

exit 0
