#!/bin/bash
# .claude/hooks/pre-write.sh
# Validations before writing files

set -e

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_input.path // empty')

# Reject .d extension — Sounio uses .sio
if [[ "$FILE_PATH" == *.d ]]; then
    CORRECTED="${FILE_PATH%.d}.sio"
    echo "BLOCKED: .d extension detected. Use .sio for Sounio files." >&2
    echo "Suggestion: $CORRECTED" >&2
    exit 2
fi

# Warn on protected files (but don't block)
PROTECTED_FILES=(
    "Cargo.lock"
    ".git/config"
    "compiler/src/main.rs"
)

for protected in "${PROTECTED_FILES[@]}"; do
    if [[ "$FILE_PATH" == *"$protected" ]]; then
        echo "WARNING: Modifying protected file: $protected" >&2
    fi
done

# Log
echo "$(date '+%Y-%m-%d %H:%M:%S') | PRE-WRITE | $FILE_PATH" >> "$PROJECT_DIR/.claude/logs/writes.log" 2>/dev/null || true

exit 0
