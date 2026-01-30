#!/bin/bash
# .claude/hooks/post-write.sh
# Automatic actions after file writes

set -e

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_input.path // empty')

# Auto-format Rust files
if [[ "$FILE_PATH" == *.rs ]]; then
    if command -v rustfmt &> /dev/null; then
        rustfmt "$FILE_PATH" 2>/dev/null || true
    fi
fi

# Check Sounio syntax
if [[ "$FILE_PATH" == *.sio ]]; then
    SOUC="$PROJECT_DIR/target/release/souc"
    if [[ -x "$SOUC" ]]; then
        if ! "$SOUC" check "$FILE_PATH" 2>/dev/null; then
            echo "WARNING: Syntax error in $FILE_PATH" >&2
        fi
    fi
fi

# Run clippy on compiler changes
if [[ "$FILE_PATH" == */compiler/src/*.rs ]]; then
    cd "$PROJECT_DIR/compiler"
    cargo clippy --quiet --message-format=short -- -D warnings 2>&1 | head -20 || true
fi

# Log
echo "$(date '+%Y-%m-%d %H:%M:%S') | POST-WRITE | $FILE_PATH" >> "$PROJECT_DIR/.claude/logs/writes.log" 2>/dev/null || true

exit 0
