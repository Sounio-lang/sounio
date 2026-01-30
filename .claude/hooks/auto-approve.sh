#!/bin/bash
# .claude/hooks/auto-approve.sh
# Auto-approves safe Sounio-specific commands

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

SAFE_COMMANDS=(
    "cargo build"
    "cargo test"
    "cargo check"
    "cargo clippy"
    "cargo fmt"
    "cargo bench"
    "./target/release/souc"
    "./target/debug/souc"
    "git status"
    "git diff"
    "git log"
    "git branch"
    "git show"
    "ls"
    "cat"
    "head"
    "tail"
    "grep"
    "find"
    "wc"
    "file"
    "which"
)

for safe in "${SAFE_COMMANDS[@]}"; do
    if [[ "$COMMAND" == "$safe"* ]]; then
        echo '{"decision": "approve", "reason": "Safe Sounio command"}'
        exit 0
    fi
done

exit 0
