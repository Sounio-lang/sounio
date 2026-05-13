#!/usr/bin/env bash
# Claude Code hook: auto-validate .sio files after Write/Edit.
# Reads JSON from stdin with { tool_name, tool_input: { file_path } }.
set -euo pipefail

# Parse stdin JSON
INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_name',''))" 2>/dev/null || echo "")
FILE_PATH=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('file_path',''))" 2>/dev/null || echo "")

# Only act on Write/Edit of .sio files
case "$TOOL_NAME" in
    Write|Edit|MultiEdit) ;;
    *) exit 0 ;;
esac

[[ "$FILE_PATH" == *.sio ]] || exit 0
[[ -f "$FILE_PATH" ]] || exit 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Lint for Rust-isms (always available, no compiler needed)
LINT="$SCRIPT_DIR/sounio-lint.sh"
if [[ -x "$LINT" ]]; then
    LINT_OUT=$("$LINT" "$FILE_PATH" 2>&1) || true
    if [[ -n "$LINT_OUT" ]] && ! echo "$LINT_OUT" | grep -q "^Clean:"; then
        echo "=== Sounio Lint ==="
        echo "$LINT_OUT"
        echo ""
    fi
fi

# 2. Type-check with souc (if available)
RESOLVE="$SCRIPT_DIR/resolve-souc.sh"
if [[ -x "$RESOLVE" ]]; then
    SOUC_PATH=$("$RESOLVE" 2>/dev/null) || true
    if [[ -n "$SOUC_PATH" ]] && [[ -x "$SOUC_PATH" ]]; then
        CHECK_OUT=$("$SOUC_PATH" check "$FILE_PATH" 2>&1) || true
        if [[ -n "$CHECK_OUT" ]]; then
            echo "=== souc check ==="
            echo "$CHECK_OUT" | head -20
        fi
    fi
fi

exit 0
