#!/usr/bin/env bash
# check_doc_snippets.sh -- Extract ```sio blocks from docs and check them
#
# Extracts Sounio code blocks from markdown docs, writes each to a temp file,
# and runs `souc check` on them. Reports pass/fail counts.
#
# Usage:
#   bash scripts/ci/check_doc_snippets.sh [--verbose] [optional dir to scan]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="${SOUC:-$REPO_ROOT/bin/souc}"
VERBOSE=""
TARGET_DIR=""

for arg in "$@"; do
    if [ "$arg" == "--verbose" ]; then
        VERBOSE="--verbose"
    else
        TARGET_DIR="$arg"
    fi
done

if [ -z "$TARGET_DIR" ]; then
    # Restrict default scanning to rosetta and website tutorials to avoid the explosion of failures in spec docs
    TARGET_DIRS=("$REPO_ROOT/docs/training/rosetta" "$REPO_ROOT/website/src/content/docs")
else
    TARGET_DIRS=("$TARGET_DIR")
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

pass=0
fail=0
skip=0
total=0

extract_snippets() {
    local md_file="$1"
    local base="$(basename "$md_file" .md)"
    local inside=false
    local snippet_idx=0
    local buf=""

    while IFS= read -r line; do
        if [[ "$line" =~ ^\`\`\`sio ]]; then
            inside=true
            buf=""
            continue
        fi
        if [[ "$line" =~ ^\`\`\`sounio ]]; then
            inside=true
            buf=""
            continue
        fi
        if $inside && [[ "$line" == '```' ]]; then
            if [[ -n "$buf" ]]; then
                local out="$TMPDIR/${base}_snippet_${snippet_idx}.sio"
                echo "$buf" > "$out"
                snippet_idx=$((snippet_idx + 1))
            fi
            inside=false
            buf=""
            continue
        fi
        if $inside; then
            buf="${buf}${line}"$'\n'
        fi
    done < "$md_file"
}

for dir in "${TARGET_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        find "$dir" -type f \( -name '*.md' -o -name '*.mdx' \) -not -path '*/archived/*' -not -path '*/internal/*' | sort | while read -r md; do
            extract_snippets "$md"
        done
    fi
done

# Check if there are any .sio files
if ls "$TMPDIR"/*.sio >/dev/null 2>&1; then
    for f in "$TMPDIR"/*.sio; do
        total=$((total + 1))

        # Skip incomplete snippets (no fn main, just expressions)
        if ! grep -q 'fn ' "$f" 2>/dev/null; then
            skip=$((skip + 1))
            if [[ "$VERBOSE" == "--verbose" ]]; then
                echo "SKIP (no fn): $(basename "$f")"
            fi
            continue
        fi

        if "$SOUC" check "$f" > /dev/null 2>&1; then
            pass=$((pass + 1))
            if [[ "$VERBOSE" == "--verbose" ]]; then
                echo "PASS: $(basename "$f")"
            fi
        else
            fail=$((fail + 1))
            echo "FAIL: $(basename "$f")"
            if [[ "$VERBOSE" == "--verbose" ]]; then
                "$SOUC" check "$f" 2>&1 | head -5
            fi
        fi
    done
fi

echo ""
echo "Doc snippet check: ${pass} pass, ${fail} fail, ${skip} skip, ${total} total"

if [ "$fail" -gt 0 ]; then
    exit 1
fi
