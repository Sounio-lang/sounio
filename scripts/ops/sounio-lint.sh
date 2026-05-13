#!/usr/bin/env bash
# Scan .sio files for Rust-like syntax drift.
# Usage: sounio-lint.sh [file.sio ...]
# If no args, scans *.sio in current directory.
# Exit 0 = clean, 1 = findings.
set -euo pipefail

FINDINGS=0

lint_file() {
    local file="$1"
    [[ -f "$file" ]] || return 0

    local line_no=0
    while IFS= read -r line || [[ -n "$line" ]]; do
        line_no=$((line_no + 1))

        # Skip comments
        [[ "$line" =~ ^[[:space:]]*//.* ]] && continue

        # Rule: rust-macro — println!() or assert!()
        if echo "$line" | grep -qP '\b(println|assert|eprintln|dbg|vec|format)!\s*\('; then
            echo "$file:$line_no: error: rust-macro: Rust macro syntax. Remove '!' → use function call"
            FINDINGS=$((FINDINGS + 1))
        fi

        # Rule: rust-borrow — &mut
        if echo "$line" | grep -qP '\b&mut\b'; then
            echo "$file:$line_no: error: rust-borrow: '&mut' is Rust. Use '&!' for exclusive references"
            FINDINGS=$((FINDINGS + 1))
        fi

        # Rule: rust-let-mut — let mut
        if echo "$line" | grep -qP '\blet\s+mut\b'; then
            echo "$file:$line_no: error: rust-let-mut: 'let mut' is Rust. Use 'var' for mutable bindings"
            FINDINGS=$((FINDINGS + 1))
        fi

        # Rule: rust-attr — #[...]
        if echo "$line" | grep -qP '^\s*#\['; then
            echo "$file:$line_no: error: rust-attr: Rust attribute syntax. Use '//@ annotation' for test metadata"
            FINDINGS=$((FINDINGS + 1))
        fi

        # Rule: semicolons — trailing ; on statements (not in strings, not in extern blocks, not in comments)
        # Skip lines that are: extern "C" { }, struct fields, enum variants, or inside string literals
        if echo "$line" | grep -qP ';\s*$'; then
            # Allow: extern declarations, use statements that the parser may accept
            if ! echo "$line" | grep -qP '(extern\s|fn\s+\w+\(.*\)\s*;|^\s*use\s)'; then
                echo "$file:$line_no: warn: semicolon: Trailing semicolon. Sounio does not use semicolons"
                FINDINGS=$((FINDINGS + 1))
            fi
        fi

        # Rule: rust-closure — |x| lambda syntax
        if echo "$line" | grep -qP '\|\w+\|\s*(->|\{|\w)'; then
            echo "$file:$line_no: error: rust-closure: Closure literal syntax. Use named function + fn ref"
            FINDINGS=$((FINDINGS + 1))
        fi

    done < "$file"
}

# Collect files
FILES=()
if [[ $# -gt 0 ]]; then
    FILES=("$@")
else
    shopt -s nullglob
    FILES=(*.sio)
    shopt -u nullglob
fi

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No .sio files found." >&2
    exit 0
fi

for f in "${FILES[@]}"; do
    lint_file "$f"
done

if [[ $FINDINGS -eq 0 ]]; then
    echo "Clean: no Rust-isms found."
    exit 0
else
    echo ""
    echo "$FINDINGS finding(s) detected."
    exit 1
fi
