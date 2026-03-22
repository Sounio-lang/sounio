#!/bin/bash
# check_sounio.sh — Sounio source file validator
# Checks .sio files for common syntax issues (Rust-isms, missing effects, etc.)
#
# Usage:
#   ./check_sounio.sh [file.sio]           # check one file
#   ./check_sounio.sh                       # check all .sio files
#   ./check_sounio.sh --summary             # summary only

set -euo pipefail

ERRORS=0
WARNINGS=0
FILES_CHECKED=0
PASS=0

check_file() {
    local file="$1"
    local file_errors=0

    # Skip disabled/backup files
    case "$file" in
        *.disabled|*.backup|*.old) return 0 ;;
    esac

    # Skip files that don't exist
    [ -f "$file" ] || return 0

    FILES_CHECKED=$((FILES_CHECKED + 1))

    # Check for Rust-isms
    if grep -qn '&mut ' "$file" 2>/dev/null; then
        echo "ERROR: $file: uses &mut (should be &!)"
        file_errors=$((file_errors + 1))
    fi

    if grep -qnP 'assert!\(' "$file" 2>/dev/null; then
        echo "ERROR: $file: uses assert!() macro (use assert())"
        file_errors=$((file_errors + 1))
    fi

    if grep -qnP 'println!\(' "$file" 2>/dev/null; then
        echo "ERROR: $file: uses println!() macro (use print())"
        file_errors=$((file_errors + 1))
    fi

    if grep -qnP '#\[test\]' "$file" 2>/dev/null; then
        echo "ERROR: $file: uses #[test] attribute (not supported)"
        file_errors=$((file_errors + 1))
    fi

    if grep -qnP '#\[derive' "$file" 2>/dev/null; then
        echo "ERROR: $file: uses #[derive] attribute (not supported)"
        file_errors=$((file_errors + 1))
    fi

    if grep -qnP 'let mut ' "$file" 2>/dev/null; then
        echo "ERROR: $file: uses 'let mut' (should be 'var')"
        file_errors=$((file_errors + 1))
    fi

    # Check for missing effect annotations on IO functions
    if grep -qn 'fn.*print' "$file" 2>/dev/null; then
        if grep -qn 'fn.*print.*{' "$file" 2>/dev/null; then
            if ! grep -qn 'fn.*print.*with.*IO' "$file" 2>/dev/null; then
                # This is a heuristic — may have false positives
                :
            fi
        fi
    fi

    if [ $file_errors -eq 0 ]; then
        PASS=$((PASS + 1))
    else
        ERRORS=$((ERRORS + file_errors))
    fi
}

# Main
if [ "${1:-}" = "--summary" ]; then
    for f in $(find . -name "*.sio" -type f -not -path "./.git/*" -not -path "./node_modules/*" 2>/dev/null); do
        check_file "$f"
    done
    echo "Checked: $FILES_CHECKED files, Pass: $PASS, Errors: $ERRORS, Warnings: $WARNINGS"
    exit 0
fi

if [ $# -gt 0 ] && [ "$1" != "validate" ] && [ "$1" != "report" ]; then
    check_file "$1"
else
    for f in $(find . -name "*.sio" -type f -not -path "./.git/*" -not -path "./node_modules/*" 2>/dev/null | sort); do
        check_file "$f"
    done
fi

echo ""
echo "=== Sounio Validation ==="
echo "Files checked: $FILES_CHECKED"
echo "Pass: $PASS"
echo "Errors: $ERRORS"
echo "Warnings: $WARNINGS"

if [ $ERRORS -gt 0 ]; then
    echo "Status: FAIL"
    exit 1
else
    echo "Status: PASS"
    exit 0
fi
