#!/usr/bin/env bash
# tools/fmt/sounio-fmt.sh — Sounio Phase 1 token-level formatter
#
# Phase 1 scope (idempotent, token-level only):
#   - Expand tabs to 4 spaces
#   - Strip trailing whitespace from every line
#   - Collapse 3+ consecutive blank lines to at most 2 blank lines
#     (i.e. at most one blank line between non-blank lines)
#   - Normalise `,` spacing: exactly one space after `,`, no space before `,`
#   - Normalise `->` spacing: exactly one space on each side
#
# Phase 1 explicitly defers (noted in docs/compiler/KNOWN_LIMITATIONS.md):
#   - Operator spacing for `+`, `-`, `*`, `/`, `:`, `=`
#     (ambiguous: `-` vs `->`, `*` deref vs multiply, `:` in URLs and types)
#   - AST-based indentation normalisation (requires full parse)
#   - Blank-line policy inside structs / impl blocks
#
# Usage:
#   tools/fmt/sounio-fmt.sh <source.sio>           # formatted output to stdout
#   tools/fmt/sounio-fmt.sh --check <source.sio>   # exit 1 if would change
#   tools/fmt/sounio-fmt.sh --in-place <source.sio># format in place
#
# All rules are applied through a single-pass awk program that correctly
# skips string literals and line comments, making each rule idempotent.
#
# Idempotency guarantee: format(format(x)) == format(x) for any input.

set -euo pipefail

MODE="stdout"
SOURCE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)
            MODE="check"
            shift
            ;;
        --in-place|-i)
            MODE="inplace"
            shift
            ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        -*)
            echo "error: unknown flag: $1" >&2
            exit 2
            ;;
        *)
            if [[ -z "$SOURCE" ]]; then
                SOURCE="$1"
            else
                echo "error: unexpected argument: $1" >&2
                exit 2
            fi
            shift
            ;;
    esac
done

if [[ -z "$SOURCE" ]]; then
    echo "error: missing source file" >&2
    echo "usage: sounio-fmt.sh [--check|--in-place] <source.sio>" >&2
    exit 2
fi

if [[ ! -f "$SOURCE" ]]; then
    echo "error: file not found: $SOURCE" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Core formatting engine — single awk pass
# ---------------------------------------------------------------------------
# The awk program processes one line at a time:
#   1. Expands leading tabs to 4-space groups (per-tab).
#   2. Strips trailing whitespace.
#   3. Applies comma spacing and `->` spacing ONLY outside string literals
#      and line comments.
#   4. Accumulates blank lines and flushes at most one.
# ---------------------------------------------------------------------------

FORMAT_AWK='
BEGIN {
    blank_run = 0
}

{
    line = $0

    # -----------------------------------------------------------------------
    # Step 1: expand leading tabs to 4 spaces each
    # -----------------------------------------------------------------------
    expanded = ""
    i = 1
    while (i <= length(line)) {
        ch = substr(line, i, 1)
        if (ch == "\t") {
            expanded = expanded "    "
        } else {
            # Once we hit a non-tab (or if we loop past leading region),
            # append the rest verbatim and stop tab expansion.
            expanded = expanded substr(line, i)
            break
        }
        i++
    }
    # If the line was all tabs (or empty after tab expansion):
    if (i > length(line)) {
        line = expanded
    } else {
        line = expanded
    }

    # -----------------------------------------------------------------------
    # Step 2: strip trailing whitespace
    # -----------------------------------------------------------------------
    sub(/[[:space:]]+$/, "", line)

    # -----------------------------------------------------------------------
    # Step 3: blank-line accumulator
    # -----------------------------------------------------------------------
    if (line == "") {
        blank_run++
        next
    }

    # Flush at most one blank line before this non-blank line.
    if (blank_run > 0) {
        print ""
    }
    blank_run = 0

    # -----------------------------------------------------------------------
    # Step 4: token-aware comma and `->` normalisation
    #
    # Walk the line character by character, tracking whether we are inside
    # a string literal (double-quoted) or a line comment (`//`). Outside
    # those regions we apply spacing rules. Inside, we copy verbatim.
    # -----------------------------------------------------------------------
    result = ""
    in_string = 0
    in_comment = 0
    n = length(line)
    j = 1

    while (j <= n) {
        c = substr(line, j, 1)

        if (in_comment) {
            # Rest of line is comment — copy verbatim to end
            result = result substr(line, j)
            break
        }

        if (in_string) {
            result = result c
            if (c == "\\") {
                # Escape sequence: copy next char too, do not change state
                if (j + 1 <= n) {
                    j++
                    result = result substr(line, j, 1)
                }
            } else if (c == "\"") {
                in_string = 0
            }
            j++
            continue
        }

        # Check for comment start
        if (c == "/" && j + 1 <= n && substr(line, j + 1, 1) == "/") {
            in_comment = 1
            result = result c
            j++
            continue
        }

        # Check for string start
        if (c == "\"") {
            in_string = 1
            result = result c
            j++
            continue
        }

        # --- Comma spacing: strip space before comma, ensure space after ---
        if (c == ",") {
            # Strip any trailing spaces we may have emitted before the comma
            sub(/ +$/, "", result)
            result = result ","
            # Skip any spaces that follow the comma in the source
            j++
            while (j <= n && substr(line, j, 1) == " ") j++
            # If not at end of line, and next char is not a closing paren/bracket,
            # add exactly one space
            if (j <= n && substr(line, j, 1) != ")") {
                result = result " "
            }
            continue
        }

        # --- Arrow `->` spacing: ensure exactly " -> " ---
        if (c == "-" && j + 1 <= n && substr(line, j + 1, 1) == ">") {
            # Strip trailing spaces before `->`
            sub(/ +$/, "", result)
            result = result " -> "
            j += 2   # skip `->`
            # Skip any spaces that follow `->` in the source
            while (j <= n && substr(line, j, 1) == " ") j++
            continue
        }

        result = result c
        j++
    }

    print result
}

END {
    # Do not emit trailing blank lines at EOF
}
'

do_format() {
    local src="$1"
    awk "$FORMAT_AWK" "$src"
}

case "$MODE" in
    stdout)
        do_format "$SOURCE"
        ;;
    check)
        tmp="$(mktemp)"
        trap 'rm -f "$tmp"' EXIT
        do_format "$SOURCE" > "$tmp"
        if diff -q "$tmp" "$SOURCE" > /dev/null 2>&1; then
            exit 0
        else
            echo "would reformat: $SOURCE" >&2
            exit 1
        fi
        ;;
    inplace)
        tmp="$(mktemp)"
        trap 'rm -f "$tmp"' EXIT
        do_format "$SOURCE" > "$tmp"
        if ! diff -q "$tmp" "$SOURCE" > /dev/null 2>&1; then
            cp "$tmp" "$SOURCE"
        fi
        ;;
esac
