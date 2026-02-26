#!/usr/bin/env bash
# Sounio REPL — interactive evaluation via file-based compilation
#
# Pragmatic approach: accumulates definitions in a temp .sio file,
# wraps each expression in a main() function, compiles and runs.
#
# Usage: bash tools/repl.sh
#
# Commands:
#   :quit, :q     — exit
#   :clear, :c    — reset accumulated definitions
#   :show, :s     — show accumulated source
#   :type EXPR    — show type of expression (uses check mode)
#   fn ..., struct ..., let ..., var ... — accumulated as definitions
#   <expression>  — wrapped in main() and executed

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve souc binary
if [[ -n "${SOUC_BIN:-}" && -x "${SOUC_BIN:-}" ]]; then
    : # already set
elif [[ -f "$ROOT_DIR/scripts/lib/resolve_souc.sh" ]]; then
    # shellcheck source=../scripts/lib/resolve_souc.sh
    source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
    sounio_require_souc
else
    echo "error: cannot find souc binary. Set SOUC_BIN or run from repo root." >&2
    exit 1
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

DEFS_FILE="$TMPDIR/repl_defs.sio"
EVAL_FILE="$TMPDIR/repl_eval.sio"
: > "$DEFS_FILE"

COUNTER=0

echo "Sounio REPL (file-based, using $SOUC_BIN)"
echo "Type :help for commands, :quit to exit."
echo ""

while true; do
    printf "sounio> "
    if ! IFS= read -r line; then
        echo ""
        break
    fi

    # Trim whitespace
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"

    # Skip empty lines
    if [[ -z "$line" ]]; then
        continue
    fi

    # Commands
    case "$line" in
        :quit|:q)
            echo "Goodbye."
            break
            ;;
        :clear|:c)
            : > "$DEFS_FILE"
            COUNTER=0
            echo "(definitions cleared)"
            continue
            ;;
        :show|:s)
            echo "--- accumulated definitions ---"
            cat "$DEFS_FILE"
            echo "--- end ---"
            continue
            ;;
        :help|:h)
            echo "Commands:"
            echo "  :quit, :q     Exit the REPL"
            echo "  :clear, :c    Clear accumulated definitions"
            echo "  :show, :s     Show accumulated source"
            echo "  :type EXPR    Show type of expression"
            echo ""
            echo "Input:"
            echo "  fn/struct/let/var/enum/impl  Accumulated as definitions"
            echo "  <expression>                 Evaluated and printed"
            continue
            ;;
        :type\ *)
            expr="${line#:type }"
            {
                cat "$DEFS_FILE"
                echo ""
                echo "fn __repl_type_check__() -> _ with IO, Mut, Panic, Div, Alloc {"
                echo "    $expr"
                echo "}"
            } > "$EVAL_FILE"
            "$SOUC_BIN" check "$EVAL_FILE" 2>&1 || true
            continue
            ;;
    esac

    # Determine if this is a definition or an expression
    if [[ "$line" =~ ^(fn |struct |enum |impl |type |use |linear ) ]]; then
        # Accumulate as definition
        echo "$line" >> "$DEFS_FILE"
        echo "(defined)"
    else
        # Wrap in main() and execute
        COUNTER=$((COUNTER + 1))
        {
            cat "$DEFS_FILE"
            echo ""
            echo "fn main() -> i32 with IO, Mut, Panic, Div, Alloc {"
            # If it looks like a statement (let/var/return), use as-is
            if [[ "$line" =~ ^(let |var |return |if |while |match |for ) ]]; then
                echo "    $line"
                echo "    return 0"
            else
                # Treat as expression — print result
                echo "    let __repl_result__ = $line"
                echo "    print_int(__repl_result__ as i64)"
                echo "    print(\"\\n\")"
                echo "    return 0"
            fi
            echo "}"
        } > "$EVAL_FILE"

        output=$("$SOUC_BIN" run "$EVAL_FILE" 2>&1) || true
        if [[ -n "$output" ]]; then
            echo "$output"
        fi
    fi
done
