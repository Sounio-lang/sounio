#!/usr/bin/env bash
# One list decides which keywords may also be used as identifiers:
# `tk_is_contextual_ident` in self-hosted/lexer/token.sio. Every predicate that
# admits identifiers must ASK it rather than carry a copy.
#
# This is not hypothetical tidiness. `at_expr_start` did not ask, and
# `at_expr_start` is what parse_return_expr consults to decide whether a value
# follows `return`. So `return policy.count` parsed as a BARE return: the
# expression was discarded in silence, the return typed as (), and the failure
# surfaced somewhere else entirely as "expected i64, found ()" with its span on
# the word `return`. A parser that REFUSES an identifier is found in one run.
# One that drops the expression is found by bisecting a reduced file.
#
# A research branch went the other way and inlined the list into each caller.
# Three copies later they had drifted, and the drift is what this gate exists to
# make impossible: with one list there is nothing to diverge.
#
# parse_module_item is the fourth caller, and it was found the expensive way:
# by building Madaros from this tree and discovering it could not parse three
# files of its OWN source — self-hosted/resolve/{scope,mod,resolve}.sio. Its
# path loop tested `== TokenKind::Ident` and nothing else, so `module
# resolve::scope` stopped at `scope`, handed the keyword to the item dispatcher
# and died. `module x::study` was worse: `study` HAS a dispatcher branch, so it
# began parsing a study item from inside a module header. The shipped binary
# predates the loop, which is why no gate and no test had ever seen it.
set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 9

HELPER="self-hosted/lexer/token.sio"
CALLERS=(
    "self-hosted/parser/exprs.sio:parse_prefix"
    "self-hosted/parser/patterns.sio:parse_pattern_atom"
    "self-hosted/parser/parser.sio:at_expr_start"
    "self-hosted/parser/items.sio:parse_module_item"
)

fail() { echo "PARSER_CONTEXTUAL_IDENT_ROUTING_GATE_FAIL: $*" >&2; exit 1; }

grep -q "fn tk_is_contextual_ident" "$HELPER" \
    || fail "the single list is gone from $HELPER — this gate is no longer reading what it claims to"

kinds=$(sed -n '/fn tk_is_contextual_ident/,/^}/p' "$HELPER" | grep -oE "TokenKind::\w+" | sort -u)
n=$(printf '%s\n' "$kinds" | grep -c .)
echo "  tk_is_contextual_ident admits $n kinds"

bad=0
for entry in "${CALLERS[@]}"; do
    file="${entry%%:*}"; fn="${entry##*:}"
    if ! grep -q "fn $fn" "$file"; then
        echo "  SKIP  $fn not found in $file"
        continue
    fi
    if grep -q "tk_is_contextual_ident" "$file"; then
        printf "  OK    %-20s routes through the helper (%s)\n" "$fn" "$file"
    else
        printf "  FAIL  %-20s does NOT ask the helper (%s)\n" "$fn" "$file"
        bad=1
    fi
done

if [[ $bad -ne 0 ]]; then
    echo
    echo "  A predicate that admits identifiers without asking tk_is_contextual_ident"
    echo "  will disagree with the ones that do. When the disagreeing predicate is"
    echo "  at_expr_start, the symptom is not a parse error — it is an expression"
    echo "  silently dropped from a \`return\` or \`break\`, and a type mismatch"
    echo "  reported against () somewhere downstream."
    exit 1
fi

echo "PARSER_CONTEXTUAL_IDENT_ROUTING_GATE_OK"
