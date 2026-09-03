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
set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 9

HELPER="self-hosted/lexer/token.sio"
CALLERS=(
    "self-hosted/parser/exprs.sio:parse_prefix"
    "self-hosted/parser/patterns.sio:parse_pattern_atom"
    "self-hosted/parser/parser.sio:at_expr_start"
)

fail() { echo "PARSER_CONTEXTUAL_IDENT_ROUTING_GATE_FAIL: $*" >&2; exit 1; }

# Anchored on a word boundary, NOT a bare substring: `fn tk_is_contextual_identX`
# contains `fn tk_is_contextual_ident`, so a plain grep -q would happily confirm
# the presence of a helper that had been renamed out from under it.
grep -qE "fn tk_is_contextual_ident[^A-Za-z0-9_]" "$HELPER" \
    || fail "the single list is gone from $HELPER — this gate is no longer reading what it claims to"

kinds=$(sed -n '/fn tk_is_contextual_ident/,/^}/p' "$HELPER" | grep -oE "TokenKind::\w+" | sort -u)
n=$(printf '%s\n' "$kinds" | grep -c .)

# The extraction above is a sed range piped into a grep. If the helper is ever
# reshaped — renamed, split, reformatted so the range stops matching — `kinds`
# comes back EMPTY, `n` is 0, and without this the gate would announce
# "admits 0 kinds" and go on to pass, because the caller checks below only look
# for the helper's NAME. An empty extraction read as a real measurement is the
# exact failure gate_vacuity_gate.sh exists to catch, and this gate was the one
# it caught.
[[ "$n" -ge 8 ]] \
    || fail "extracted only $n kinds from tk_is_contextual_ident in $HELPER — the list has at least 12 today, so this gate is no longer reading what it claims to"
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
