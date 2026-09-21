#!/usr/bin/env bash
# A module path segment is a NAME, and any word may be one:
# `module resolve::scope`, `module interop::kernel` are real declarations in
# this tree.
#
# parse_module_item consumed the path with a flat
#
#     while parser_peek(p) == TokenKind::Ident
#         || parser_peek(p) == TokenKind::ColonColon
#         || parser_peek(p) == TokenKind::Ontology
#
# so any segment that lexed as some other keyword stopped the loop, the keyword
# fell through to the item dispatcher, and the file did not parse. A compiler
# built from that tree could not parse self-hosted/resolve/scope.sio — a file in
# its own source. `module x::study` was worse than a refusal: `study` HAS a
# dispatcher branch, so it began parsing a study item from inside a module
# header.
#
# Nothing caught it. The shipped artifacts/self-hosted/madaros predates the
# loop — it still used the "consume until Newline" version the code's own
# comment describes replacing — and no gate rebuilds Madaros to re-check the
# tree. The bug was reachable only by building the compiler and pointing it at
# its own sources.
#
# TWO properties are required, and they are load-bearing together:
#
#   1. the segment test asks tk_is_keyword, not `== TokenKind::Ident`
#   2. the shape is STRUCTURAL — one segment, then (:: segment)*
#
# Blanket-accepting keywords in a FLAT loop would be worse than the bug: the
# token after the last segment is usually `struct` or `fn`, themselves keywords,
# so the loop would swallow the rest of the file. The alternation is what makes
# the blanket accept safe. Separate them and this gate goes red.
set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 9

FILE="self-hosted/parser/items.sio"
fail() { echo "PARSER_MODULE_PATH_GATE_FAIL: $*" >&2; exit 1; }

body=$(sed -n '/fn parse_module_item/,/^    }/p' "$FILE")
[[ -n "$body" ]] || fail "parse_module_item not found in $FILE — this gate is no longer reading what it claims to"

if ! grep -q "tk_is_keyword" <<<"$body"; then
    echo "  FAIL  parse_module_item does not ask tk_is_keyword for its path segments" >&2
    echo >&2
    echo "  Testing == TokenKind::Ident alone stops the path at the first keyword" >&2
    echo "  segment, hands the keyword to the item dispatcher, and the file does not" >&2
    echo "  parse. self-hosted/resolve/scope.sio is that file." >&2
    exit 1
fi
echo "  OK    segment test asks tk_is_keyword"

if ! grep -qE "while parser_peek\(p\) == TokenKind::ColonColon[^A-Za-z0-9_]" <<<"$body"; then
    echo "  FAIL  parse_module_item no longer consumes the path structurally" >&2
    echo >&2
    echo "  Accepting keywords in a FLAT loop swallows the \`struct\` or \`fn\` after the" >&2
    echo "  last segment, and then the rest of the file. The alternation is what makes" >&2
    echo "  the blanket accept safe; the two must not be separated." >&2
    exit 1
fi
echo "  OK    path consumed structurally — segment, then (:: segment)*"

echo "PARSER_MODULE_PATH_GATE_OK"
