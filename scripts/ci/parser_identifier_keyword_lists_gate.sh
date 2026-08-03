#!/usr/bin/env bash
# The parser keeps more than one hand-written list of "which keyword may also be
# used as an identifier", and nothing checked that they agree. On 2026-08-03 that
# cost two separate hunts:
#
#   `return policy.count`  parsed as a BARE return. `policy` lexes as
#   TokenKind::PolicyLower; parse_prefix accepts it as an identifier, but
#   at_expr_start — which parse_return_expr asks whether a value follows — did
#   not. So the expression was dropped in SILENCE, the return typed as (), and
#   the error surfaced as "expected i64, found ()" with the span on the word
#   `return`, six times in native_compile_driver.sio and six more in lean.sio.
#   `policy.count` on a line of its own always worked.
#
# The invariant this gate enforces is the one that broke:
#
#     every identifier-like keyword accepted by parse_prefix must also be
#     accepted by at_expr_start
#
# because at_expr_start is what decides whether `return X`, `break X` and their
# relatives take an expression at all, and answering "no" there loses code
# rather than rejecting it. A parser that REFUSES an identifier is a bug you
# find in one run; a parser that silently drops the expression is the kind that
# survives for months.
#
# The pattern list (parse_pattern_atom) is reported but NOT enforced: a keyword
# usable in expression position need not be usable in pattern position, so
# equality there would be a claim this gate cannot justify. Divergence is
# printed so it stays visible.
set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 9

python3 - <<'PY'
import io, re, sys

def tokens_in(path, start_pat, span):
    """Collect TokenKind::X mentioned within `span` lines of the anchor."""
    src = io.open(path, encoding="utf-8").read().split("\n")
    for i, line in enumerate(src):
        if re.search(start_pat, line):
            blob = "\n".join(src[i:i + span])
            return set(re.findall(r"TokenKind::(\w+)", blob)), i + 1
    return None, None

prefix, l_prefix = tokens_in("self-hosted/parser/exprs.sio",
                             r"else if k == TokenKind::Ident \|\| k == TokenKind::Knowledge", 1)
expr_start, l_expr = tokens_in("self-hosted/parser/parser.sio",
                               r"fn at_expr_start", 60)
patterns, l_pat = tokens_in("self-hosted/parser/patterns.sio",
                            r"else if k == TokenKind::Ident \|\| k == TokenKind::Handle", 1)

missing = []
for name, got, line in (("parse_prefix", prefix, l_prefix),
                        ("at_expr_start", expr_start, l_expr),
                        ("parse_pattern_atom", patterns, l_pat)):
    if got is None:
        print(f"  FAIL: could not locate {name} — its shape changed, so this gate "
              f"is no longer reading what it claims to read")
        missing.append(name)
print()
if missing:
    print("PARSER_IDENT_LISTS_GATE_FAIL: anchors not found: " + ", ".join(missing))
    sys.exit(1)

# at_expr_start legitimately carries literals, brackets and operators too; only
# the identifier-like keywords are comparable.
comparable = prefix - {"Ident"}
gap = sorted(comparable - expr_start)

print(f"  parse_prefix        (exprs.sio:{l_prefix})    {len(prefix):2d} kinds")
print(f"  at_expr_start       (parser.sio:{l_expr})     {len(expr_start):2d} kinds")
print(f"  parse_pattern_atom  (patterns.sio:{l_pat})    {len(patterns):2d} kinds")
print()
print("  identifier-like keywords accepted in expression position:")
print("    " + ", ".join(sorted(comparable)))
print()

pat_only = sorted(patterns - prefix - {"Ident"})
pfx_only = sorted(comparable - patterns)
if pat_only or pfx_only:
    print("  pattern-position divergence (reported, not enforced):")
    if pfx_only: print("    accepted in expressions, not in patterns: " + ", ".join(pfx_only))
    if pat_only: print("    accepted in patterns, not in expressions: " + ", ".join(pat_only))
    print()

if gap:
    print("PARSER_IDENT_LISTS_GATE_FAIL: at_expr_start is missing " + ", ".join(gap))
    print()
    print("  These are accepted as identifiers by parse_prefix but at_expr_start")
    print("  does not consider them the start of an expression. The consequence is")
    print("  not a rejection — it is silence: `return <name>.field` becomes a bare")
    print("  `return`, the expression is discarded, and the function reports a")
    print("  type mismatch against () at a span pointing to the word `return`.")
    print("  Add each to at_expr_start in self-hosted/parser/parser.sio.")
    sys.exit(1)

print("PARSER_IDENT_LISTS_OK — every identifier-like keyword parse_prefix accepts")
print("is also an expression start, so `return X` cannot silently lose X.")
PY
rc=$?
[[ $rc -eq 0 ]] || exit $rc
echo "PARSER_IDENTIFIER_KEYWORD_LISTS_GATE_OK"
