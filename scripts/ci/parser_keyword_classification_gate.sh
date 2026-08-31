#!/usr/bin/env bash
# Every word the lexer turns into a keyword must be CLASSIFIED: either it is
# reserved in every position, or it is usable as an identifier. A word in
# neither state is how `policy` and `level` got through.
#
# What those two cost, and why a gate rather than a note:
#
#   `policy`  — accepted by parse_prefix, absent from at_expr_start, so
#               `return policy.count` parsed as a BARE return, typed as (), and
#               surfaced as "expected i64, found ()" pointing at the word
#               `return`. Six errors in native_compile_driver.sio, six in
#               lean.sio, misattributed to multimodule checking, duplicate
#               definitions, struct size and field offset before a bisect
#               named it.
#   `level`   — in NO identifier list at all, so `let level = 3` was a parse
#               error. Thirteen of them in stdlib/plot/line.sio, which in turn
#               made every importer look broken. The dissertation's clinical
#               validation test was blamed for those errors for hours.
#
# Neither failed loudly. Both produced numbers that looked like something else.
#
# The companion gate (parser_identifier_keyword_lists_gate.sh) enforces
# parse_prefix ⊆ at_expr_start. It could not have caught `level`, which was in
# neither set — there was no divergence to see. This gate is the complement:
# it asks about the lexer's OUTPUT, not about agreement between two lists.
#
# RESERVED below is measured, not assumed: each entry was probed with
# `let <word>: i64 = 1` against a real compiler and rejected. Adding a keyword
# to the lexer without adding it here — or making it identifier-capable — turns
# this red. Moving a word INTO reserved is a deliberate line in a diff, which is
# the point.
set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 9

python3 - <<'PY'
import io, re, sys

# Words that are keywords in every position. Each was probed and rejected as a
# variable name; the ones with a syntactic form of their own are noted.
RESERVED = {
    "affine", "algebra", "as", "async", "await", "break", "const",
    "contest", "continue", "crate", "effect", "else", "enum", "export",
    "extern", "false", "fn", "for", "handler", "if", "impl", "import", "in",
    "let", "linear", "loop", "match", "models", "mut", "ontology",
    "perform", "pub", "resume", "return", "self", "spawn", "static",
    "struct", "super", "trait", "true", "type", "unsafe", "use", "where",
    "while", "with",
}

# Words that lex as keywords but must remain usable as ordinary identifiers,
# because they are domain vocabulary that appears in real code as variable and
# parameter names.
IDENTIFIER_OK = {
    "acquisition_policy", "alternative_frontier", "alternative_policy",
    "audit", "belief", "box", "chan", "close", "computed", "confidence",
    "copy", "decision_policy", "deferral_policy", "derived", "drop", "dual",
    "dyn", "entropy", "evidence", "handle", "infer", "is", "kernel",
    "knowledge", "level", "likelihood", "max_steps", "measured", "model",
    "module", "monitoring_policy", "move", "objective", "observe", "policy",
    "posterior", "prior", "recourse_policy", "recv", "ref", "require",
    "rollback_route", "route", "sample", "scope", "send", "session", "step",
    "strategy", "study", "transition_policy", "transition_protocol",
    # `on` is a keyword in exactly one position -- the Contest form
    # `contest [ m1, m2 ] on subject` -- and domain vocabulary everywhere else.
    # It was RESERVED here while lean_single had no such class at all, so
    # a parameter or local named `on` parse-failed under Madaros and compiled
    # under lean. Four stdlib files use it as a name and one, graphics/surface.sio,
    # blocked every viz_* test through a three-deep import chain.
    "on",
    "uncertain", "unit", "valid", "validation", "var",
}

# TWO sources, because the first version of this gate read only parser.sio and
# would therefore have missed `policy`, one of the two bugs that motivated it:
# parser.sio carries a by-length byte matcher, lexer/tables.sio a by-index one.
words = {}

src = io.open("self-hosted/parser/parser.sio", encoding="utf-8").read().split("\n")
cur = None
for line in src:
    m = re.search(r"if len == (\d+) \{", line)
    if m:
        cur = int(m.group(1)); continue
    if cur is None:
        continue
    mm = re.search(r"return TokenKind::(\w+)", line)
    if mm and "as i8" in line:
        bs = [int(x) for x in re.findall(r"== \((\d+) as i8\)", line)]
        if len(bs) == cur:
            w = "".join(chr(b) for b in bs)
            if w.isalpha() or "_" in w:     # words only; operators are not at issue
                words[w] = mm.group(1)

for line in io.open("self-hosted/lexer/tables.sio", encoding="utf-8"):
    mm = re.search(r"return TokenKind::(\w+)", line)
    if not mm or "text[" not in line:
        continue
    bs = [int(x) for x in re.findall(r"== (\d+) as i8", line)]
    if not bs:
        continue
    w = "".join(chr(b) for b in bs)
    if w.isalpha() or "_" in w:
        words.setdefault(w, mm.group(1))

if not words:
    print("PARSER_KEYWORD_CLASSIFICATION_GATE_FAIL: found no word-keywords — the "
          "by-length matcher in parser.sio changed shape and this gate is no "
          "longer reading what it claims to read")
    sys.exit(1)

lower = {w: k for w, k in words.items() if w[:1].islower()}
unclassified = sorted(set(lower) - RESERVED - IDENTIFIER_OK)

print(f"  word-keywords in parser.sio's matcher : {len(words)}")
print(f"  lower-case (identifier-shaped)        : {len(lower)}")
print(f"  reserved everywhere                   : {len(RESERVED & set(lower))}")
print(f"  must stay identifier-capable          : {len(IDENTIFIER_OK & set(lower))}")
print()

if unclassified:
    print("PARSER_KEYWORD_CLASSIFICATION_GATE_FAIL: unclassified keyword(s): "
          + ", ".join(unclassified))
    print()
    print("  Each of these lexes as a keyword and nothing says whether that is")
    print("  intended. Decide, in scripts/ci/parser_keyword_classification_gate.sh:")
    print()
    print("    RESERVED       — it has a syntactic form of its own, and code may")
    print("                     not use it as a name. Say which form, in a comment.")
    print("    IDENTIFIER_OK  — it is domain vocabulary. Then add it to the three")
    print("                     identifier lists: parse_prefix (exprs.sio),")
    print("                     parse_pattern_atom (patterns.sio) and")
    print("                     at_expr_start (parser.sio).")
    print()
    print("  Leaving it unclassified is what `policy` and `level` did, and both")
    print("  failed as a wrong NUMBER somewhere else rather than as an error here.")
    sys.exit(1)

print("PARSER_KEYWORD_CLASSIFICATION_OK — every identifier-shaped keyword is")
print("either reserved on purpose or accepted as an identifier.")
PY
rc=$?
[[ $rc -eq 0 ]] || exit $rc
echo "PARSER_KEYWORD_CLASSIFICATION_GATE_OK"
