#!/usr/bin/env bash
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/gate_artifact.sh"
# fn_type_effect_ratchet_gate.sh — freeze the number of function types that
# carry no effect clause, so the gap to SOUNIO-SPEC-06 §6.0 cannot widen.
#
# §6.0 (founder ruling, 2026-08-19): "A function type carries the effects of the
# function." At the time of the ruling, 559 function types occurred in live .sio
# source and NOT ONE declared an effect. This gate does not implement the
# ruling. It stops the distance from growing while the ruling is unimplemented:
# a new bare function type fails; converting one to carry effects passes and
# lowers the frozen count.
#
# Why a ratchet and not a check: refusing all 559 today would refuse the whole
# repository. Refusing the 560th costs nothing and is the only thing that can be
# true right now.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
# Shared assertions rather than hand-rolled ones: gate_vacuity_gate.sh requires
# them, and it is right to. count_matches separates "no match" from "the tool
# broke", which `grep -c ... || true` collapses into the same 0 — the exact error
# that voided a measurement of mine earlier today.
. "scripts/lib/gate_assert.sh"
gate_name "fn_type_effect_ratchet"

REF="scripts/ci/fn_type_effect_ratchet.frozen"
REF_LIST="scripts/ci/fn_type_effect_ratchet.frozen.list"
OUT="${GATE_ARTIFACT:-artifacts/gates/fn_type_effect_ratchet.json}"

# A function TYPE is `fn(` — no name between `fn` and `(`. A function
# DECLARATION is `fn name(`. The distinction is the whole instrument.
# Only PARAMETER position. A function type in return position ("-> fn(i64) ->
# i64 with Mut") sits on the same line as the enclosing declaration's own `with`
# clause, and no line-local pattern can tell the two apart — an earlier revision
# of this gate counted the outer clause as the type's and undercounted bare
# types. Parameter position is anchored by the `:` that introduces the
# annotation, so the capture cannot reach the outer clause.
PAT_TYPE=':[[:space:]]*fn\([^)]*\)[[:space:]]*->'
# Copilot follow-up (#2570): the return-type tail used to stop at the first
# `,` or `)`, on the assumption that a fn-type's return type is a simple,
# comma-free type. A TUPLE return type (`fn() -> ([f64; 2], i64) with Mut,
# Panic`) breaks that assumption -- the tuple's own internal comma truncates
# the capture before it ever reaches `with`, so an EFFECT-BEARING tuple-
# returning fn-type parameter was misreported as bare. A first fix tolerated
# one level of balanced `(...)`, but tuple elements recursively call
# parse_type (self-hosted/parser/types.sio:865-889), so a NESTED tuple return
# (`fn() -> ((i64, i64), f64) with Mut`) is equally valid source and the
# one-level ERE truncated at the first inner `)` just the same -- there is no
# fixed nesting depth an ERE can bound, since POSIX ERE cannot express
# arbitrary balanced delimiters at all (that needs a context-free grammar,
# not a regular one). Copilot follow-up: replaced the ERE tail with an actual
# balanced-delimiter scan (scan_tail below) that tracks paren depth
# character-by-character and stops at the first `,` or `)` seen at depth 0 --
# correct for any nesting depth, not just zero or one.
#
# Copilot follow-up (#2570): the with-clause detection regex used to be
# passed into awk as a `-v` runtime string (withpat). Now that "does the tail
# contain a with-clause" isn't even the right question (see the AWK_SCAN
# comment below), the detection regex lives directly inside
# count_with_clauses as a lexical regex literal -- consistent with keeping
# PAT_TYPE itself out of the `-v` path -- so there is no longer a bash-side
# `withpat` variable to pass through.

strip_noise() {
  # drop // line comments and "..." string literals before matching, so a
  # function type written inside prose or a message is not counted.
  sed -e 's|//.*$||' -e 's/"[^"]*"//g' "$1"
}

# Copilot follow-up (#2570): scan forward from just after a matched `... ->`
# tracking delimiter depth, so a tuple return type of ANY nesting depth is
# consumed correctly instead of truncating at the first `)` or `,`. Ends at
# an unbalanced `)` seen at depth 0, or a depth-0 `,` that genuinely starts
# the next sibling parameter (see comma_starts_new_parameter below -- NOT
# every depth-0 comma qualifies, since a "with" clause's own effect list is
# itself comma-separated and does not end the declared type).
#
# Copilot follow-up (#2570): depth tracked only `(`/`)`. A multi-argument
# GENERIC return type (`Result<i64, Error>` -- tuple-typed generics
# recursively call parse_type_args, self-hosted/parser/types.sio:307-309) has
# its own internal comma inside `<...>`, at PAREN depth 0, so an
# effect-bearing `fn() -> Result<i64, Error> with IO` parameter was
# truncated at the generic's comma before ever reaching `with` -- the exact
# same failure mode the paren fix addressed, one delimiter kind over. Track
# `<`/`>`, `[`/`]` and `{`/`}` in the SAME depth counter as `(`/`)` (this
# gate only needs "are we nested inside some bracket construct right now",
# not which kind opened it) so a comma or `)` is terminal only once every
# kind of nesting has closed. The enclosing parameter list's own closing
# paren is always literally `)` regardless of what else is nested inside it,
# so only `)` at depth 0 (not `>`/`]`/`}`) is treated as that terminator; a
# stray `>`/`]`/`}` at depth 0 is defensive dead code for malformed input
# and is ignored rather than going negative.
#
# Copilot follow-up (#2570): PAT_TYPE's `\(`/`\)` used to be passed in via
# `awk -v pat="$PAT_TYPE"`. Escape processing of a `-v` value's backslash
# sequences is implementation-defined for a sequence awk doesn't recognise
# (`\(` is not a C-style escape) -- this repo's local mawk keeps the
# backslash, but CI's awk strips it, silently turning `fn\([^)]*\)` into
# `fn([^)]*)`, an ERE GROUP rather than literal parens, which no longer
# requires literal "(" ")" characters at all and broke the positive control
# (measured: CI's awk emitted "escape sequence `\(` treated as plain `(`"
# and the ratchet aborted outright). A regex LITERAL written directly in the
# awk program text (delimited by `/.../`, not a runtime string) is parsed by
# the awk lexer's regex-constant rules, which do not have this ambiguity --
# every conformant awk treats `\(` inside a `/.../` literal as a literal
# paren the same way. PAT_TYPE is spliced into the program text below at
# script-construction time (bash string concatenation, not a `-v` value), so
# it becomes a real lexical regex literal rather than a runtime string.
# Copilot follow-up (#2570): "does the captured tail contain a with-clause
# anywhere" is not the same question as "does the OUTER function type have
# its own with-clause". parse_fn_type (self-hosted/parser/types.sio:908-966)
# parses a return type by recursing into parse_type FIRST and only checks for
# a trailing `with` AFTER that recursive call returns -- so for
# `fn() -> fn() -> i64 with IO`, the INNER `fn() -> i64` is parsed (and
# claims the trailing `with IO` as ITS OWN effects) before the OUTER
# `fn() -> ...` ever gets to check for a with-clause of its own, and by then
# the tokens are already consumed. The outer type is genuinely bare here,
# but the old check (`hit !~ withpat`) saw "with IO" anywhere in the tail and
# called the whole thing non-bare, undercounting a real violation.
#
# The grammar's actual binding rule, worked out from that recursion order:
# consecutive `with` clauses in a return-type chain bind innermost-first --
# the Nth with-clause (counting from the left) binds to the Nth-from-the-
# inside function-type layer. So the OUTER (1st) layer has its own
# with-clause if and only if there are AT LEAST AS MANY with-clauses as
# total layers (1 for the outer own match, plus one more per nested `fn(`
# found in the tail). count_fn_parens / count_with_clauses below count each
# via gsub, and the outer is bare iff with_n < fn_layers.
# Copilot follow-up (#2570): tried disambiguating a bare ">" by checking
# whether the PRECEDING character was "-" (an arrow, not a generic close).
# That assumed "Sounio types never use \"<\"/\">\" as comparison operators" --
# false: a REFINEMENT return type (`{x: i64 | x > 0}`) uses ">" as an actual
# comparison inside the refinement's own `{...}`, not preceded by "-" and not
# a generic close either. A single-character lookback cannot distinguish
# every source of a bare ">"; what CAN is checking whether there is an
# actually-open "<" for it to close. Replaced the single combined depth
# counter with a genuine STACK of open-delimiter characters (`stack[depth]`,
# a local array -- awk gives every extra function parameter, arrays
# included, fresh call-local storage, so no state leaks between the
# separate scan_tail calls enumerate() makes for multiple fn-typed
# parameters on one line): pushing on "(", "<", "[", "{", and popping a
# closer ONLY when it matches what is actually on top of the stack. A ">"
# (or any other closer) that does not match the top -- an arrow's ">", a
# refinement's comparison ">" -- is left alone: neither opens nor closes
# anything, exactly like any other ordinary character in the text.
#
# Copilot follow-up (#2570): the mirror case predicted above -- a
# refinement's own LESS-THAN comparison (`{x: i64 | x < 0}`) -- pushed a
# spurious "<" the same way an unmatched "(" would corrupt the stack: with
# that "<" on top, the refinement's own "}" no longer matches (top is now
# "<", not "{"), so neither the spurious "<" nor the real "{" ever pop, and
# depth never returns to 0 for the rest of the scan -- the tail then runs
# past its real terminator entirely, potentially consuming a "with" clause
# that belongs to a DIFFERENT, unrelated enclosing declaration and reporting
# a genuinely bare type as non-bare (undercounting, the opposite direction
# from every earlier finding here). Unlike ")"/">"/"]"/"}", an opening "<"
# has no stack to consult -- there is nothing yet to disambiguate it
# against. What distinguishes a generic's opening "<" from a comparison's is
# local text shape instead: this codebase always writes a generic tight
# against its type name (`Result<`, `Vec<`, no space), while a comparison is
# always written with a space on both sides (`x < 0`, matching the `x > 0`
# shape Copilot's own earlier example already used). Only push "<" when the
# character immediately before it is an identifier character; a "<"
# preceded by anything else (whitespace, punctuation, start of the tail) is
# left alone, same as any other ordinary character.
#
# Copilot follow-up (#2570): every fix above sharpened scan_tail, the part
# that runs AFTER a match is found -- but PAT_TYPE itself, the part that
# FINDS a match, has the exact same "naive `[^)]*`" flaw scan_tail's tail
# once had. `register_test: fn(&!TestRunner, string, fn() -> TestResult,
# TestMetadata)` (an existing line, tools/test-framework/src/lib.sio:390) has
# a NESTED fn-type PARAMETER, not just a nested return type: PAT_TYPE's own
# `fn\([^)]*\)` stops at the FIRST `)` -- the INNER `fn()`'s own closing
# paren -- so it reads "fn(&!TestRunner, string, fn()" as if that whole span
# were the OUTER type's parameter list, then finds " -> TestResult" right
# after and treats THAT as the outer's return type. The result is one
# garbled hit that is neither the real outer type (which has no arrow at all
# -- itself bare, but for a different reason) nor the real inner one
# (`fn() -> TestResult`, independently bare), and a NESTED fn-type parameter
# is never visited on its own terms: whether IT carries an effects clause
# has no bearing on whether it gets counted, so adding one there cannot
# raise the ratchet, and removing one cannot lower it.
#
# Fixed by replacing PAT_TYPE-as-a-single-regex-match with a genuine
# recursive descent: `match_close_paren` finds the TRUE matching `)` for a
# given `(` via the same stack-matching scan_tail already uses (so it is
# just as correct against arrows, refinements and generics nested inside).
# `classify_fn_type_at`, given the position of an `fn(`, bracket-matches its
# own parameter list, checks for its OWN arrow/effects via scan_tail (same
# scope PAT_TYPE always had: no arrow at all is not counted here, matching
# its existing "return position" exclusion rather than silently widening
# what "bare" means), and then walks that parameter list splitting on its
# own depth-0 commas -- any entry that starts with "fn(" is a nested
# function-type parameter, classified by recursing into this SAME function.
# That recursion is what visits `register_test`'s inner `fn() -> TestResult`
# independently, and continues to whatever depth further nesting occurs at.
#
# Copilot follow-up (#2570): "push `<` only when preceded by an identifier
# character" was itself still a formatting heuristic, and Sounio does not
# require whitespace around a refinement's own comparison operators
# (self-hosted/parser/types.sio:697-705 tokenizes them independently of
# whitespace). `{x:i64|x<0}` (no spaces at all) has an identifier ("x")
# directly before its comparison "<" too, so the old heuristic pushed it as
# a generic opener exactly as it once pushed `x < 0`'s spaced version --
# same corruption, same undercount, just reachable without a space. No
# purely LOCAL character-shape rule (spacing, or anything else about the
# one or two characters immediately around "<") can distinguish these,
# because the two spellings are locally identical; what differs is
# GRAMMATICAL CONTEXT -- whether we are past the "|" that starts a
# refinement's predicate. So `pred[depth]` tracks that directly: pushing a
# new bracket inherits its enclosing level's predicate state, a literal "|"
# seen while the current bracket is "{" marks that level as predicate mode,
# and once in predicate mode a bare "<" is left alone unconditionally --
# not a question of what precedes it at all, because inside an active
# predicate every "<"/">" is a comparison, full stop. Factored the
# open/other-close handling out of match_close_paren, scan_tail and the
# parameter-list walker into one shared step_open_or_other_close, so this
# fix (and any future one to this exact logic) lives in a single place
# instead of three independently-drifting copies.
#
# Copilot follow-up (#2570): an unmatched ")" at depth 0 has always meant
# "we have left the scope this scan was searching" (match_close_paren and
# scan_tail both handle it directly, as their own terminator). "]"/">"/"}"
# never got that same treatment -- at depth 0 they were silent no-ops -- so a
# NESTED fn-type wrapped inside an array or generic parameter
# (`f: fn([fn() -> i64; 3]) -> i64`, `f: fn(Vec<fn() -> i64>) -> i64`) had no
# way to signal "this is where MY enclosing array/generic ends" when its own
# return-type scan reached that enclosing "]"/">" : scan_tail just kept
# going, silently absorbing whatever followed as if it were still part of
# the nested type's own return type. step_open_or_other_close now returns
# -1 for exactly that case (a closer with nothing of its own kind open to
# match) -- the same "we have left this scope" signal a depth-0 ")" already
# gives -- and every caller checks for it exactly the way match_close_paren
# already checked for ")".
#
# ">" needs one more guard beyond that: it is also the second character of
# every "->" arrow, and unlike ")"/"]"/"}", an arrow's ">" can legitimately
# occur at depth 0 constantly (`fn() -> i64` has one on every plain,
# non-generic function type) -- treating THOSE as "we have left this scope"
# broke ordinary direct return-type chains outright. Checked first and
# unconditionally: a ">" preceded by "-" is never a closer and never a
# terminator, regardless of depth or stack state.
#
# Copilot follow-up (#2570): every anchor and every recursive "fn(" search
# required NO whitespace between "fn" and "(" -- a hardcoded 3-character
# span, baked into fn_pos arithmetic in several places. self-hosted/parser/
# types.sio's own lexer skips whitespace before parse_fn_type expects "(",
# so `f: fn (i64) -> i64` is equally valid source and was invisible to every
# one of those checks. skip_ws replaces the fixed "+2" offsets with an
# actual scan past any whitespace/newlines between "fn" and "(", and the
# "fn[ \t\n]*\(" regexes used for nested-type discovery (count_fn_parens,
# scan_entry_for_fn_types) tolerate the same gap. The TOP-LEVEL anchor
# could not just widen its own regex the same way, because it used to
# compute fn_pos by subtracting a fixed 3 from the match's end
# (`mstart + mlen - 3`) -- correct only when the match is exactly ":fn("
# with no extra characters. Replaced with an explicit scan instead: find
# each ":", skip whitespace, check for "fn", skip whitespace again, check
# for "(" -- using match()''s own RSTART/RLENGTH only to locate candidate
# colons quickly, never to back-compute a fixed-width span.
# Copilot follow-up (#2570): the "<" branch below only ever pushed a
# generic-open when the LITERAL immediately-preceding character was an
# identifier character -- but whitespace between a type name and its
# generic opener is lexically insignificant (self-hosted/parser/types.sio
# skips it the same way it skips whitespace everywhere else), so
# `Result <i64, Error>` is equally valid source. With a space there, prevc
# held " " at the "<", the push never happened, and the later ">" then hit
# the depth-0 case and was treated as "we have left this scope" -- exactly
# the same class of false terminator step_open_or_other_close was already
# built to avoid for genuinely unmatched closers. Fixed at the tracking
# site, not the check: every prevc update (match_close_paren, scan_tail,
# and the parameter-list entry-walker) now skips whitespace instead of
# overwriting prevc with it, so prevc always holds the last NON-whitespace
# character seen, matching what the "<" and the arrow ("-" before ">")
# checks actually care about. Safe for the arrow check too: a genuine "->"
# arrow is a single adjacent two-character token in valid Sounio source --
# nothing legitimate ever separates "-" from ">" with whitespace -- so
# widening prevc to "last non-whitespace" cannot misfire there.
AWK_SCAN='
function skip_ws(line, p,    n) {
    n = length(line)
    while (p <= n && substr(line, p, 1) ~ /[ \t\n]/) { p++ }
    return p
}
function step_open_or_other_close(c, prevc, depth, stack, pred) {
    if (c == "(" || c == "[" || c == "{") {
        depth++
        stack[depth] = c
        pred[depth] = pred[depth - 1]
        return depth
    }
    if (c == "<") {
        if (pred[depth] != 1 && prevc ~ /[A-Za-z0-9_]/) {
            depth++
            stack[depth] = "<"
            pred[depth] = pred[depth - 1]
        }
        return depth
    }
    if (c == "|") {
        if (depth > 0 && stack[depth] == "{") { pred[depth] = 1 }
        return depth
    }
    if (c == ">") {
        if (prevc == "-") { return depth }
        if (depth > 0 && stack[depth] == "<") { return depth - 1 }
        if (depth == 0) { return -1 }
        return depth
    }
    if (c == "]") {
        if (depth > 0 && stack[depth] == "[") { return depth - 1 }
        if (depth == 0) { return -1 }
        return depth
    }
    if (c == "}") {
        if (depth > 0 && stack[depth] == "{") { return depth - 1 }
        if (depth == 0) { return -1 }
        return depth
    }
    return depth
}
function match_close_paren(line, open_pos,    i, n, c, prevc, depth, stack, pred, r) {
    depth = 1
    stack[1] = "("
    pred[1] = 0
    prevc = ""
    n = length(line)
    i = open_pos + 1
    while (i <= n) {
        c = substr(line, i, 1)
        if (c == ")") {
            if (stack[depth] == "(") { depth--; if (depth == 0) { return i } }
        } else {
            r = step_open_or_other_close(c, prevc, depth, stack, pred)
            if (r == -1) { return i }
            depth = r
        }
        if (c !~ /[ \t\n]/) { prevc = c }
        i++
    }
    return n + 1
}
# Copilot follow-up (#2570): general counterpart of match_close_paren, for a
# top-level colon whose type starts with "(" or "[" directly -- a tuple or
# array type wrapping a function type, e.g. `f: (fn() -> i64, i64)`. Same
# structure as match_close_paren (seed the stack with the ALREADY-CONSUMED
# opener, walk from open_pos+1, close only on the matching closer), just
# parameterized by which opener/closer pair this call is matching instead of
# hard-coding "(" / ")".
#
# ")" needs its OWN explicit branch here, separate from step_open_or_other_
# close, the same way match_close_paren and scan_tail both give it one:
# step_open_or_other_close never handles ")" at all -- by design, since
# EVERY existing caller already intercepts ")" itself before delegating
# anything else to it. When the outer bracket THIS matcher is closing is
# "[" (not "("), a plain delegation of ")" here fell through step_open_or_
# other_close catch-all `return depth` unchanged: a nested "(" (from a
# "fn(" found by scan_entry_for_fn_types inside this span) got pushed
# correctly via that same delegation, but its matching ")" was silently a
# no-op -- never popped -- corrupting depth for the rest of the scan and
# leaving the array real "]" unrecognized (stack[depth] was still "(", not
# "[").
#
# Copilot follow-up (#2570): "<" is also a supported open_char now, for a
# top-level NAMED generic wrapper (`Vec<fn() -> i64>`). The arrow check
# (a ">" immediately preceded by "-" is never a closer) has to be its OWN
# unconditional FIRST check here too, the same way step_open_or_other_close
# already gives it one: when close_char is ">", a plain "if (c ==
# close_char)" would otherwise treat the arrow inside "fn() -> i64" as the
# generic closing ">" and return one character too early -- the exact
# "Vec<fn() -> i64>" bug already hard-won once for step_open_or_other_close
# itself (see its own comment), now needing the identical guard here
# because this function intercepts the outer closer BEFORE delegating
# anything else to that function.
function match_close_bracket(line, open_pos, open_char,    close_char, i, n, c, prevc, depth, stack, pred, r) {
    if (open_char == "[") { close_char = "]" } else if (open_char == "<") { close_char = ">" } else { close_char = ")" }
    depth = 1
    stack[1] = open_char
    pred[1] = 0
    prevc = ""
    n = length(line)
    i = open_pos + 1
    while (i <= n) {
        c = substr(line, i, 1)
        if (c == ">" && prevc == "-") {
            # inert arrow, never a closer regardless of what open_char is
        } else if (c == close_char) {
            if (stack[depth] == open_char) { depth--; if (depth == 0) { return i } }
        } else if (c == ")") {
            if (stack[depth] == "(") { depth--; if (depth == 0) { return i } }
        } else {
            r = step_open_or_other_close(c, prevc, depth, stack, pred)
            if (r == -1) { return i }
            depth = r
        }
        if (c !~ /[ \t\n]/) { prevc = c }
        i++
    }
    return n + 1
}
# Copilot follow-up (#2570): a depth-0 comma after "with" is NOT always the
# end of the declared type -- an effects list is itself comma-separated
# (`with IO, Mut`), and a fn-type nested as a return type carries its OWN
# "with" clause independently of whatever effects clause encloses it
# (`fn() -> fn() -> i64 with IO, Mut with Panic`: the INNER "with IO, Mut"
# and the OUTER "with Panic" are two separate clauses, neither one a
# sibling-parameter separator). scan_tail used to terminate on EVERY
# depth-0 comma unconditionally, so it stopped right after "IO", never
# reaching "Mut with Panic" -- count_with_clauses then saw only one "with"
# for what is actually two effectful layers, misreporting a bare hit.
#
# Distinguishes a genuine sibling DECLARATION parameter (`, x: i64` -- an
# identifier followed by ":") from another effect name in the SAME or a
# later with-clause (`, Mut`, `, Mut with Panic` -- an identifier NOT
# followed by ":") by peeking past the comma: skip whitespace, skip one
# identifier, skip whitespace again, and check for ":".
#
# Copilot follow-up (#2570): gated on `seen_with` (only applied to a comma
# AFTER a "with" has actually been seen in THIS tail scan) after a real
# corpus run caught the naive "check every depth-0 comma" version
# regressing a genuinely different shape: a nested fn-type used as ONE
# BARE-TYPE entry inside the OWN parameter list of an OUTER fn-type (no "with"
# anywhere at all) is followed by a SIBLING bare type with no colon either
# (`fn(A, fn() -> TestResult, TestMetadata)` -- TestMetadata has no colon,
# it is a type, not a "name: type" declaration parameter). Without the
# `seen_with` gate, the colon-lookahead alone could not tell that comma
# apart from an effect-list continuation and swallowed "TestMetadata)" into
# the tail. Gating on `seen_with` restores the original (correct)
# behavior for that case -- terminate on the very first depth-0 comma, no
# "with" ever having been seen -- while still applying the colon-lookahead
# once inside a genuine effects list, where it correctly tells "another
# effect name" apart from "a real declaration parameter with a colon"
# (the `with IO, Mut, x: i64` sibling-after-effects shape, NEGATIVO 28).
function comma_starts_new_parameter(line, comma_pos,    p, n) {
    n = length(line)
    p = skip_ws(line, comma_pos + 1)
    if (p > n || substr(line, p, 1) !~ /[A-Za-z_]/) { return 1 }
    while (p <= n && substr(line, p, 1) ~ /[A-Za-z0-9_]/) { p++ }
    p = skip_ws(line, p)
    return (p <= n && substr(line, p, 1) == ":")
}
function scan_tail(line, tail_start,    i, n, c, prevc, depth, stack, pred, r, seen_with) {
    depth = 0
    prevc = ""
    seen_with = 0
    n = length(line)
    i = tail_start
    while (i <= n) {
        c = substr(line, i, 1)
        # Copilot follow-up (#2570): the word-boundary check before "with"
        # must look at the RAW immediately-preceding character
        # (substr(line, i-1, 1)), not `prevc` -- `prevc` deliberately skips
        # whitespace (tracks the last NON-whitespace character, the
        # established convention this whole scanner already relies on
        # elsewhere), so for "i64 with" it holds "4" (alnum) right at the
        # "w" of "with", even though a real whitespace character sits
        # directly between them. Checking `prevc` here always misfired on
        # exactly the common case (a space before "with"), so this
        # detection never actually armed and the fix silently did nothing.
        if (!seen_with && (i == tail_start || substr(line, i - 1, 1) !~ /[A-Za-z0-9_]/) && substr(line, i, 4) == "with" && substr(line, i + 4, 1) ~ /[ \t\n]/) {
            seen_with = 1
        }
        if (c == ")") {
            if (depth == 0) { return i }
            if (stack[depth] == "(") { depth-- }
        } else if (c == "," && depth == 0) {
            if (!seen_with || comma_starts_new_parameter(line, i)) { return i }
        } else {
            r = step_open_or_other_close(c, prevc, depth, stack, pred)
            if (r == -1) { return i }
            depth = r
        }
        if (c !~ /[ \t\n]/) { prevc = c }
        i++
    }
    return n + 1
}
function count_fn_parens(str,    tmp) {
    tmp = " " str
    return gsub(/[^A-Za-z0-9_]fn[ \t\n]*\(/, "@", tmp)
}
function count_with_clauses(str,    tmp) {
    tmp = " " str
    return gsub(/[^A-Za-z0-9_]with[ \t\n]+[A-Za-z]/, "@", tmp)
}
# Copilot follow-up (#2570): the entry-walker used to check only whether a
# parameter-list ENTRY, after trimming leading whitespace, itself STARTS
# with "fn(" -- so `f: fn((fn() -> i64, i64)) -> i64` (a tuple-wrapped
# nested fn-type parameter) was missed entirely: the entry text is
# "(fn() -> i64, i64)", which does not start with "fn(", even though it
# contains one. The same hole applies to an array (`[fn() -> i64; 3]`) or a
# generic (`Vec<fn() -> i64>`) wrapping a fn-type parameter. Scans the WHOLE
# entry text for "fn(" at ANY position (word-bounded -- not part of a longer
# identifier) instead of only its first token, recursing into
# classify_fn_type_at at each one found. Copilot follow-up (#2570): "fn(" was
# a fixed 3-character literal, requiring no whitespace between "fn" and "(" --
# self-hosted/parser/types.sio own lexer skips whitespace before
# parse_fn_type expects "(", so `fn (i64)` is equally valid; tolerated here
# too (RSTART still correctly locates "f" of "fn" regardless of how much
# whitespace the match consumes afterward, so no other arithmetic here needs
# to change).
#
# Copilot follow-up (#2570): advancing past just the matched "fn(" token
# double-counted a chained return type used AS a parameter.
# `f: fn(fn() -> fn() -> i64) -> i64` has an entry whose own text is
# "fn() -> fn() -> i64" -- a 2-layer direct chain. classify_fn_type_at,
# called on the entry FIRST "fn(", already walks and counts the WHOLE
# chain internally via fn_layers/with_n (both of its layers, correctly, two
# hits). But this loop then kept searching from just past that first match,
# found the SAME chain second "fn(" again as if it were an unrelated
# sibling entry, and classified it a second time -- three hits for what
# classify_fn_type_at own arithmetic had already fully accounted for in
# two, four total against the correct three (the entry own outer type
# plus its two chain layers). Resume the search from `result` -- the
# position classify_fn_type_at itself reports its own span ends at -- not
# from just past the matched token, so a chain nested this way is walked
# exactly once. A GENUINE sibling "fn(" (two different fn-type elements of
# a wrapped tuple, e.g. `(fn() -> i64, fn() -> f64)`) still starts strictly
# after where the first one own span ended, so it is still found the
# normal way.
function scan_entry_for_fn_types(line, entry_start, entry_end, at_eof,    seg, pos_in_seg, abs_match_start, prevc, mstart, mlen, result) {
    seg = substr(line, entry_start, entry_end - entry_start)
    pos_in_seg = 1
    while (pos_in_seg <= length(seg) && match(substr(seg, pos_in_seg), /fn[ \t\n]*\(/)) {
        # RSTART/RLENGTH are awk globals set by match() -- save them before
        # the recursive classify_fn_type_at call below runs its own match()
        # calls internally and overwrites them out from under this loop
        # (measured: without this, pos_in_seg advanced by whatever the LAST
        # nested match() call happened to leave behind, not this loop own
        # match, re-finding and re-classifying the same "fn(" repeatedly).
        mstart = RSTART
        mlen = RLENGTH
        abs_match_start = entry_start + (pos_in_seg - 1) + mstart - 1
        if (abs_match_start > 1) { prevc = substr(line, abs_match_start - 1, 1) } else { prevc = "" }
        if (prevc !~ /[A-Za-z0-9_]/) {
            result = classify_fn_type_at(line, abs_match_start, at_eof)
            if (result != -1 && result > abs_match_start) {
                pos_in_seg = result - entry_start + 1
            } else {
                pos_in_seg = pos_in_seg + (mstart - 1) + mlen
            }
        } else {
            pos_in_seg = pos_in_seg + (mstart - 1) + mlen
        }
    }
}
function classify_fn_type_at(line, fn_pos, at_eof,    open_pos, close_pos, after, rest, hit, hit_tail, fn_layers, with_n, bare_layers, k, tail_start, tail_end, entry_start, i, n, c, prevc, depth, stack, pred, r) {
    open_pos = skip_ws(line, fn_pos + 2)
    close_pos = match_close_paren(line, open_pos)
    # Copilot follow-up (#2570): a genuine `n + 1` from match_close_paren
    # (see its own comment) means "ran off the end of the searchable text
    # without finding a match" -- for THIS caller, "the searchable text" is
    # only the current line/buffer, not the whole file, so that could mean
    # either a truly malformed type OR a valid one whose closing paren is on
    # a LATER line not read yet (`f: fn(\n  i64\n) -> i64`). -1 tells the
    # caller "not yet resolvable, try again with more text" instead of
    # silently treating it as arrow-less and out of scope -- unless at_eof
    # is set, meaning there IS no more text coming (see the END block
    # below), so `n + 1` here really is the honest last word.
    if (!at_eof && close_pos > length(line)) { return -1 }
    after = close_pos + 1
    rest = substr(line, after)
    if (match(rest, /^[ \t\n]*->/)) {
        tail_start = after + RLENGTH
        tail_end = scan_tail(line, tail_start)
        hit_tail = substr(line, tail_start, tail_end - tail_start)
        # Copilot follow-up (#2570): deliberately not ALWAYS deferring here
        # the way close_pos above does -- but the narrower finding predicted
        # in an earlier version of this comment did arrive: `f: fn() ->` on
        # one physical line, `(i64, i64) with Mut` on the next (self-hosted/
        # parser/types.sio:935-951 parses the return type and effects as
        # tokens across whitespace, so this is equally valid source). There,
        # tail_start already points past the very end of `line` (nothing at
        # all followed the arrow yet), scan_tail returns immediately, and
        # hit_tail is empty -- genuinely no information yet, not "there is
        # real content and it happens not to close". Measured on the live
        # corpus separately: `let sin_fn: fn(c_double) -> c_double = unsafe {
        # match ... }` (examples/ffi_demo.sio) -- after the return type,
        # " = unsafe {" is the LET-BINDING own initializer, not more type
        # syntax, but its "{" still looks like an opener to
        # step_open_or_other_close (which cannot tell a refinement "{" from
        # an unrelated code block one), so deferring UNCONDITIONALLY here
        # tried to bracket-match through the ENTIRE unsafe block as if it
        # were part of the type, corrupting the hit. hit_tail there is
        # "c_double = unsafe {" -- real, substantial, non-whitespace content
        # -- which is exactly what distinguishes the two: defer only when
        # hit_tail, after running off the end, is ENTIRELY whitespace (truly
        # nothing seen yet); anything else means real content was found and
        # simply does not close within what has been read so far, which
        # must NOT defer, for the same reason ffi_demo.sio must not.
        if (!at_eof && tail_end > length(line) && match(hit_tail, /^[ \t\n]*$/)) {
            return -1
        }
        hit = substr(line, fn_pos, tail_end - fn_pos)
        # A hit assembled across multiple physical lines still carries their
        # newlines; enumerate() and bare_hits_of() both expect one hit per
        # OUTPUT line, so collapse them to spaces before printing.
        gsub(/\n/, " ", hit)
        fn_layers = 1 + count_fn_parens(hit_tail)
        with_n = count_with_clauses(hit_tail)
        # Copilot follow-up (#2570): `fn_layers - with_n` is not just "is
        # the outer layer bare" -- it is literally the COUNT of bare layers
        # in this chain, by the same innermost-first with-clause binding
        # worked out earlier: the innermost with_n layers each get one of
        # the with-clauses, leaving exactly fn_layers - with_n outer layers
        # with none. `fn() -> fn() -> i64` (two layers, no with-clauses at
        # all) is TWO distinct bare function types, not one -- printing the
        # hit only once (whenever with_n < fn_layers was true at all)
        # undercounted every chain with more than one bare layer, and
        # adding a THIRD bare layer to an existing bare chain could never
        # raise the ratchet. Emit the hit once per bare layer instead of
        # once per match.
        bare_layers = fn_layers - with_n
        for (k = 0; k < bare_layers; k++) { print hit }
    } else if (!at_eof && match(rest, /^[ \t\n]*$/)) {
        # Copilot follow-up (#2570): a valid bare type formatted with the
        # arrow on its OWN line (`f: fn()\n -> i64`) closed its parameter
        # list within bounds (the close_pos check above passed), but nothing
        # follows the close paren in what has been read SO FAR except
        # whitespace/newlines -- genuinely inconclusive, not "no arrow",
        # since an arrow could still be the very next non-whitespace token
        # once a further line is read. Only when `rest` is ENTIRELY
        # whitespace is this ambiguous; any other non-arrow content
        # immediately following (a comma, a closing paren, "= unsafe {") is
        # conclusive -- see the case above this one for why those must NOT
        # defer.
        return -1
    } else {
        tail_end = close_pos + 1
    }
    entry_start = open_pos + 1
    i = entry_start
    depth = 0
    prevc = ""
    n = close_pos
    while (i < n) {
        c = substr(line, i, 1)
        if (c == "," && depth == 0) {
            # A nested entry is, by construction, entirely within [entry_start,
            # close_pos) -- a span already confirmed present in `line` (close_pos
            # itself passed the bound check above, or at_eof waived it). Its own
            # match_close_paren therefore cannot legitimately need more text than
            # `line` already has.
            scan_entry_for_fn_types(line, entry_start, i, at_eof)
            entry_start = i + 1
        } else if (c == ")") {
            if (depth > 0 && stack[depth] == "(") { depth-- }
        } else {
            r = step_open_or_other_close(c, prevc, depth, stack, pred)
            # An unmatched closer inside an already-bounded parameter list
            # can only mean malformed input (nothing legitimate enclosing
            # THIS span could still be open) -- skip it rather than treat it
            # as ending the walk early.
            if (r == -1) { if (c !~ /[ \t\n]/) { prevc = c }; i++; continue }
            depth = r
        }
        if (c !~ /[ \t\n]/) { prevc = c }
        i++
    }
    scan_entry_for_fn_types(line, entry_start, n, at_eof)
    return tail_end
}
# Copilot follow-up (#2570): the top-level driver below only ever entered
# classify_fn_type_at when the token right after a ":" was literally "fn" --
# so a parameter whose declared type WRAPS a function type one level out,
# `f: (fn() -> i64, i64)` (a tuple) or `f: [fn() -> i64; 3]` (an array), was
# invisible: the token right after the COLON here is "(" / "[", not "fn",
# and the scanner moved straight on to the next ":" without ever looking
# INSIDE the wrapper. The entry-walker used by classify_fn_type_at itself
# (scan_entry_for_fn_types) already handles exactly this shape -- but only
# when reached from WITHIN an outer fn(...) own parameter list, which this
# top-level case is not. match_close_bracket (below) finds the matching
# closer for the wrapper the same way match_close_paren does for a "fn(",
# then scan_entry_for_fn_types searches the WHOLE span between the brackets
# for any nested "fn(" the normal way.
#
# Deliberately SAME-LINE ONLY, no pending/deferral. Copilot follow-up
# (#2570): tried enabling deferral here on the theory that strip_noise
# (stripping `//` comments and "..." string literals before this scanner
# ever runs) already removes the ordinary-text triggers a blanket ":"
# followed by "(" / "[" could misfire on. That theory was WRONG, confirmed
# by actually enabling it and re-running the real corpus (not assumed):
# the run hung, bisected by prefix-truncating files down to the exact
# triggering line -- self-hosted/native/codegen_x86_linux.sio:483, a STRING
# LITERAL containing an escaped quote, `"\":["`. The strip_noise regex
# itself, `s/"[^"]*"//g`, does not understand `\"` as an escaped quote
# inside a string (POSIX BRE/ERE has no lookbehind, and this is a plain
# substitution, not a real lexer): it matches from the literal opening
# quote to the FIRST quote character it finds AT ALL -- the escaped one --
# stripping only `"\"` and leaving that literal own `:[` content and
# trailing `"` behind as if they were ordinary code. The result has a
# genuine, permanent `:[` with no matching `]` anywhere in the rest of the
# file (it never was a real bracket), so `pending` grew for the remainder
# of the file and never cleared -- exactly the failure mode this
# same-line-only design was already built to avoid, just from a different
# and more surprising source than plain "comments/strings" as originally
# assumed. This escaped-quote gap in strip_noise is real and could be fixed
# separately (a job for a proper escape-aware regex, e.g.
# a two-branch alternation of "non-quote-non-backslash" and
# "backslash-anything", under -E), but that is its own change to a function
# several existing selftest controls already depend on, and is not needed
# to fix THIS finding safely -- same-line-only already fails closed against
# the corruption either way (no match found on this line, so nothing is
# reported, rather than hanging).
# Copilot follow-up (#2570): this whole scanner was line-local -- `line = $0`
# reset fresh every record, so `f: fn(` on one physical line followed by
# `i64` and `) -> i64` on the next two never resolved at all: match_close_paren
# ran off the end of the FIRST line, classify_fn_type_at (before this fix)
# had no way to say "wait, there might be more" and just treated it as
# arrow-less, letting a genuinely bare multiline type bypass the ratchet
# silently. Buffering the WHOLE FILE per awk record (`buf = buf $0 "\n"`)
# was tried and measured: ~9s just to concatenate lower.sio (26k lines) in
# this repo awk, before any scanning even starts -- unusable across the
# whole corpus. Instead, only ever carry over `pending`: normally empty (no
# cost for the overwhelming majority of single-line declarations), and only
# ever holds the tail of an in-progress multiline match, bounded by how many
# lines that ONE declaration actually spans, not by file size (measured:
# lower.sio drops back to ~0.03s with this scoped carry-over).
{
    if (pending != "") {
        line = pending "\n" $0
    } else {
        line = $0
    }
    if (pending != "") {
        # `pending` always starts exactly at a "fn(" already confirmed to
        # follow a `:` on an earlier line, so retry it directly rather than
        # re-running the `:...` anchor search (which would fail here -- the
        # `:` that justified this match is no longer in `pending`).
        result = classify_fn_type_at(line, 1, 0)
        if (result == -1) { pending = line; next }
        pending = ""
        pos = result
    } else {
        pos = 1
    }
    # Copilot follow-up (#2570): used to match ":[ \t\n]*fn\(" as one regex
    # and compute fn_pos by subtracting a fixed 3 from the match end
    # (mstart + mlen - 3) -- correct only when the match is exactly ":fn("
    # with no extra characters, which broke the moment "fn" and "(" could
    # have whitespace between them (see skip_ws above). Finds each ":" via
    # match() (fast, still lets the awk regex engine do the coarse search),
    # then does the fine-grained "is this really followed by fn(" check with
    # explicit skip_ws-based position arithmetic instead of trying to fold
    # it all into one regex match length.
    while (pos <= length(line) && match(substr(line, pos), /:/)) {
        colon_pos = pos + RSTART - 1
        p = skip_ws(line, colon_pos + 1)
        if (substr(line, p, 2) == "fn") {
            p2 = skip_ws(line, p + 2)
            if (substr(line, p2, 1) == "(") {
                result = classify_fn_type_at(line, p, 0)
                if (result == -1) { pending = substr(line, p); next }
                pos = result
                continue
            }
        } else if (substr(line, p, 1) == "(" || substr(line, p, 1) == "[") {
            # Copilot follow-up (#2570): a declared type that WRAPS a
            # function type one level out -- `f: (fn() -> i64, i64)` (tuple)
            # or `f: [fn() -> i64; 3]` (array) -- never reaches the "fn"
            # branch above (the token right after ":" is the wrapper opener,
            # not "fn"), so classify_fn_type_at was never invoked at all and
            # a bare fn-type nested inside one of these wrappers bypassed the
            # ratchet silently. match_close_bracket finds the matching closer
            # for the wrapper, then scan_entry_for_fn_types searches the
            # WHOLE span between the brackets for any nested "fn(" the normal
            # way. Deliberately SAME-LINE ONLY (see the comment above
            # match_close_bracket for why deferring this across lines is
            # unsafe): if the closer is not on this line, this declaration
            # is skipped rather than carried into `pending`.
            close_pos = match_close_bracket(line, p, substr(line, p, 1))
            if (close_pos <= length(line)) {
                scan_entry_for_fn_types(line, p + 1, close_pos, 0)
                pos = close_pos + 1
                continue
            }
        } else if (substr(line, p, 1) ~ /[A-Za-z_]/) {
            # Copilot follow-up (#2570): a declared type that wraps a
            # function type in a NAMED GENERIC (`f: Vec<fn() -> i64>`) is
            # a third wrapper shape alongside the tuple/array ones just
            # above -- the token right after ":" is an identifier, not
            # "fn"/"("/"[", so this fell through untouched too.
            # scan_entry_for_fn_types can already find a generic-wrapped
            # fn-type once reached (it already handles this shape from
            # WITHIN an outer fn(...) parameter list, selftest 18) -- the
            # gap was only ever in getting here from the top level. Skips
            # the identifier, tolerates whitespace before "<" the same way
            # the spaced-generic-return-type fix does. Same-line-only, same
            # reasoning as the tuple/array branch just above.
            gp = p
            while (gp <= length(line) && substr(line, gp, 1) ~ /[A-Za-z0-9_]/) { gp++ }
            gp = skip_ws(line, gp)
            if (substr(line, gp, 1) == "<") {
                close_pos = match_close_bracket(line, gp, "<")
                if (close_pos <= length(line)) {
                    scan_entry_for_fn_types(line, gp + 1, close_pos, 0)
                    pos = close_pos + 1
                    continue
                }
            }
        }
        pos = colon_pos + 1
    }
}
END {
    # Whatever is left in `pending` ran off the end of every line this file
    # had -- there really is no more text coming now, so finalize it with
    # at_eof=1, which is exactly the non-deferring behavior this scanner
    # always had before this fix (an `n + 1` sentinel simply means "runs to
    # the end of the searchable text", the correct reading once that text is
    # truly exhausted).
    if (pending != "") {
        classify_fn_type_at(pending, 1, 1)
    }
}
'

enumerate() {
  git ls-files -z '*.sio' \
    | tr '\0' '\n' \
    | grep -vE '^(archive|bootstrap)/' \
    | grep -vE '\.sio\.old$' \
    | while IFS= read -r f; do
        [ -f "$f" ] || continue
        strip_noise "$f" | awk "$AWK_SCAN" | while IFS= read -r hit; do
          # a bare function type is one whose text carries no `with` clause
          printf '%s\t%s\n' "$f" "$hit"
        done
      done
}

# Copilot follow-up (#2570): shared by the positive/negative-3/5/6/7 selftest
# controls below and by enumerate() itself -- runs the same balanced scan and
# returns only the BARE hits (the awk script already filters on withpat), so
# a control checking "is this reported as bare" and enumerate() can never
# disagree about what bare means.
bare_hits_of() {
  strip_noise "$1" | awk "$AWK_SCAN"
}

selftest() {
  local tmp rc=0
  tmp="$(mktemp -d)"
  # POSITIVE control: a bare function type must be seen.
  printf 'fn deriv(f: fn(f64) -> f64, x: f64) -> f64 with Div { 0.0 }\n' > "$tmp/pos.sio"
  if bare_hits_of "$tmp/pos.sio" | grep -q .; then
    echo "  ok   POSITIVO: tipo-funcao nu e detectado"
  else echo "  FALHA POSITIVO: nao detectou um tipo-funcao nu"; rc=1; fi
  # NEGATIVE control 1: a function DECLARATION must not be counted as a type.
  printf 'fn soma(a: i64, b: i64) -> i64 { a + b }\n' > "$tmp/neg1.sio"
  if strip_noise "$tmp/neg1.sio" | grep -oE "${PAT_TYPE}" | grep -q .; then
    echo "  FALHA NEGATIVO 1: contou uma DECLARACAO como tipo-funcao"; rc=1
  else echo "  ok   NEGATIVO 1: declaracao nao conta como tipo"; fi
  # NEGATIVE control 2: a function type inside a comment must not be counted.
  printf '// takes fn(f64) -> f64 as the kernel\nfn k(x: i64) -> i64 { x }\n' > "$tmp/neg2.sio"
  if strip_noise "$tmp/neg2.sio" | grep -oE "${PAT_TYPE}" | grep -q .; then
    echo "  FALHA NEGATIVO 2: contou um tipo-funcao dentro de comentario"; rc=1
  else echo "  ok   NEGATIVO 2: comentario nao conta"; fi
  # NEGATIVE control 3: a function type that DOES carry effects must not be
  # reported as bare — otherwise the ratchet can never be lowered.
  printf 'fn m(f: fn(f64) -> f64 with Div, x: f64) -> f64 { 0.0 }\n' > "$tmp/neg3.sio"
  if bare_hits_of "$tmp/neg3.sio" | grep -q .; then
    echo "  FALHA NEGATIVO 3: tipo-funcao COM efeitos contado como nu"; rc=1
  else echo "  ok   NEGATIVO 3: tipo com efeitos nao conta como nu"; fi
  # NEGATIVE control 4: a function type in RETURN position must not be counted,
  # because the `with` on that line belongs to the enclosing declaration.
  printf 'fn select_op(w: i64) -> fn(i64) -> i64 with Mut, Panic { f }\n' > "$tmp/neg4.sio"
  if strip_noise "$tmp/neg4.sio" | grep -oE "${PAT_TYPE}" | grep -q .; then
    echo "  FALHA NEGATIVO 4: contou um tipo-funcao em posicao de RETORNO"; rc=1
  else echo "  ok   NEGATIVO 4: posicao de retorno nao conta"; fi
  # NEGATIVE control 5 (#2570): a TUPLE-returning fn-type parameter that DOES
  # carry effects must not be reported as bare either -- the tuple's own
  # internal comma used to truncate the capture before it ever reached
  # `with`, so an effect-bearing tuple-returning fn-type had no way to be
  # recognized as non-bare at all. This is the control that would have
  # caught it.
  printf 'fn use_it(f: fn() -> ([f64; 2], i64) with Mut, Panic) -> f64 with Mut, Panic { 0.0 }\n' > "$tmp/neg5.sio"
  if bare_hits_of "$tmp/neg5.sio" | grep -q .; then
    echo "  FALHA NEGATIVO 5: tipo-funcao com retorno tupla e efeitos contado como nu"; rc=1
  else echo "  ok   NEGATIVO 5: tipo com retorno tupla e efeitos nao conta como nu"; fi
  # NEGATIVE control 6 (#2570): a NESTED-tuple-returning fn-type parameter
  # that DOES carry effects must not be reported as bare either -- tuple
  # elements recursively call parse_type (self-hosted/parser/types.sio:
  # 865-889), so `((i64, i64), f64)` is equally valid source, and the
  # one-level-paren ERE this gate used to have truncated at the first inner
  # `)` just as it once truncated at the first inner `,`. This is the
  # control that would have caught it; only the balanced-delimiter scan
  # (scan_tail, not any fixed nesting-depth ERE) can pass it for every depth.
  printf 'fn use_it(f: fn() -> ((i64, i64), f64) with Mut, Panic) -> f64 with Mut, Panic { 0.0 }\n' > "$tmp/neg6.sio"
  if bare_hits_of "$tmp/neg6.sio" | grep -q .; then
    echo "  FALHA NEGATIVO 6: tipo-funcao com retorno tupla ANINHADA e efeitos contado como nu"; rc=1
  else echo "  ok   NEGATIVO 6: tipo com retorno tupla aninhada e efeitos nao conta como nu"; fi
  # NEGATIVE control 7 (#2570): a multi-argument GENERIC return type that
  # DOES carry effects must not be reported as bare either -- tuple-typed
  # generics recursively call parse_type_args
  # (self-hosted/parser/types.sio:307-309), so `Result<i64, Error>` is
  # equally valid source, and depth tracking that only knew about `(`/`)`
  # truncated at the generic's own internal comma, inside `<...>`, before
  # ever reaching `with`. This is the control that would have caught it.
  printf 'fn use_it(f: fn() -> Result<i64, Error> with IO) -> f64 with Mut, Panic { 0.0 }\n' > "$tmp/neg7.sio"
  if bare_hits_of "$tmp/neg7.sio" | grep -q .; then
    echo "  FALHA NEGATIVO 7: tipo-funcao com retorno generico e efeitos contado como nu"; rc=1
  else echo "  ok   NEGATIVO 7: tipo com retorno generico e efeitos nao conta como nu"; fi
  # POSITIVE control 8 (#2570): a function type whose RETURN is itself a
  # function type carrying the only with-clause in the chain must still be
  # reported as bare -- parse_fn_type (self-hosted/parser/types.sio:908-966)
  # parses the return type (recursing into the nested `fn() -> i64`) BEFORE
  # checking for its own trailing `with`, so `with IO` in
  # `fn() -> fn() -> i64 with IO` is consumed by the INNER fn-type's own
  # with-check, and the OUTER `fn() -> ...` never sees a with-clause at all.
  # The old check (`hit !~ withpat`, "does the tail contain 'with' ANYWHERE")
  # saw "with IO" and wrongly called the whole thing non-bare, undercounting
  # a real bare outer function type. This is the control that would have
  # caught it.
  printf 'fn use_it(f: fn() -> fn() -> i64 with IO) -> f64 with Mut, Panic { 0.0 }\n' > "$tmp/pos8.sio"
  if bare_hits_of "$tmp/pos8.sio" | grep -q .; then
    echo "  ok   POSITIVO 8: tipo-funcao externo com retorno-de-funcao-com-efeitos e nu"
  else echo "  FALHA POSITIVO 8: tipo-funcao externo nu nao detectado (efeito do retorno atribuido ao externo)"; rc=1; fi
  # NEGATIVE control 8 (#2570): companion to POSITIVE 8 -- the grammar also
  # allows STACKED with-clauses, one per nesting level, consumed
  # innermost-first as parse_fn_type's recursion unwinds. With two
  # consecutive with-clauses for two layers, the OUTER layer DOES get its
  # own ("with Panic", the second one) and must not be reported as bare.
  # Without this control, "always call it bare whenever a nested fn( is
  # present" would pass POSITIVE 8 while still being wrong.
  printf 'fn use_it(f: fn() -> fn() -> i64 with IO with Panic) -> f64 { 0.0 }\n' > "$tmp/neg8.sio"
  if bare_hits_of "$tmp/neg8.sio" | grep -q .; then
    echo "  FALHA NEGATIVO 8: tipo-funcao externo COM efeito proprio (with empilhado) contado como nu"; rc=1
  else echo "  ok   NEGATIVO 8: tipo-funcao externo com with empilhado nao conta como nu"; fi
  # NEGATIVE control 9 (#2570): a nested function type INSIDE A TUPLE, where
  # the nested type's own "->" arrow sits at nonzero depth (inside the
  # tuple's paren). scan_tail used to treat any bare ">" as a generic closer
  # regardless of context, so the inner arrow's own ">" dropped depth back to
  # the tuple's enclosing level early, and the tuple's internal comma (still
  # meant to be protected by the tuple's still-open paren) then terminated
  # the scan before the OUTER "with Mut" was ever seen -- an effect-bearing
  # outer type falsely reported as bare. Fixed (then, and still, after the
  # stack-based rewrite below) by not letting a ">" that does not correspond
  # to an actually-open "<" close anything.
  printf 'fn use_it(f: fn() -> (fn() -> i64 with IO, f64) with Mut) -> f64 { 0.0 }\n' > "$tmp/neg9.sio"
  if bare_hits_of "$tmp/neg9.sio" | grep -q .; then
    echo "  FALHA NEGATIVO 9: tipo-funcao aninhada em tupla com efeito proprio contada como nu"; rc=1
  else echo "  ok   NEGATIVO 9: tipo-funcao aninhada em tupla com efeito proprio nao conta como nu"; fi
  # NEGATIVE control 10 (#2570): a REFINEMENT return type INSIDE A TUPLE,
  # where the refinement's own predicate uses ">" as an actual comparison
  # (`{x: i64 | x > 0}`), not an arrow. The single-character "preceded by -"
  # heuristic control 9's fix relied on does not cover this: the comparison
  # ">" is not preceded by "-" either, so it was STILL misread as a generic
  # close, dropping depth from the tuple's paren back to 0 one character
  # early (at the refinement's own closing "}", which then ALSO closed
  # nothing since depth was already wrongly at 0) -- again letting the
  # tuple's internal comma terminate the scan before the outer "with Mut".
  # This is the control that forced the fix from a single-character lookback
  # to genuine stack-based delimiter matching (scan_tail's `stack[depth]`):
  # a ">" only closes something when the top of the stack is actually "<".
  printf 'fn use_it(f: fn() -> ({x: i64 | x > 0}, f64) with Mut) -> f64 { 0.0 }\n' > "$tmp/neg10.sio"
  if bare_hits_of "$tmp/neg10.sio" | grep -q .; then
    echo "  FALHA NEGATIVO 10: tipo-funcao com refinamento em tupla e efeito proprio contada como nu"; rc=1
  else echo "  ok   NEGATIVO 10: tipo-funcao com refinamento em tupla e efeito proprio nao conta como nu"; fi
  # POSITIVE control 11 (#2570): the mirror of control 10 -- a refinement's
  # own LESS-THAN comparison (`{x: i64 | x < 0}`), not a generic open. Before
  # this fix, ANY bare "<" was pushed onto the stack unconditionally, so this
  # comparison's "<" landed on top of the stack right where the refinement's
  # own "{" should be; when the refinement's real "}" arrived, it no longer
  # matched the (wrong) top of the stack and neither delimiter ever popped --
  # depth never returned to 0 for the rest of the line, so the scan ran past
  # its real terminator and could consume an unrelated ENCLOSING
  # declaration's own with-clause from later on the same line, reporting a
  # genuinely bare parameter type as non-bare (undercounting a real
  # violation -- the opposite direction from every earlier finding here, but
  # the same root cause: an opening delimiter pushed without checking
  # whether it is really an opener in context). This is the control that
  # would have caught it.
  printf 'fn use_it(f: fn() -> ({x: i64 | x < 0}, f64)) -> f64 with IO { 0.0 }\n' > "$tmp/pos11.sio"
  if bare_hits_of "$tmp/pos11.sio" | grep -q .; then
    echo "  ok   POSITIVO 11: tipo-funcao com refinamento (comparacao <) em tupla e nu"
  else echo "  FALHA POSITIVO 11: tipo-funcao nu nao detectado (< de comparacao tratado como generico)"; rc=1; fi
  # NEGATIVE control 11 (#2570): companion to POSITIVE 11 -- the SAME
  # refinement-with-less-than shape, but this time the fn-type parameter
  # DOES carry its own effects. Without this control, "never push '<' at
  # all" would also pass POSITIVE 11 while breaking every actual generic
  # (NEGATIVO 7) at the same time.
  printf 'fn use_it(f: fn() -> ({x: i64 | x < 0}, f64) with Mut) -> f64 { 0.0 }\n' > "$tmp/neg11.sio"
  if bare_hits_of "$tmp/neg11.sio" | grep -q .; then
    echo "  FALHA NEGATIVO 11: tipo-funcao com refinamento (comparacao <) e efeito proprio contada como nu"; rc=1
  else echo "  ok   NEGATIVO 11: tipo-funcao com refinamento (comparacao <) e efeito proprio nao conta como nu"; fi
  # POSITIVE control 12 (#2570): a NESTED fn-type PARAMETER (not a nested
  # RETURN type -- the earlier findings), pinning the exact reported live
  # case (tools/test-framework/src/lib.sio:390): `register_test: fn(...,
  # fn() -> TestResult, ...)`. PAT_TYPE's own `fn\([^)]*\)` stopped at the
  # FIRST `)` -- the inner fn()'s own closing paren -- so this used to
  # produce one garbled hit spanning neither the real outer type (which has
  # no arrow at all, so is out of this gate's scope the same way any
  # arrow-less parameter type already is) nor the real inner one, and the
  # inner `fn() -> TestResult` was never visited on its own terms at all.
  # Confirms classify_fn_type_at's recursion into the parameter list finds
  # it independently.
  printf 'fn use_it(register_test: fn(i64, string, fn() -> TestResult, TestMetadata)) -> f64 { 0.0 }\n' > "$tmp/pos12.sio"
  if bare_hits_of "$tmp/pos12.sio" | grep -q "TestResult"; then
    echo "  ok   POSITIVO 12: parametro fn-type aninhado (nao retorno) e nu"
  else echo "  FALHA POSITIVO 12: tipo-funcao aninhado como PARAMETRO nao detectado"; rc=1; fi
  # NEGATIVE control 12 (#2570): companion to POSITIVE 12 -- the same nested
  # PARAMETER shape, but the nested type carries its own effects clause.
  # Without this control, "count every entry inside a parameter list as
  # bare, unconditionally" would also pass POSITIVE 12 while never being
  # able to lower the ratchet by adding effects to a nested parameter.
  printf 'fn use_it(register_test: fn(i64, string, fn() -> TestResult with IO, TestMetadata)) -> f64 { 0.0 }\n' > "$tmp/neg12.sio"
  if bare_hits_of "$tmp/neg12.sio" | grep -q "TestResult"; then
    echo "  FALHA NEGATIVO 12: parametro fn-type aninhado COM efeito proprio contado como nu"; rc=1
  else echo "  ok   NEGATIVO 12: parametro fn-type aninhado com efeito proprio nao conta como nu"; fi
  # POSITIVE control 13 (#2570): `fn_layers - with_n` is a COUNT of bare
  # layers, not a yes/no question -- `fn() -> fn() -> i64` (two layers, no
  # with-clauses at all) is TWO distinct bare function types, but the gate
  # used to print the hit at most ONCE per match regardless of how many
  # layers were actually bare, so adding a third bare returned-function
  # layer to an existing bare annotation could never raise the ratchet.
  # Pins the exact COUNT (2), not just presence, matching Copilot's report.
  printf 'fn use_it(f: fn() -> fn() -> i64) -> f64 { 0.0 }\n' > "$tmp/pos13.sio"
  n13=$(bare_hits_of "$tmp/pos13.sio" | wc -l | tr -d ' ')
  if [ "$n13" = "2" ]; then
    echo "  ok   POSITIVO 13: duas camadas nuas produzem dois hits (nao um)"
  else echo "  FALHA POSITIVO 13: esperava 2 hits para duas camadas nuas, obteve $n13"; rc=1; fi
  # NEGATIVE control 13 (#2570): companion to POSITIVE 13 -- three levels
  # deep, no with-clauses anywhere, must produce exactly THREE hits. Without
  # this control, "always print exactly 2 regardless of fn_layers" would
  # also pass POSITIVE 13.
  printf 'fn use_it(f: fn() -> fn() -> fn() -> i64) -> f64 { 0.0 }\n' > "$tmp/neg13.sio"
  n13b=$(bare_hits_of "$tmp/neg13.sio" | wc -l | tr -d ' ')
  if [ "$n13b" = "3" ]; then
    echo "  ok   NEGATIVO 13: tres camadas nuas produzem tres hits"
  else echo "  FALHA NEGATIVO 13: esperava 3 hits para tres camadas nuas, obteve $n13b"; rc=1; fi
  # POSITIVE control 14 (#2570): a WHITESPACE-FREE refinement comparison
  # (`{x:i64|x<0}`, no spaces anywhere). Sounio tokenizes refinement
  # operators independently of whitespace
  # (self-hosted/parser/types.sio:697-705), so this is equally valid source
  # to POSITIVO 11's spaced version -- but the old "push '<' only when
  # preceded by an identifier character" heuristic saw "x" directly before
  # "<" here too (spacing was never actually what made POSITIVO 11 safe;
  # the heuristic just happened to also reject the unspaced form as a side
  # effect of requiring a space specifically). Now disambiguated by
  # predicate context (pred[depth]) instead of spacing at all, so this must
  # pass identically to POSITIVO 11.
  printf 'fn use_it(f: fn() -> ({x:i64|x<0}, f64)) -> f64 with IO { 0.0 }\n' > "$tmp/pos14.sio"
  if bare_hits_of "$tmp/pos14.sio" | grep -q .; then
    echo "  ok   POSITIVO 14: refinamento sem espacos (comparacao <) e nu"
  else echo "  FALHA POSITIVO 14: refinamento sem espacos nao detectado como nu"; rc=1; fi
  # NEGATIVE control 14 (#2570): companion -- same whitespace-free
  # refinement, but the fn-type parameter DOES carry its own effects.
  printf 'fn use_it(f: fn() -> ({x:i64|x<0}, f64) with Mut) -> f64 { 0.0 }\n' > "$tmp/neg14.sio"
  if bare_hits_of "$tmp/neg14.sio" | grep -q .; then
    echo "  FALHA NEGATIVO 14: refinamento sem espacos e efeito proprio contado como nu"; rc=1
  else echo "  ok   NEGATIVO 14: refinamento sem espacos e efeito proprio nao conta como nu"; fi
  # POSITIVE control 15 (#2570): a function type whose OWN parameter list
  # spans multiple physical lines (`f: fn(` / `i64` / `) -> i64`). This
  # scanner used to reset `line = $0` fresh every record, so
  # match_close_paren ran off the end of the FIRST line without ever
  # finding the closing paren, and the type was silently treated as
  # arrow-less (out of scope) rather than the bare violation it actually
  # is -- a genuinely bare multiline function type could bypass the ratchet
  # entirely. Pins that the deferred cross-line matching (the `pending`
  # carry-over) resolves it.
  printf 'fn use_it(f: fn(\ni64\n) -> i64) -> f64 { 0.0 }\n' > "$tmp/pos15.sio"
  if bare_hits_of "$tmp/pos15.sio" | grep -q .; then
    echo "  ok   POSITIVO 15: tipo-funcao multilinha e nu"
  else echo "  FALHA POSITIVO 15: tipo-funcao multilinha nao detectado como nu"; rc=1; fi
  # NEGATIVE control 15 (#2570): companion -- same multiline shape, but the
  # type DOES carry its own effects, also spread across the lines.
  printf 'fn use_it(f: fn(\ni64\n) -> i64 with IO) -> f64 { 0.0 }\n' > "$tmp/neg15.sio"
  if bare_hits_of "$tmp/neg15.sio" | grep -q .; then
    echo "  FALHA NEGATIVO 15: tipo-funcao multilinha com efeito proprio contado como nu"; rc=1
  else echo "  ok   NEGATIVO 15: tipo-funcao multilinha com efeito proprio nao conta como nu"; fi
  # POSITIVE control 16 (#2570): a nested fn-type PARAMETER wrapped inside a
  # TUPLE (`f: fn((fn() -> i64, i64)) -> i64`) -- the entry text is
  # "(fn() -> i64, i64)", which does not itself START with "fn(", so the old
  # entry-walker (checking only the entry's first token) missed the inner
  # type entirely. Pins the exact COUNT: TWO distinct bare function types
  # (the outer AND the inner), not one.
  printf 'fn use_it(f: fn((fn() -> i64, i64)) -> i64) -> f64 { 0.0 }\n' > "$tmp/pos16.sio"
  n16=$(bare_hits_of "$tmp/pos16.sio" | wc -l | tr -d ' ')
  if [ "$n16" = "2" ]; then
    echo "  ok   POSITIVO 16: tipo-funcao aninhado em tupla (wrapper) produz 2 hits"
  else echo "  FALHA POSITIVO 16: esperava 2 hits (externo + aninhado em tupla), obteve $n16"; rc=1; fi
  # NEGATIVE control 16 (#2570): companion -- same wrapped-tuple shape, but
  # the INNER type carries its own effects. Only the outer should count (1
  # hit), pinning that the recursion resolves the inner type's own
  # arrow/with independently, not by inheriting the outer's classification.
  printf 'fn use_it(f: fn((fn() -> i64 with IO, i64)) -> i64) -> f64 { 0.0 }\n' > "$tmp/neg16.sio"
  n16b=$(bare_hits_of "$tmp/neg16.sio" | wc -l | tr -d ' ')
  if [ "$n16b" = "1" ]; then
    echo "  ok   NEGATIVO 16: apenas o tipo externo conta quando o aninhado tem efeito proprio"
  else echo "  FALHA NEGATIVO 16: esperava 1 hit (so o externo), obteve $n16b"; rc=1; fi
  # POSITIVE control 17 (#2570): same hole, wrapped in an ARRAY
  # (`[fn() -> i64; 3]`) instead of a tuple.
  printf 'fn use_it(f: fn([fn() -> i64; 3]) -> i64) -> f64 { 0.0 }\n' > "$tmp/pos17.sio"
  n17=$(bare_hits_of "$tmp/pos17.sio" | wc -l | tr -d ' ')
  if [ "$n17" = "2" ]; then
    echo "  ok   POSITIVO 17: tipo-funcao aninhado em array (wrapper) produz 2 hits"
  else echo "  FALHA POSITIVO 17: esperava 2 hits (externo + aninhado em array), obteve $n17"; rc=1; fi
  # POSITIVE control 18 (#2570): same hole, wrapped in a GENERIC
  # (`Vec<fn() -> i64>`). This is also the control that would have caught a
  # SEPARATE bug found while building this fix: the nested type's own return
  # arrow (`-> i64`) sits at depth 1 relative to the OUTER generic's "<" (a
  # generic wrapping a fn-type puts an arrow INSIDE an open "<...>" for the
  # first time anywhere in this corpus's test shapes) -- treating that
  # arrow's ">" as closing the generic (matching stack[depth]=="<" the same
  # way a REAL generic-closing ">" would) closed "Vec<" one character early,
  # at the wrong ">". Fixed by checking "preceded by -" FIRST, before the
  # stack-match, so an arrow's ">" is inert regardless of stack state.
  printf 'fn use_it(f: fn(Vec<fn() -> i64>) -> i64) -> f64 { 0.0 }\n' > "$tmp/pos18.sio"
  n18=$(bare_hits_of "$tmp/pos18.sio" | wc -l | tr -d ' ')
  if [ "$n18" = "2" ]; then
    echo "  ok   POSITIVO 18: tipo-funcao aninhado em generico (wrapper) produz 2 hits"
  else echo "  FALHA POSITIVO 18: esperava 2 hits (externo + aninhado em generico), obteve $n18"; rc=1; fi
  # POSITIVE control 19 (#2570): the arrow on its own line, AFTER the
  # parameter list's closing paren (`f: fn()` on one physical line, `-> i64`
  # on the next). The multiline fix two commits ago only deferred on
  # close_pos (the parameter list itself spanning lines); once close_pos
  # resolved within bounds, the arrow-check ran immediately against
  # whatever came right after on the SAME accumulated buffer, and if
  # nothing but trailing whitespace was there yet, concluded "no arrow" --
  # arrow-less, out of scope -- instead of "not yet known, might still be on
  # a line not read yet". A genuinely bare type formatted this way bypassed
  # the ratchet.
  printf 'fn use_it(f: fn()\n -> i64) -> f64 { 0.0 }\n' > "$tmp/pos19.sio"
  if bare_hits_of "$tmp/pos19.sio" | grep -q .; then
    echo "  ok   POSITIVO 19: seta em linha propria apos o fecha-parenteses e nu"
  else echo "  FALHA POSITIVO 19: seta em linha propria nao detectada como nu"; rc=1; fi
  # NEGATIVE control 19 (#2570): companion -- same split, but the type
  # carries its own effects, also split across the arrow-and-after line.
  printf 'fn use_it(f: fn()\n -> i64 with IO) -> f64 { 0.0 }\n' > "$tmp/neg19.sio"
  if bare_hits_of "$tmp/neg19.sio" | grep -q .; then
    echo "  FALHA NEGATIVO 19: seta em linha propria com efeito proprio contada como nu"; rc=1
  else echo "  ok   NEGATIVO 19: seta em linha propria com efeito proprio nao conta como nu"; fi
  # POSITIVE control 20 (#2570): whitespace between "fn" and "(" itself
  # (`f: fn (i64) -> i64`). self-hosted/parser/types.sio's own lexer skips
  # whitespace before parse_fn_type expects "(", so this is equally valid
  # source; every "fn(" search in this scanner previously required the two
  # characters adjacent with nothing between them.
  printf 'fn use_it(f: fn (i64) -> i64) -> f64 { 0.0 }\n' > "$tmp/pos20.sio"
  if bare_hits_of "$tmp/pos20.sio" | grep -q .; then
    echo "  ok   POSITIVO 20: espaco entre fn e ( e nu"
  else echo "  FALHA POSITIVO 20: espaco entre fn e ( nao detectado como nu"; rc=1; fi
  # NEGATIVE control 20 (#2570): companion -- same spacing, with effects.
  printf 'fn use_it(f: fn (i64) -> i64 with IO) -> f64 { 0.0 }\n' > "$tmp/neg20.sio"
  if bare_hits_of "$tmp/neg20.sio" | grep -q .; then
    echo "  FALHA NEGATIVO 20: espaco entre fn e ( com efeito proprio contado como nu"; rc=1
  else echo "  ok   NEGATIVO 20: espaco entre fn e ( com efeito proprio nao conta como nu"; fi
  # POSITIVE control 21 (#2570): a nested fn-type parameter (per POSITIVO 16)
  # where the NESTED "fn" ALSO has whitespace before its own "(" -- pins
  # that scan_entry_for_fn_types' widened search composes correctly with
  # the wrapped-type recursion from two commits ago, not just the top-level
  # anchor.
  printf 'fn use_it(f: fn((fn () -> i64, i64)) -> i64) -> f64 { 0.0 }\n' > "$tmp/pos21.sio"
  n21=$(bare_hits_of "$tmp/pos21.sio" | wc -l | tr -d ' ')
  if [ "$n21" = "2" ]; then
    echo "  ok   POSITIVO 21: fn aninhado com espaco antes do ( produz 2 hits"
  else echo "  FALHA POSITIVO 21: esperava 2 hits, obteve $n21"; rc=1; fi
  # NEGATIVE control 22 (#2570): a chained return type used AS A PARAMETER
  # (`f: fn(fn() -> fn() -> i64) -> i64`). classify_fn_type_at, called on the
  # entry's first "fn(", already walks and counts the whole 2-layer chain
  # internally via fn_layers/with_n (both layers, correctly). Advancing
  # scan_entry_for_fn_types past just the matched "fn(" token instead of past
  # what classify_fn_type_at itself already consumed found the SAME chain's
  # second "fn(" again as an apparently-independent sibling and classified it
  # a second time: four hits (outer=1, chain=2, spurious re-classification=1)
  # against the correct three (outer=1, chain=2) -- which could spuriously
  # raise the frozen count and block a valid, unrelated change. Named
  # NEGATIVE (not POSITIVE) because the assertion here is an exact hit
  # COUNT, matching the convention POSITIVO/NEGATIVO 13 already established
  # for that shape of check.
  printf 'fn use_it(f: fn(fn() -> fn() -> i64) -> i64) -> f64 { 0.0 }\n' > "$tmp/neg22.sio"
  n22=$(bare_hits_of "$tmp/neg22.sio" | wc -l | tr -d ' ')
  if [ "$n22" = "3" ]; then
    echo "  ok   NEGATIVO 22: cadeia aninhada como parametro produz 3 hits (nao 4)"
  else echo "  FALHA NEGATIVO 22: esperava 3 hits (externo + 2 camadas), obteve $n22"; rc=1; fi
  # POSITIVE control 22 (#2570): companion, confirming the fix does not
  # over-correct into UNDER-counting -- TWO genuinely DIFFERENT sibling
  # fn-type elements inside one wrapped tuple entry
  # (`(fn() -> i64, fn() -> f64)`) must still both be found independently;
  # only advancing PAST what classify_fn_type_at already consumed for the
  # FIRST one, not skipping the rest of the entry outright.
  printf 'fn use_it(f: fn((fn() -> i64, fn() -> f64)) -> i64) -> f64 { 0.0 }\n' > "$tmp/pos22.sio"
  n22b=$(bare_hits_of "$tmp/pos22.sio" | wc -l | tr -d ' ')
  if [ "$n22b" = "3" ]; then
    echo "  ok   POSITIVO 22: dois irmaos genuinos em uma tupla produzem 3 hits"
  else echo "  FALHA POSITIVO 22: esperava 3 hits (externo + 2 irmaos), obteve $n22b"; rc=1; fi
  # POSITIVE control 23 (#2570): the arrow itself resolves within bounds
  # (`f: fn() ->` on one physical line), but the RETURN TYPE text -- not
  # just the arrow -- starts on the next line (`(i64, i64) with Mut`).
  # self-hosted/parser/types.sio:935-951 parses the return type and effects
  # as tokens across whitespace, so this is equally valid source. Before
  # this fix, scan_tail ran off the end of line 1 immediately (nothing at
  # all followed the arrow there) and the type was finalized as bare from
  # an empty hit_tail, never reaching line 2's "with Mut" at all -- a false
  # ratchet violation, not an undercount.
  printf 'fn use_it(f: fn() ->\n(i64, i64) with Mut) -> f64 { 0.0 }\n' > "$tmp/pos23.sio"
  if bare_hits_of "$tmp/pos23.sio" | grep -q .; then
    echo "  FALHA POSITIVO 23: tipo com retorno multilinha e efeito proprio contado como nu"; rc=1
  else echo "  ok   POSITIVO 23: tipo com retorno multilinha e efeito proprio nao conta como nu"; fi
  # NEGATIVE control 23 (#2570): companion -- same split, but genuinely bare
  # (no with-clause anywhere), pinning that this defers to find the true
  # end rather than becoming unconditionally non-bare.
  printf 'fn use_it(f: fn() ->\n(i64, i64)) -> f64 { 0.0 }\n' > "$tmp/neg23.sio"
  if bare_hits_of "$tmp/neg23.sio" | grep -q .; then
    echo "  ok   NEGATIVO 23: tipo com retorno multilinha genuinamente nu e detectado"
  else echo "  FALHA NEGATIVO 23: tipo com retorno multilinha nu nao detectado"; rc=1; fi
  # Regression guard, same shape as the earlier ffi_demo.sio control: real,
  # substantial content that simply never closes must still NOT defer.
  printf 'let sin_fn: fn(c_double) -> c_double = unsafe {\n' > "$tmp/ffi23.sio"
  if bare_hits_of "$tmp/ffi23.sio" | grep -q .; then
    echo "  ok   REGRESSAO 23: bloco unsafe nao fechado continua nao-diferido (fim de linha 1)"
  else echo "  FALHA REGRESSAO 23: bloco unsafe nao fechado parou de ser detectado"; rc=1; fi
  # NEGATIVE control 24 (#2570): whitespace between a generic type name and
  # its "<" is lexically insignificant (self-hosted/parser/types.sio skips
  # it), so `Result <i64, Error>` is equally valid source as `Result<i64,
  # Error>`. Before this fix, prevc held " " (not an identifier char) right
  # at the "<", so it was never pushed as a generic-open -- the later ">"
  # then hit the depth-0 case and was treated as "we have left this scope",
  # so the generic comma wrongly terminated the tail before "with IO" and
  # this was miscounted as bare.
  printf 'fn use_it(f: fn() -> Result <i64, Error> with IO) -> f64 { 0.0 }\n' > "$tmp/neg24.sio"
  if bare_hits_of "$tmp/neg24.sio" | grep -q .; then
    echo "  FALHA NEGATIVO 24: tipo com retorno generico espacado e efeito proprio contado como nu"; rc=1
  else echo "  ok   NEGATIVO 24: tipo com retorno generico espacado e efeito proprio nao conta como nu"; fi
  # POSITIVE control 24: companion -- same spaced generic, but genuinely bare
  # (no with-clause), pinning that the fix does not just unconditionally
  # suppress every spaced-generic return type.
  printf 'fn use_it(f: fn() -> Result <i64, Error>) -> f64 { 0.0 }\n' > "$tmp/pos24.sio"
  if bare_hits_of "$tmp/pos24.sio" | grep -q .; then
    echo "  ok   POSITIVO 24: tipo com retorno generico espacado e nu e detectado"
  else echo "  FALHA POSITIVO 24: tipo com retorno generico espacado nu nao detectado"; rc=1; fi
  # POSITIVE control 25 (#2570): a TOP-LEVEL parameter whose declared type
  # WRAPS a function type one level out in a TUPLE -- `f: (fn() -> i64,
  # i64)` -- was invisible to the old scanner entirely: the token right
  # after ":" is "(" , not "fn", so classify_fn_type_at was never even
  # reached for it.
  printf 'fn use_it(f: (fn() -> i64, i64)) -> f64 { 0.0 }\n' > "$tmp/pos25.sio"
  if bare_hits_of "$tmp/pos25.sio" | grep -q .; then
    echo "  ok   POSITIVO 25: tipo-funcao envolto em tupla no nivel superior e nu e detectado"
  else echo "  FALHA POSITIVO 25: tipo-funcao envolto em tupla no nivel superior nu nao detectado"; rc=1; fi
  # NEGATIVE control 25: companion -- same tuple-wrapped shape, but the
  # nested fn-type carries its own effects clause, pinning that the fix does
  # not just unconditionally flag every tuple-wrapped fn-type.
  printf 'fn use_it(f: (fn() -> i64 with Div, i64)) -> f64 { 0.0 }\n' > "$tmp/neg25.sio"
  if bare_hits_of "$tmp/neg25.sio" | grep -q .; then
    echo "  FALHA NEGATIVO 25: tipo-funcao envolto em tupla com efeito proprio contado como nu"; rc=1
  else echo "  ok   NEGATIVO 25: tipo-funcao envolto em tupla com efeito proprio nao conta como nu"; fi
  # POSITIVE control 26: same hole, wrapped in an ARRAY (`[fn() -> i64; 3]`)
  # at the top level instead of a tuple.
  printf 'fn use_it(f: [fn() -> i64; 3]) -> f64 { 0.0 }\n' > "$tmp/pos26.sio"
  if bare_hits_of "$tmp/pos26.sio" | grep -q .; then
    echo "  ok   POSITIVO 26: tipo-funcao envolto em array no nivel superior e nu e detectado"
  else echo "  FALHA POSITIVO 26: tipo-funcao envolto em array no nivel superior nu nao detectado"; rc=1; fi
  # NEGATIVE control 26: companion for the array-wrapped shape.
  printf 'fn use_it(f: [fn() -> i64 with Div; 3]) -> f64 { 0.0 }\n' > "$tmp/neg26.sio"
  if bare_hits_of "$tmp/neg26.sio" | grep -q .; then
    echo "  FALHA NEGATIVO 26: tipo-funcao envolto em array com efeito proprio contado como nu"; rc=1
  else echo "  ok   NEGATIVO 26: tipo-funcao envolto em array com efeito proprio nao conta como nu"; fi
  # NEGATIVE control 27 (#2570): a NESTED fn-type return whose OWN effects
  # list has MULTIPLE comma-separated effects, followed by the outer
  # fn-type's own effects list -- `fn() -> fn() -> i64 with IO, Mut with
  # Panic`. scan_tail used to terminate at the first depth-0 comma
  # unconditionally, stopping right after "IO" and never reaching "Mut with
  # Panic": count_with_clauses then saw only one "with" for what is
  # genuinely two effectful layers (inner "with IO, Mut", outer "with
  # Panic"), misreporting a bare hit. Neither layer is actually bare.
  printf 'fn use_it(f: fn() -> fn() -> i64 with IO, Mut with Panic) -> f64 { 0.0 }\n' > "$tmp/neg27.sio"
  if bare_hits_of "$tmp/neg27.sio" | grep -q .; then
    echo "  FALHA NEGATIVO 27: retorno aninhado com lista de efeitos multipla contado como nu"; rc=1
  else echo "  ok   NEGATIVO 27: retorno aninhado com lista de efeitos multipla nao conta como nu"; fi
  # NEGATIVE control 28: companion -- a genuine SIBLING parameter after a
  # multi-effect with-clause must still correctly end the declared type
  # there (comma_starts_new_parameter's "ident:" lookahead), not swallow the
  # sibling parameter into the scan.
  printf 'fn use_it(f: fn() -> i64 with IO, Mut, x: i64) -> f64 with Div { 0.0 }\n' > "$tmp/neg28.sio"
  if bare_hits_of "$tmp/neg28.sio" | grep -q .; then
    echo "  FALHA NEGATIVO 28: parametro irmao apos lista de efeitos multipla contado como nu"; rc=1
  else echo "  ok   NEGATIVO 28: parametro irmao apos lista de efeitos multipla nao conta como nu"; fi
  # POSITIVE control 27: companion -- the same multi-effect nested shape,
  # but the OUTER layer is genuinely bare (no with-clause of its own), so
  # exactly the outer layer must still be flagged.
  printf 'fn use_it(f: fn() -> fn() -> i64 with IO, Mut) -> f64 { 0.0 }\n' > "$tmp/pos27.sio"
  n27=$(bare_hits_of "$tmp/pos27.sio" | wc -l | tr -d ' ')
  if [ "$n27" = "1" ]; then
    echo "  ok   POSITIVO 27: apenas a camada externa nua e detectada quando a interna tem lista de efeitos multipla"
  else echo "  FALHA POSITIVO 27: esperava 1 hit (so a camada externa), obteve $n27"; rc=1; fi
  # POSITIVE control 29 (#2570): a TOP-LEVEL parameter whose declared type
  # wraps a function type in a NAMED GENERIC -- `f: Vec<fn() -> i64>` --
  # was invisible to the top-level scanner: the token right after ":" is an
  # identifier ("Vec"), not "fn"/"("/"[", so this never reached
  # scan_entry_for_fn_types at all, even though that function already finds
  # a generic-wrapped fn-type once reached from WITHIN an outer fn(...)
  # parameter list (selftest 18). This is a genuinely TOP-LEVEL wrapper, not
  # nested inside another fn-type, unlike POSITIVO 18.
  printf 'fn use_it(f: Vec<fn() -> i64>) -> f64 { 0.0 }\n' > "$tmp/pos29.sio"
  if bare_hits_of "$tmp/pos29.sio" | grep -q .; then
    echo "  ok   POSITIVO 29: tipo-funcao envolto em generico nomeado no nivel superior e nu e detectado"
  else echo "  FALHA POSITIVO 29: tipo-funcao envolto em generico nomeado no nivel superior nu nao detectado"; rc=1; fi
  # NEGATIVE control 29: companion -- same top-level named-generic wrapper,
  # but the nested fn-type carries its own effects clause.
  printf 'fn use_it(f: Vec<fn() -> i64 with Div>) -> f64 { 0.0 }\n' > "$tmp/neg29.sio"
  if bare_hits_of "$tmp/neg29.sio" | grep -q .; then
    echo "  FALHA NEGATIVO 29: tipo-funcao envolto em generico nomeado com efeito proprio contado como nu"; rc=1
  else echo "  ok   NEGATIVO 29: tipo-funcao envolto em generico nomeado com efeito proprio nao conta como nu"; fi
  # POSITIVE control 30: the spaced-generic form (identifier, whitespace,
  # then "<") at the top level, mirroring the tolerance POSITIVO 24 already
  # pins for a spaced generic RETURN type.
  printf 'fn use_it(f: Vec <fn() -> i64>) -> f64 { 0.0 }\n' > "$tmp/pos30.sio"
  if bare_hits_of "$tmp/pos30.sio" | grep -q .; then
    echo "  ok   POSITIVO 30: generico nomeado espacado no nivel superior e nu e detectado"
  else echo "  FALHA POSITIVO 30: generico nomeado espacado no nivel superior nu nao detectado"; rc=1; fi
  rm -rf "$tmp"
  echo "falhas: $rc"
  return $rc
}

[ "${1:-}" = "--selftest" ] && { selftest; exit $?; }

selftest >/dev/null 2>&1 || {
  echo "ABORT: the gate's own controls fail — its number would be noise, not evidence."
  selftest
  exit 2
}

# Anti-vacuity: the sweep must see the corpus at all. If enumerate returns
# nothing because the pattern rotted or the file list came back empty, that is a
# broken instrument, not a repository with zero bare function types.
ficheiros=$(git ls-files '*.sio' | grep -vE '^(archive|bootstrap)/' | wc -l | tr -d ' ')
require_nonempty "$ficheiros" "the .sio file list came back empty"
require_min_count "$ficheiros" 500 "live .sio files"

[ "${1:-}" = "--list" ] && { enumerate | sort; exit 0; }

atual=$(enumerate | wc -l | tr -d ' ')
require_nonempty "$atual" "the bare-function-type count came back empty"
[ -f "$REF" ] || printf '%s\n' "$atual" | gate_write_artifact "$REF"
congelado=$(head -1 "$REF" | tr -d ' ')

# The count and the list must describe the same corpus. The sibling ratchet
# carried .frozen=472 against a 474-line .frozen.list for weeks, which is how a
# count gets lowered without regenerating what it summarises.
if [ -f "$REF_LIST" ]; then
  lista_n=$(wc -l < "$REF_LIST" | tr -d ' ')
  if [ "$lista_n" != "$congelado" ]; then
    echo "REFUSE: ${REF} says ${congelado} but ${REF_LIST} holds ${lista_n} lines." >&2
    echo "  They must describe the same corpus. Regenerate the list:" >&2
    echo "    bash $0 --list > ${REF_LIST}" >&2
    exit 1
  fi
fi

mkdir -p "$(dirname "$OUT")"
estado=pass; falhou=0
if [ "$atual" -gt "$congelado" ]; then
  estado=fail; falhou=1
  echo "REFUSE: bare function types rose ${congelado} -> ${atual}."
  echo "SOUNIO-SPEC-06 §6.0 rules that a function type carries the function's effects."
  echo "A new function type without a 'with' clause widens the gap to that ruling."
  # Name the sites by SET DIFFERENCE, never by `tail -n <delta>`.
  #
  # This used to print `enumerate | tail -n $(( atual - congelado ))` -- the last
  # N lines of the whole corpus in git ls-files order, which has nothing to do
  # with which sites are new. Measured 2026-08-30 on #2225: it named six sites in
  # tools/test-framework/src/lib.sio, a file that PR does not touch, while the
  # six real ones were in examples/higher_order.sio and examples/newton_root.sio.
  # A gate that reports a violation and points at the wrong file sends the reader
  # to edit innocent code; that is worse than reporting a bare count.
  if [ -f "$REF_LIST" ]; then
    echo "New sites (present now, absent from ${REF_LIST}):"
    comm -13 "$REF_LIST" <(enumerate | sort) | sed 's/^/  /'
  else
    echo "New sites: cannot say -- ${REF_LIST} is missing, so there is nothing to"
    echo "  diff against. Regenerate it with:"
    echo "    bash $0 --list > ${REF_LIST}"
    echo "  Reporting the count only, rather than guessing at filenames."
  fi
elif [ "$atual" -lt "$congelado" ]; then
  echo "OK: bare function types fell ${congelado} -> ${atual}. Lower the frozen count:"
  echo "  printf '%s\\n' ${atual} > ${REF}"
else
  echo "OK: bare function types hold at ${congelado}."
fi

cat <<JSON | gate_write_artifact "$OUT"
{
  "gate": "fn_type_effect_ratchet",
  "status": "${estado}",
  "spec_section": "SOUNIO-SPEC-06",
  "frozen": ${congelado},
  "measured": ${atual},
  "metrics": { "total": ${atual}, "passed": $(( atual - falhou )), "failed": ${falhou}, "not_run": 0 }
}
JSON
exit "${falhou}"
