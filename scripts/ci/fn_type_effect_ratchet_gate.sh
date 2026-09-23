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
# the first `,` or unbalanced `)` seen at depth 0 -- that is either the next
# parameter in the enclosing list, or the enclosing parameter list's own
# closing paren, neither of which belongs to the function type itself.
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
AWK_SCAN='
function scan_tail(line, tail_start,    i, n, c, prev, depth, stack) {
    depth = 0
    prev = ""
    n = length(line)
    i = tail_start
    while (i <= n) {
        c = substr(line, i, 1)
        if (c == "<") {
            if (prev ~ /[A-Za-z0-9_]/) {
                depth++
                stack[depth] = c
            }
        } else if (c == "(" || c == "[" || c == "{") {
            depth++
            stack[depth] = c
        } else if (c == ")") {
            if (depth == 0) { return i }
            if (stack[depth] == "(") { depth-- }
        } else if (c == ">") {
            if (depth > 0 && stack[depth] == "<") { depth-- }
        } else if (c == "]") {
            if (depth > 0 && stack[depth] == "[") { depth-- }
        } else if (c == "}") {
            if (depth > 0 && stack[depth] == "{") { depth-- }
        } else if (c == "," && depth == 0) {
            return i
        }
        prev = c
        i++
    }
    return n + 1
}
function count_fn_parens(str,    tmp) {
    tmp = " " str
    return gsub(/[^A-Za-z0-9_]fn\(/, "@", tmp)
}
function count_with_clauses(str,    tmp) {
    tmp = " " str
    return gsub(/[^A-Za-z0-9_]with[ \t]+[A-Za-z]/, "@", tmp)
}
{
    line = $0
    pos = 1
    while (pos <= length(line) && match(substr(line, pos), /'"$PAT_TYPE"'/)) {
        start = pos + RSTART - 1
        matchlen = RLENGTH
        tail_start = start + matchlen
        tail_end = scan_tail(line, tail_start)
        hit_tail = substr(line, tail_start, tail_end - tail_start)
        hit = substr(line, start, tail_end - start)
        fn_layers = 1 + count_fn_parens(hit_tail)
        with_n = count_with_clauses(hit_tail)
        if (with_n < fn_layers) { print hit }
        pos = start + matchlen
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
