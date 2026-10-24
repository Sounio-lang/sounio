<!-- docs:meta
topic_id: repo.docs.spec.2026-10.18-lexical-execution-receipt
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.2026-10.18-lexical-execution-receipt
-->

# 18. Lexical Execution Receipt

## Result and claim boundary

Chapter [01](01-lexical-syntax-and-modules.md) states `LEX-001`–`LEX-013` from
source reading at `f5b2c585`, with no compiler run. This receipt **executes**
them: 125 probe programs, each run under both engines, on `main` at `27afa340`
(2026-10-22). It does not promote any rule's standing and validates no prose of
chapter 01. It records, per rule, whether execution **confirms** it,
**contradicts** it, or finds behaviour the rule does not mention, and it records
the Madaros defects fixed by the companion change.

Engines measured:

| column | binary |
|---|---|
| committed | `bin/madaros-linux-x86_64` (from `.gz`), the ELF `bin/souc` runs when no local build exists |
| source | Madaros built from `27afa340` with `scripts/ci/build_modular_madaros.sh` (129,688,636 bytes) |
| fixed | the same build with the companion lexer change applied (§4) |
| seed | `bin/souc-lean-single-x86_64` via `SOUNIO_SOUC_ENGINE=lean_single` |

The committed and source columns agree on all 125 probes **except the character
literals**, which exist in the source and not in the committed ELF (§3.8).

Raw material: [`docs/audit/S01_LEXICAL_PROBES_2026-10-22/`](../../audit/S01_LEXICAL_PROBES_2026-10-22/README.md)
(`gen_probes.sh` writes the probes; `run_probes.sh` runs them, with
`MADAROS_ELF=` selecting a specific Madaros binary; one TSV per column). The
probes are measurements, not conformance tests: none asserts an outcome until
the rulings of [`PENDING.md` D-33](../../decisions/PENDING.md#d-33) are made.

## 1. Rule by rule

| rule | execution | finding |
|---|---|---|
| `LEX-001` frontend identity | **confirmed** | The live route is `lex_source_to_globals` up to `return tc`; everything after it in the function, and every function only that tail calls, is unreachable: at least 1,336 of the 3,261 lines in `self-hosted/lexer/` (caller search over `self-hosted/`). The two numberings of `PT_KIND_CODE` are real and collide (flat 123 is `Ident`, discriminant 123 is `Valid`); diagnostics printed both — **fixed**, §4 |
| `LEX-002` encoding and positions | **confirmed, plus one divergence** | ASCII identifiers only on Madaros. The seed lexes `ε` (U+03B5) and `⊥` (U+22A5) as tokens (`lex_all`, bytes `206 181`, `226 138 165`); Madaros emits Error tokens for them, which the `Knowledge` parameter and call-label parsers consume by name. The string scanner's line defect — **fixed**, §4 |
| `LEX-003` name limits | **confirmed** | A 300-byte identifier is two tokens on Madaros (parse error at column 265); the seed runs it |
| `LEX-004` whitespace and comments | **confirmed; contradicted on the seed** | Nesting works on Madaros. Unterminated `/*` at end of file was accepted — **fixed** (E272). The seed has **no block comments** at all (`lex_all` handles `//` only); block comments between items appear to work there only because the seed accepts junk between items (§3.10) |
| `LEX-005` keyword inventory | **confirmed** | 147 spellings returned by `keyword_lookup`. Spelling a keyword is not reserving it. 54 probes (`kw_let_*`) use a word as `let K = 1; println(K)`: 50 keyword spellings plus 4 non-keyword controls (`model`, `measured`, `valid`, `input`; the table has only the capitalised forms). Madaros accepts the 4 controls and 41 of the 50 and refuses `affine async await crate linear perform resume spawn super`; the seed accepts all 54 and **computes 0** for `affine`, `linear` and `study` |
| `LEX-006` contextual names | **confirmed, wider than stated** | Words outside `tk_is_contextual_ident`, such as `sample` and `prior`, are accepted in expression position on Madaros; the path was not traced |
| `LEX-007` punctuators, maximal munch | **confirmed; one sentence superseded** | `x<-1`, `a+-2`, `a&!b` are refused by Madaros because the lexer commits to `<-`, `+-`, `&!`; the seed computes `x<-1` and `a+-2` as intended. `<<=` is refused (silently before §4). *"A single quote currently has no scalar character-literal branch"* is no longer true on `main`: the branch exists (§3.8) |
| `LEX-008` capacity refusals | not reached | No probe reaches 16 MiB or 2,097,152 tokens; read from source only |
| `LEX-009` integers | **confirmed** | No overflow check: `9223372036854775808` → `−9223372036854775808`, `99999999999999999999` → `7766279631452241919`, `0x1FFFFFFFFFFFFFFFF` → `−1`, **on both engines** |
| `LEX-010` decimal floats | **confirmed; the seed contradicts it** | `.5` is not a float token on Madaros (refused); the seed accepts it and **computes 0** |
| `LEX-011` extended spellings | **confirmed** | `0b101`, `0o17`, `5u8`, `2.5f64` refused on both, never with a diagnostic about the literal: on Madaros `0b101` and `0o17` are parse errors at the trailing word and the suffixes are `E137` undeclared variables; on the seed all four are `E200` undefined identifiers. `0x1.8p1` = 3.0 on Madaros, refused on the seed. `500_mg` is a unit-suffixed literal on the seed only |
| `LEX-012` strings and escapes | **confirmed in every defect it names** | `\0` was the digit `0`, `\x41` was `x41`, `\q` was `q`, an unterminated string ran to end of file, raw LF did not advance the line — **all fixed**, §4. The seed decodes `\0` and `\x41` and refuses unknown escapes |
| `LEX-013` string size | **confirmed** | 400-byte literal refused with E258; the seed runs it |

## 2. Midpoint rounding (not a `LEX` rule)

`1.0000000000000001110223024625156541 > 1.0` lies above the midpoint between
1.0 and the next double, so a correctly rounded parse gives `1 + 2⁻⁵²` and
`true`. **All four columns give `false`.** The seed keeps at most 18 fraction
digits (`FL_FRAC_DIGITS < 18`, `lex_all`); Madaros's `pfl_slow` path discards
excess digits (chapter 01, `LEX-011`). See `PENDING.md` D-27.

## 3. Engine divergences, condensed

"Refused" means the program does not compile; "silent" means no diagnostic
names a line (true of every Madaros refusal marked so before §4).

| § | input | Madaros (committed / source) | seed |
|---|---|---|---|
| 3.1 | `$` between tokens | refused, silent → **E271 with location** | **skipped**; `1 $ 2` prints `1` |
| 3.1 | form feed between tokens | refused, silent → E271 | skipped |
| 3.2 | `/* … */` inside a function | comment | **not a comment**; contents are undefined identifiers |
| 3.2 | unterminated `/*` at end of file | **accepted** → **E272** | accepted (junk rule, 3.10) |
| 3.3 | 300-byte identifier | refused at column 265 | runs |
| 3.4 | `affine`, `linear`, `study` as names | refused, refused, accepted | accepted, **value 0** |
| 3.5 | `1_000_000`, `0x1_0` | 1000000, 16 | refused |
| 3.5 | out-of-range integer | **wraps** | **wraps** |
| 3.6 | `.5` | refused, silent | **0.000000** |
| 3.6 | `0x1.8p1` | 3.0 | refused |
| 3.7 | `\0`, `\x41`, `\q` | `0`, `x41`, `q` → **NUL, `A`, E274** | NUL, `A`, refused |
| 3.7 | `println("a\0b")` (after §4) | prints `a`: Madaros's string printing stops at the NUL | prints the three bytes `a`, NUL, `b` |
| 3.7 | string with newlines, then an error | error reported lines early → **correct line** | correct line |
| 3.8 | `'a'`, `'\n'` | refused (committed) / 97, 10 (source) | refused (`'` skipped) |
| 3.8 | `'\q'` | 113 (source) → **E274** | refused |
| 3.8 | `let c = 'a` (no closing quote) | accepted (source) | refused |
| 3.9 | `x<-1`, `a+-2` | refused | correct |
| 3.9 | `a ++ b` on arrays | fails in code generation (rc=12) | runs |
| 3.10 | `garbage tokens here` between functions | refused, silent | **accepted** |

**Character literals and lifetimes.** Refusing the unclosed literal in 3.8 is
not a fix but a ruling, and a blocked one: `&'static str` occurs in the standard
library, and the live branch reads it today as the character `s` followed by the
word `tatic`. The character-literal ruling must decide lifetime syntax with it
(D-33.4).

## 4. Madaros defects fixed by the companion change

The change in [PR #3016](https://github.com/Sounio-lang/sounio/pull/3016) (branch `fix/s01-lexer-defects`, **not merged when this receipt was written**) fixes every Madaros defect above
that contradicts a rule the tree already states (fail closed; report the real
location) and needs no ruling. Each row has a test that fails on the source
column and passes on the fixed column; the evidence is in the change's commit
messages.

| defect | before | after | test |
|---|---|---|---|
| unknown byte | `module failed to parse`, no location | `error[E271]: Unexpected character (byte 36)` `--> file:9:15`; `ε`, `⊥` still tolerated where the parser consumes them | `tests/ui/lexer/unknown_char_located.sio` |
| unterminated block comment | accepted; rest of file discarded | `error[E272]` at the opening `/*` | `tests/ui/lexer/unterminated_block_comment.sio` |
| unterminated string | `expected=186 actual=0` at end of file | `error[E273]` at the opening quote | `tests/ui/lexer/unterminated_string.sio` |
| unknown escape in a string | `"\q"` compiled as `"q"` | `error[E274]` naming `\q` | `tests/ui/lexer/invalid_escape.sio` |
| unknown escape in a character literal | `'\q'` = 113 | `error[E274]` | `tests/ui/lexer/invalid_char_escape.sio` |
| `\u{41}` in a string | compiled as the text `u{41}` | `error[E274]`: `\u{…}` is not decoded by the lowering, so it is refused until D-33.3 decides it | probe `str_escape_unicode` |
| `\0`, `\xHH` | the text `0`, `xHH` | the byte 0, the byte 0xHH (checked byte by byte; printing a string still stops at its first NUL, §3.7, which is a ruling under D-33.3) | `tests/run-pass/string_escape_bytes.sio` |
| line after a multi-line string | three lines early | correct line | `tests/ui/lexer/line_after_multiline_string.sio` |
| two numberings in one diagnostic | `expected=146 actual=124` | `` expected `=`, found integer literal [expected kind=146, found code=124] `` | `tests/ui/lexer/line_after_multiline_string.sio` |
| no expression can start (`let x = @y`) | `module failed to parse`, no message | `parse error: expected expression at line L:C: found `@`` | `tests/ui/lexer/invalid_char.sio` |
| `tests/ui/parser/nested_comment.sio` asserted nesting is unsupported | contradicted `LEX-004` | asserts nesting works | the fixture itself |

Blast radius, measured before the change with a C++23 mirror of the live lexer
over 9,616 `.sio` files: no source outside test fixtures has an unterminated
comment or string or an escape outside the accepted set. `\x` occurs 505 times
(ANSI escapes in `stdlib/plot` among them), which printed as literal text
before and print as the intended bytes after.

## 5. What this receipt does not claim

- That the committed ELF and the source agree in general; they agreed on these
  125 probes except character literals.
- The exactness of hexadecimal-float assembly in the parser.
- That any keyword should be reserved or any literal form should exist; those
  are D-33.
- That the seed's defects are fixed. `.5` and the three keywords are refused by
  PR #3019, unmerged when this was written (§6); skipped bytes, missing block comments and junk
  between items are open.

## 6. The seed's silent zero is wider than the lexer

`affine`, `linear`, `study` and `.5` read as 0 on the seed for one reason: the
last arm of `compile_primary` (and `compile_primary_a64`) consumes any token
that starts no expression and loads 0, without a diagnostic. Measured by
rebuilding the seed with that arm refusing every such token and compiling every
`.sio` file under `tests/`, `stdlib/` and `examples/` with both seeds
([`seed_fallback_sweep.tsv`](../../audit/S01_LEXICAL_PROBES_2026-10-22/seed_fallback_sweep.tsv)):
532 files reach the arm, and **166 of them compile today** (126 tests, 22
stdlib, 18 examples). By token: `var` 56, `]` 50, `,` 25, `)` 10, `{` 9, `=` 5,
`;` 3, `}` 2, `else` 2, `^`, `+`, `.`, `affine` 1 each.

Some of these are recovery that happens to be harmless (a trailing comma).
Others are wrong values. Measured on the seed, against Madaros as referent:

| program | seed | Madaros |
|---|---|---|
| `let var = 5` then `println(var)` | `0` | `5` |
| `println()` | `0` | empty line |
| `let b = +a` with `a = 7`, then `println(b)` | `0` | refused, located |

The first row reaches the standard library: `pb_apply2_monotone_inc_dec`
(`stdlib/epistemic/knightian.sio:307–309`) binds `let var = …` and returns
`variance: var`, so under the seed every p-box it combines has variance 0.

The same arm reached Madaros, which the seed compiles: `active_study_has_hypothesis_outcome`
(`self-hosted/check/check.sio`) binds `let study = …`, and the seed reads
`study.hypothesis_count` as 0 (reproduced in isolation), so a seed-built Madaros
could never find a pre-registered outcome there.

The companion seed change, [PR #3019](https://github.com/Sounio-lang/sounio/pull/3019),
refuses only the cases no file relies on (the three keywords and a dot followed
by a digit) and renames that local; a tree-wide sweep with it refuses nothing
else, and Madaros still builds with it. Refusing the whole arm moves 126
tests, so it is per-construct work — `var` as a name, empty argument lists,
unary `+`, character literals in array repeats — each with its own fix and
golden review. It is listed in the [backlog](../../planning/LANGUAGE_BACKLOG_2026-10-22.md) §1.
