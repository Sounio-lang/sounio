<!-- docs:meta
topic_id: repo.docs.audit.madaros-hex-float-unsigned-exponent-dispatch-2026-09-26
authority: repo_only
audience: users
last_validated: 2026-09-26
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-hex-float-unsigned-exponent-dispatch-2026-09-26
-->

# Madaros refuses a C99 hex-float with an unsigned exponent (dispatch)

**Date:** 2026-09-26
**Scope:** C99 hex-float literals whose binary exponent has no sign
(`0x1p0`, `0x1.8p0`, `0x1.0000000000000001p0`). C99 and the V0-E.5.9
exact-literal design both allow them.
**Status:** FIXED in this dispatch's PR. The owning code is the parser's
hex-float reassembly in `self-hosted/parser/parser.sio` and
`self-hosted/parser/exprs.sio`. The lexer and the literal converters are
unchanged.
**Witness:** [`tests/run-pass/f128_hex_float_unsigned_exponent.sio`](../../tests/run-pass/f128_hex_float_unsigned_exponent.sio)

## 1. Symptom

Reported on a Madaros built from `98315edcd` (`make build-madaros`, md5
`5764851f3d229372e26aac1e951c95e1`). `self-hosted/` is identical from there to
`main` at `f141ad5d9` (`git diff --stat 98315edcd f141ad5d9 -- self-hosted` is
empty).

```sounio
use math::softfloat_f128_fmt::{print_f128}
fn main() -> i32 with IO, Mut, Panic, Div {
    let a: f128 = 0x1.0000000000000001p0
    print_f128(a)
    println("")
    0
}
```

`error[E012] ... at 105..127: this type has no field named`, then
`error[E137] ... at 125..127: use of undeclared variable`. With
`0x1.0000000000000001p+0` the same program builds and prints
`1.00000000000000000005421010862427522e+0000`, which is 1 + 2^-64 exactly.

## 2. Minimal matrix: which variable matters

This matrix follows operating principle 12. Each row is a program of 5 to 7
lines that binds one literal. For f64 it compares the literal against a
decimal constant. For f128 it prints the value with `print_f128`. The
baseline is the `5764851f` ELF, run as
`ulimit -s 524288; madaros build r.sio r.elf` with
`SOUC_BIN SOUNIO_SOUC_BIN MADAROS_RAW_BIN SOUNIO_MADAROS_BIN` unset and
`SOUNIO_STDLIB_PATH` set to the worktree's `stdlib`.

| Literal | Type | Fraction | Sign | Baseline Madaros |
|---|---|---|---|---|
| `0x1p+0` | f64 | none | `+` | builds, `== 1.0` |
| `0x1p0` | f64 | none | none | E001 + E137 (`p0` undeclared) |
| `0x1p10` | f64 | none | none | E001 + E137 (`p10` undeclared) |
| `0x1.8p+0` | f64 | 1 digit | `+` | builds, `== 1.5` |
| `0x1.8p0` | f64 | 1 digit | none | E012 + E137 |
| `0x1.fp0` | f64 | 1 hex letter | none | E012 |
| `0x1.0fp0` | f64 | digit, then letter | none | E012 + E137 (`fp0`) |
| `0x1.0000000000001p0` | f64 | 13 digits | none | E012 + E137 |
| `0X1.8P1` | f64 | 1 digit, uppercase | none | E012 + E137 |
| `0x1p0 + 2.0` | f64 | none | none | E001 + E137 |
| `0x1.0000000000000001p+0` | f128 | 16 digits | `+` | builds, 1 + 2^-64 |
| `0x1.0000000000000001p0` | f128 | 16 digits | none | E012 + E137 (the report) |
| `0x1.8p+0` | f128 | 1 digit | `+` | builds |
| `0x1.8p0` | f128 | 1 digit | none | E012 + E137 |
| `0x1p0` | f128 | none | none | E137 |

**The only variable is the sign.** Every unsigned form fails. The fraction
length, the f64 or f128 target and the letter case make no difference. The
16-digit fraction in the report is incidental.

## 3. Root cause

Madaros does not lex through `scan_number` / `extend_c99_hex_float`
(`self-hosted/lexer/mod.sio`). That Cursor-based scanner accepts `p` followed
by a plain digit, but it is unreachable. The comment above the digit-separator
loop in `lex_source_to_globals` already records this. The live lexer is the
flat loop in `lex_source_to_globals`. For a `0x` literal it consumes hex
digits only and emits `HexLit` (code 129). Everything after that is lexed by
the ordinary rules:

| Source | Tokens |
|---|---|
| `0x1.8p+0` | `HexLit(0x1)` `Dot` `IntLit(8)` `Ident(p)` `Plus` `IntLit(0)` |
| `0x1.8p0` | `HexLit(0x1)` `Dot` `IntLit(8)` `Ident(p0)` |
| `0x1p10` | `HexLit(0x1)` `Ident(p10)` |
| `0x1.fp0` | `HexLit(0x1)` `Dot` `Ident(fp0)` |

The identifier rule swallows the exponent digits, because `p0` is a valid
identifier. A sign stops the identifier, which is why only the signed form
worked.

The parser puts the literal back together (`parser_hex_float_tail_ahead` in
`self-hosted/parser/parser.sio`, then `parse_c99_hex_float_literal` in
`self-hosted/parser/exprs.sio`). The lookahead recognised the exponent marker
in only two shapes: an Ident that is exactly `p`/`P`
(`parser_ahead_is_p_ident`), or hex digits ending in `p`
(`parser_ahead_is_hex_frac_p_ident`). In both shapes it then required a
separate `IntLit`. `Ident(p0)` matched neither shape, so the `HexLit` fell
through to `parse_int_literal`. `.8` became a field access on the integer
`0x1` (E012), and `p0` was left behind as an undeclared variable (E137).

The byte-level converters `f64_hex_literal_from_source` (`parser.sio`) and
`f128lit_from_hex` (`self-hosted/parser/f128_literal.sio`) already accept an
unsigned exponent: the sign is optional in both. The defect was only that the
parser never handed them the bytes.

## 4. Fix

- `parser_ahead_fused_exp_ident(p, offset, min_hex)` recognises a glued
  Ident of the form `<hex digit>{min_hex,} [pP] <decimal digit> <ident chars>*`.
  Any identifier characters after the exponent digits are a glued suffix. They
  are accepted on the same terms as a glued suffix Ident after `p+0`, and the
  converters stop reading at the first non-digit.
- `parser_hex_float_tail_ahead` accepts that Ident, glued to the previous
  token, in every position where it already accepted `p`:
  - after the `HexLit`, as in `0x1p0`;
  - after `Dot` + fraction token, as in `0x1.8p0` and `0x1.0fp0`;
  - after `Dot` alone, as in `0x1.fp0`;
  - after a fused `FloatLit`.
- `parse_c99_hex_float_literal` records whether the exponent digits arrived
  inside the Ident. If they did, it consumes nothing further. Without that
  guard, `0x1p0 + 2.0` would swallow `+ 2` as a signed exponent, and a glued
  identifier would be taken as a suffix.

The fix does not touch the lexer, lean_single or the aarch64 path.

## 5. lean_single is not a control for this literal, at any sign

Measured with `bin/souc-lean-single-x86_64 r.sio r.elf` on the same programs:

| Literal | lean_single |
|---|---|
| `0x1p+0` (f64) | E001 + E200 `undefined identifier p` |
| `0x1.8p+0` (f64) | E200 `undefined identifier p` |
| `0x1.8p0` (f64) | E200 `undefined identifier p0` |
| `0x1.8p+0` (f128) | "f128 literal is not exactly representable in binary64", then E200 `p` |
| `0x1.0000000000000001p+0` (f128) | same |

lean_single has no C99 hex-float support in f64 or in f128, with or without a
sign. Its f128 message blames binary64 widening, but that text follows from
the token split: 1.5 is exact in binary64. This is recorded here and is out of
scope for this dispatch. No lean_single-visible test changes, so the aarch64
compile-fail parity gate is not affected.

## 6. Verification

Madaros built from this branch's parser fix on base `f141ad5d9`
(`make build-madaros`, run bare; `artifacts/self-hosted/madaros` md5
`f8bde091ed0ed34d7b4519b02a71a39b`). The branch was then rebased onto `main`
`9c5ffa236`, which adds only `self-hosted/ir/lower.sio` (#2666) and CLAUDE.md
(#2712) since that base.

| Check | Fixed build | Baseline `5764851f` |
|---|---|---|
| repro matrix (§2), 15 rows | 15/15 build; each value exact, and the f128 prints match the signed twin | 5 signed rows build, 10 unsigned rows refused |
| `0x1.8p0f128` (glued suffix, unsigned) | builds, prints 1.5, same as `0x1.8p+0f128` | E012 + E137 |
| `0x1.8p` (no exponent digits) | still refused (E012 + E137) | refused |
| `0x10 + s.p0` (a real field named `p0`) | builds, `== 23` | builds |
| `tests/run-pass/f128_hex_float_unsigned_exponent.sio` | PASS, rc=0 | parse error at line 77 (`0x1p0 + 2.0`), rc=1 |
| `madaros_f128_f256_ladder_gate.sh --stage v0e59` | PASS, FAIL_COUNT=0 | not run |
| `madaros_f128_f256_ladder_gate.sh --stage v0e510` | PASS, FAIL_COUNT=0 | not run |
| `run_sio_test_suite.sh --filter-prefix f128_ --jobs 1`, `SOUNIO_MADAROS_AVAILABLE=1` | Pass 19, Fail 0, Skip 8 (`requires: lean_single` or ignored) | not run |
| `run_sio_test_suite.sh --filter-prefix hexfloat_ --jobs 1`, `SOUNIO_MADAROS_AVAILABLE=1` | Pass 2, Fail 0 | not run |

The `f128_` run also reports two XPAS: `f128_v0b_cast_rejected` and
`f128_v0b_implicit_conversion_rejected`. Neither fixture contains a hex
literal, and #2716 records the same two XPAS on the baseline `5764851f`, so
they are not caused by this change.

The witness declares no `//@ sabotage:` token, because no existing sabotage
switch reaches the parser. Its control is the baseline column above: the
unfixed compiler refuses it.
