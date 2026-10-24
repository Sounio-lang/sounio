<!-- docs:meta
topic_id: repo.docs.decisions.adr-015-lexical-structure
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-015-lexical-structure
-->

# ADR-015: Lexical structure (the thirteen rulings of D-33)

**Status**: accepted (implementation pending)
**Date**: 2026-10-24

## Context

The lexical execution receipt (`docs/spec/2026-10/18-lexical-execution-receipt.md`) ran chapter 01's `LEX-001`–`LEX-013` with 125 probes on both engines and on a Madaros built from source. Where the engines disagreed, or where both accepted something they should not, the receipt could measure the behaviour but not choose it. `PENDING.md` D-33 collected those thirteen choices. The defects that needed no choice (fail closed with a location) are fixed separately in PR #3016 and are not part of this ADR.

Two of the measured behaviours compute a wrong value without any diagnostic:

- an integer literal that does not fit wraps (`99999999999999999999` becomes `7766279631452241919` on both engines);
- `lean_single` evaluates `.5` to `0`.

## Decision

The founder ruled on all thirteen items on 2026-10-24. Probe names refer to the receipt.

| # | item | ruling | discriminating probes |
|---|---|---|---|
| 1 | integer literal that does not fit its type | **Refuse**, with a location, on both engines | `int_huge`, `int_max_plus1`, `int_hex_over` refused on both |
| 2 | float with a leading dot (`.5`) | **Refuse now**, with a diagnostic that suggests `0.5`. Reopen as 0.5 only when both engines agree on the value | `flt_leading_dot` refused on both |
| 3 | escape set | **Closed set**: `\n \t \r \0 \\ \" \'`, `\xHH`, `\u{…}`; anything else refused. **A string may contain NUL**: strings are bytes plus length, and Madaros must stop truncating at the first NUL | `str_escape_hex`, `str_escape_nul` (prints three bytes on both), `str_escape_unicode`, `str_escape_unknown` refused |
| 4 | character literals and lifetimes | `'x'` is a **Unicode scalar value**. `'name` with no closing quote is a **lifetime**, so `&'static str` lexes as a lifetime, not as the character `s` followed by `tatic`. A one-character literal with no closing quote is refused | `chr_basic`, `chr_utf8` (scalar value), `chr_unknown_escape` refused, `chr_unclosed` refused; stdlib `&'static` still compiles |
| 5 | binary and octal literals | **Implement** `0b` and `0o` in the live lexer on both engines. `BinLit`/`OctLit` stay | `int_bin`, `int_oct` evaluate correctly on both |
| 6 | type suffixes (`5u8`, `2.0f32`) | **Accept**. The suffix types the literal, and the range check of item 1 applies to the suffixed type (`300u8` is refused) | `int_suffix_u8`, `int_suffix_i32`, `flt_suffix`; new probe `int_suffix_u8_over` refused |
| 7 | unit suffixes (`500_mg`) | **Part of the language**, normative on both engines, not a `lean_single` extension. The unit grammar itself belongs to the units decision (D-24) | `lit_unit_suffix` accepted on both with the same unit |
| 8 | digit separators | **Madaros rule**: `_` only between digits. `lean_single` follows | `int_sep` accepted; `int_sep_double`, `int_sep_trail` refused on both; `flt_sep` |
| 9 | maximal munch of `<-`, `+-`, `&!` | **Measure** use of each multi-character token in the corpus; **remove** the tokens with no use; for the rest, **refuse** the ambiguous spelling (`x<-1`) with a diagnostic that suggests a space | `op_lt_neg`, `op_plus_neg`, `op_and_not` and their `_sp` variants; corpus count recorded in the implementing PR |
| 10 | reserved words | Reserve **only what the grammar requires in name position**, and list them in chapter 01. Contextual words (`affine`, `linear`, `study`, …) remain valid names outside keyword position; reading them as `0` is a defect regardless | `kw_let_*` (50 spellings and 4 controls) agree on both engines |
| 11 | identifiers | **Length limit with refusal**, never a split. Alphabet: **Unicode UAX #31 XID** (`XID_Start XID_Continue*`), compared after **NFC** normalisation. In addition, `⊤ ∂ ∇ ∞` are accepted as identifier characters. **`⊥` is not an identifier character**: it becomes a token of its own in the grammar (see Consequences) | `id_long300` refused with location; `id_nonascii` (`σ`, `κ`) accepted; `id_epsilon` (`ε`) accepted; NFC and non-NFC spellings of the same name resolve to one binding |
| 12 | documentation comments (`///`, `/** */`) | **Tokens**, attached as metadata to the following item, so the reference documentation can be generated from the source | `cm_doc_line`, `cm_doc_block` produce doc tokens; doc text reachable from the item |
| 13 | dead lexer code | **Remove** the unreachable code in `mod.sio` (after `return tc`, and the scanners only it calls), `numparse.sio` and the dead half of `tables.sio`, **after** porting what items 3–6 and 12 need from them | none; build and probe set unchanged before and after removal |

## Consequences

- Two silent wrong values (items 1 and 2) become refusals. Programs that relied on wrapping literals or on `.5` stop compiling, with a location.
- `0_⊥`, the typed bottom zero of the ontology-algebra research record, can no longer be an ordinary identifier. It needs a grammar rule for `⊥` as a token; that rule is owed to the typed-zero line and is **not** decided here.
- Item 4 removes the lifetime collision measured in D-33: refusing an unclosed character literal no longer breaks `&'static` in the standard library, because `'static` lexes as a lifetime first.
- Item 7 makes unit literals normative before the unit grammar is fixed; until D-24 is decided, the implementation must refuse unit suffixes it cannot resolve rather than drop them.
- Items 6 and 1 interact: a suffix changes the type whose range is checked.

## Implementation

Not started. Each row above is one implementing change, gated by its probes in the receipt; the receipt is re-run after each, and a row is closed when its probes agree on both engines. Item 13 runs last.

## Grounded in

- `docs/spec/2026-10/18-lexical-execution-receipt.md` and `docs/audit/S01_LEXICAL_PROBES_2026-10-22/`.
- `docs/decisions/PENDING.md` D-33 (removed by this change; its text is preserved in the history of that file).
