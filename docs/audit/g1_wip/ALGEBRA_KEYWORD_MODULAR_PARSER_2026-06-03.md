# algebra/study keyword recognition in modular pt_read_kind — +3 PASS, 0 regr (2026-06-03)

Branch `parser/algebra-keyword-e008` (stacked on `parser/sci-notation-modular-e008`, off
integration/e008 tip `df8d1db36`). Commit `2b04c0153`.

## Root cause (small dispatch gap, not a grammar port)
`parse_algebra_item` / `parse_study_item` / `parse_ontology_item` AND the
`item_kind == TokenKind::Algebra/Study/Ontology` dispatch ALL already exist in
`self-hosted/parser/items.sio`. The gap was ONE layer up: `pt_read_kind` (parser.sio:448, the
active classifier `parser_peek` uses) recognizes keywords purely by BYTE-SCANNING CURSOR_SOURCE
(PT_KIND_CODE is only used for the EOF check). It had **no `len == 7` block at all** — so the
7-char word `algebra` lexed as Ident and `parse_item` fell to error recovery ("parser reported
N syntax errors"). `study` (len 5) was likewise missing from the `len == 5` block. Ontology
(len 8) was already present and passes end-to-end.

## Fix
Two byte-scan entries: `study` into the `len == 5` block, a new `len == 7` block for `algebra`.

## Result (modular census, 504 run-pass, mc rebuilt via build lock)
| | PASS | CRASH | regr |
|---|---:|---:|---:|
| sci-notation baseline | 235 | 0 | — |
| + algebra/study keywords | **238** | 0 | **0** |

**+3 FAIL→PASS** (algebra_decl_basic, octonion_hessian_fano_annotated, study_block_basic),
0 regressions.

## Honest limit — 5 algebra programs double-blocked downstream (advisor's parse-further case)
The keyword fix is CORRECT and complete (the parse stage is closed): the 5 richer algebra
programs (algebra_commutative_default, algebra_g2_invariants, algebra_g2_null_model,
algebra_observe_synthesis, algebra_properties_basic) now PARSE cleanly and fail at TYPE CHECKING
with **E016 "field initializer has wrong type — expected f32"** — an f32 literal-width narrowing
bug in struct field initializers (same family as E004/E008 return narrowing), a CHECKER fix, not
a parser one. So the realized +3 is the sole-blocker subset; the f32-field E016 narrowing is the
next lever for the remaining 5.

## Blast radius
parser.sio change to the MODULAR path only; `lean_single.sio` untouched ⇒ `bin/souc` unchanged,
`canonical_compiler_gate.sh` PASS (md5 `05348095`). x86/arch-neutral (byte comparisons).
ontology was NOT a gap (already works); the earlier "9 ontology" no-line count was a first-line
keyword over-count.
