Madaros parses `Knowledge[T, ε=⊥]` (Knightian ignorance). The lexer tolerated
the `⊥` bytes, but the parser stopped on them, so every `ε=⊥` annotation was a
parse error and the checker's mode-3 rules were dead code. A Knightian value
now meets a `ε=⊥` requirement and is refused wherever an error bound, a
confidence floor or an exact confidence is required, with the note lean_single
prints ("Knightian uncertainty (ε=⊥) cannot satisfy required confidence").
`knightian_mixed` and `covid_2020_knightian_refusal` are now refused by Madaros
for that reason instead of by a parse error. This is EPI-006 step 3a; named
Knightian sources (`knight("name", lo, hi)`) and the certified interval follow.

Only the exact `⊥` glyph (U+22A5, written once) is a bottom bound. `ε=⊥⊥` and `⊥` on the left of a bound are refused with E241. An identifier as a type-level ε bound (`ε < tol`, `ε=ε`) is also refused with E241; before, it was skipped and the bound became 0.0 without a diagnostic.
