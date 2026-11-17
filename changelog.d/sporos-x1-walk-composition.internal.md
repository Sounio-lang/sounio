New module `formal/Bootstrap/SporosX1Comp.lean`, registered in
`formal/lakefile.lean` (`Bootstrap` library). It composes the
per-handler Hoare triples of `Bootstrap.SporosX1` (walk dispatch,
the thirteen special handlers and primaries, `fall_tail`,
`none_tail`) into the single-pass stack-effect walk of
`Sporos.Check1`. Handler and tail specs in `SporosX1` gained the
register frames and memory write-shape clauses the composition
needs, and five small handlers were proved again in the framed
`HandlerSpecF` form. Built by `lake build Bootstrap`; no
`ci.yml` change, so the lean cache contract hash is untouched
(verified with `scripts/ci/lean_cache_contract.mjs --selftest`).
The composition now reaches the accept exit: `annval` (annOk), `codewalk`
(initial walk invariant), the outer `check_word` loop and
`SporosX1.checker_accepts_from_start` are proved in
`formal/Bootstrap/SporosX1Comp.lean` and audited in `formal/AxiomReport.lean`
(axioms: propext, Classical.choice, Quot.sound). The refusal direction has
begun in the same module: a generic status-exit block and `fail1` exit,
refusal of a word with no instructions, and refusal at the first
out-of-range annotation entry after a good prefix. `gen_seed_data.sh`
now also generates the stage-1 listing and its decoded bytes into
`SeedData.lean`, and `formal/Bootstrap/SporosX1Image.lean` (registered
in the lakefile) instantiates `AllCode`, `OuterCode` and `RefuseCode`
at `imageOf` of those bytes. The remaining refusal branches and the
parse/decode phase remain open.
