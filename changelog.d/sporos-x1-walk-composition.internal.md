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
