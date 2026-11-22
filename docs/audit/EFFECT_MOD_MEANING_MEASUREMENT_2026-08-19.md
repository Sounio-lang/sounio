<!-- docs:meta
topic_id: repo.docs.audit.effect-mod-meaning-measurement-2026-08-19
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.effect-mod-meaning-measurement-2026-08-19
-->

# Effect `with Mod`: historical audit and structural instruments

Autoria e responsabilidade humana: Demetrios Agourakis. Agent names in historical
records identify execution and review, not ownership of the language or repository.

## Status of the 2026-08-19 report

The original report is preserved in Git at
[`b438b42adbbd9deda1991d79cc788f6d5ab3b8f3`](https://github.com/Sounio-lang/sounio/blob/b438b42adbbd9deda1991d79cc788f6d5ab3b8f3/docs/audit/EFFECT_MOD_MEANING_MEASUREMENT_2026-08-19.md).
Its aggregate counts and conclusions are superseded. Review of #2631 identified
three defects in its instruments: signatures with comma-separated effects were
omitted, return-type arrows caused effect clauses to be erased before co-occurrence
counting, and the neighbour script used an outdated recognized-name list. These
invalidate the exhaustive sample, zero-co-occurrence claims and orphan
classifications. The historical report is not a current compiler validation receipt.

The coordination claim about #1963 was also incorrect: that change held `Mod`
unknown, admitted the phase-2b named effects, and used the row-polymorphism slot for
`EffVar`; it did not admit `Mod` at id 29. The current checker subsequently gives
`Correlated` the next named slot and treats `Confidence` as an `Epistemic` alias.
The executable source, not an old proposed numbering, is authoritative.

## Reconciled instruments

- [`measure_with_mod.cjs`](../../scripts/audit/measure_with_mod.cjs) masks strings
  and comments before brace matching, extracts comma-separated effect clauses
  after return types, includes `Mod` in any clause position, and checks every
  declared variable for a later assignment pattern. It reports structural
  mutation/modulo patterns and co-occurring names as JSON.
- [`measure_with_neighbors.cjs`](../../scripts/audit/measure_with_neighbors.cjs)
  counts each name in comma-separated clauses and derives named effects from
  `self-hosted/check/effects.sio` production enum, excluding `EffVar` and including
  the documented `Confidence` alias. Missing enum/alias contract fails the scan.

Focused structural fixtures run without executing scientific calculations:

```sh
node scripts/audit/measure_with_mod.cjs --selftest
node scripts/audit/measure_with_neighbors.cjs --selftest
node scripts/audit/measure_with_mod.cjs .
node scripts/audit/measure_with_neighbors.cjs .
```

Selftests cover leading/trailing comma positions, return-type effect clauses,
multiline clauses, braces and effect words in strings/comments, a reassigned
second variable, and the current phase-2b/Correlated/Confidence recognized names.

## What the output establishes

The output is a source-text heuristic over the included `.sio` files.
It is not an exhaustive parser/typechecker census: the walk excludes symlinks,
`archive`, `bootstrap`, `target`, `.git` and `node_modules`; nested function-type
clauses may be included in a signature; complex declarations can exceed the
text scanner grammar. Assignment patterns do not establish semantic mutation,
spellings near builtins do not prove typos, and absence of a declaration in this
walk does not establish that an effect is invalid. Runtime behaviour and the
meaning of `Mod` require separate compiler evidence and human semantic decisions.

The historical snapshot was `5f3cc9b296`. Reconciliation was checked against main
`3fd2caee8fa114e9bc92d62c109fe64d11a8dc36`; no `.sio` source was changed. Results
must identify their input source and scanner version when used again. No current
scientific calculation, runtime result or effect admission is claimed here.
