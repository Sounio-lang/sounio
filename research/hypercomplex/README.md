# Hypercomplex Algebra Research Lane

This directory contains the Wave 1 baseline plus the Wave 2 semantic-contract
package and the Wave 3 prototype-safe scaffolding for hypercomplex algebra in
the compiler.

Wave 1 baseline:

- anchored at `compiler-institutional-finalization`
- baseline commit `5f6b8e5502004b1b017b53ac662aa3efa75c20dd`

Wave 2 baseline:

- builds on `hypercomplex-algebra-wave1`
- baseline commit `69da08bc663f7346743e31d0d92baa34d0b18340`

Wave 3 baseline:

- builds on `hypercomplex-algebra-wave2`
- baseline commit `55eb15cdc7e80069657e8033cb945afa4f1a8833`

Scope:

- inventory the hypercomplex surface already present in the accepted baseline
- classify touchpoints with explicit maturity labels
- map hazards before any broader implementation or support claim
- add a repo-local validation entrypoint for this lane

Non-goals:

- no public support expansion
- no required CI promotion
- no optimizer-law broadening
- no compiler/runtime behavior changes in this wave

Files:

- `taxonomy.v1.json`: machine-readable maturity classes, categories, and state vocabulary
- `inventory.v1.json`: concrete touchpoints, research assets, contradictions, and gaps
- `hazards.v1.json`: semantic hazard map tied back to actual touchpoints
- `semantics.v1.json`: compact Wave 2 semantic contract for prototype-safe vs deferred assumptions
- `touchpoints.v1.json`: bounded map of prototype-safe compiler-facing candidate areas
- `roadmap.v1.json`: internal Wave 3 experiment frame
- `expected_fail.v1.json`: Wave 3 forbidden-rewrite coverage, mixing executable counterexamples with explicit validation-only gaps
- `compiler_audit.v1.json`: bounded internal audit seams for type, IR, and lowering touchpoints

Validation:

```bash
bash scripts/research/hypercomplex_wave1_gate.sh
bash scripts/research/hypercomplex_wave2_gate.sh
bash scripts/research/hypercomplex_wave3_gate.sh
```

What the baseline already contains:

- Parser support for `algebra Name over Type { ... reassociate: strategy }`
- `Hyper<Algebra, T>` types plus `NonAssoc` and division-domain checks
- IR algebra metadata, hyper opcodes, and Fano-selective reassociation hooks
- Native lowering for quaternion, octonion, and sedenion-oriented ops
- Stdlib octonion and sedenion surfaces plus execution/test gates
- Research and paper material around associators, zero divisors, and Cayley-Dickson structure

What Wave 1 still treats conservatively:

- broad reassociation or normalization for non-associative expressions
- public maturity/support claims for hypercomplex execution surfaces
- ABI or interop promises around hypercomplex native layout
- optimizer laws that rely on norm, inverse, or zero-divisor assumptions outside proven domains

What Wave 2 adds without changing support claims:

- a minimal semantic contract for parenthesization, associativity, distributivity,
  zero divisors, division domains, norm/conjugation, reorderability, and optimizer-law safety
- a bounded touchpoint map for prototype-safe compiler-facing work
- a non-public validator that checks wave1 + wave2 manifests together
- a short internal roadmap for Wave 3 experiments

What Wave 3 adds without changing support claims:

- executable non-public witnesses for the forbidden rewrite classes that already
  have concrete repo-local counterexamples
- explicit validation-only coverage for forbidden rewrite classes that remain
  unsupported because the compiler still has no checked general law surface
- a bounded compiler-audit manifest covering type tagging, IR metadata, and
  native lowering seams
- a Wave 3 validator that executes the safe witnesses and cross-checks the new
  manifests against Wave 1 and Wave 2

Current inventory summary:

- active entries: 9
- historical entries: 2
- contradictory entries: 1
- absent entries: 2

Required maturity classes:

- `research-only`: evidence or ideas that are real, but not safe to promote into support claims or broad compiler transformations
- `prototype-safe`: bounded, testable work that can proceed without changing public support semantics
- `production-deferred`: present or desirable surfaces that need stronger contracts, validation, and governance before any production claim

Important current gaps:

- no hypercomplex-specific required check in `scripts/selfhost/selfhost_required_checks.v1.json`
- no general symbolic normalizer for non-associative expressions beyond narrow Fano-selective basis reassociation
- public docs do not yet tell one consistent maturity story across README, limitations, and technical-report surfaces

Wave 2 operating rule:

- this remains a research/prototype lane
- no public support claim is expanded here
- no required CI gate is promoted here
- any future compiler experiment must stay bounded by `semantics.v1.json`
- Wave 3 witnesses stay non-public and are not part of the normal run-pass or
  compile-fail suite
