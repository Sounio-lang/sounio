# Hypercomplex Algebra Wave 1

This directory is the Wave 1 research package for hypercomplex algebra in the
compiler baseline anchored at `compiler-institutional-finalization`
(`5f6b8e5502004b1b017b53ac662aa3efa75c20dd`).

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

Validation:

```bash
bash scripts/research/hypercomplex_wave1_gate.sh
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
