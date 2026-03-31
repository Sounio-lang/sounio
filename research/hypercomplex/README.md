# Hypercomplex Algebra Research Lane

This directory contains the Wave 1 baseline plus the Wave 2 semantic-contract
package, the Wave 3 prototype-safe scaffolding, the Wave 4 compile-proof
prototype harnesses, the Wave 5 reassociation metadata scaffolding, and the
Wave 6 forbidden-law metadata scaffolding, and the Wave 7 hyper-expression law
profile scaffolding, and the Wave 8 law-profile observability scaffolding, and
the Wave 9 law-profile fingerprint observability scaffolding for hypercomplex
algebra in the compiler.

Wave 1 baseline:

- anchored at `compiler-institutional-finalization`
- baseline commit `5f6b8e5502004b1b017b53ac662aa3efa75c20dd`

Wave 2 baseline:

- builds on `hypercomplex-algebra-wave1`
- baseline commit `69da08bc663f7346743e31d0d92baa34d0b18340`

Wave 3 baseline:

- builds on `hypercomplex-algebra-wave2`
- baseline commit `55eb15cdc7e80069657e8033cb945afa4f1a8833`

Wave 4 baseline:

- builds on `hypercomplex-algebra-wave3`
- baseline commit `8c71e758c4f8eb7cc986dac26212cb8ca320d927`

Wave 5 baseline:

- builds on `hypercomplex-algebra-wave4`
- baseline commit `57c593e0a617174bcc24580e967cfee54368b437`

Wave 6 baseline:

- builds on `hypercomplex-algebra-wave5-codex`
- baseline commit `d7f14f150358e3afdad9e6350175d45b5e844775`

Wave 7 baseline:

- builds on `hypercomplex-algebra-wave6-codex`
- baseline commit `27283e9f04a5c4f68d6abe0a7fd9850289ee4b6c`

Wave 8 baseline:

- builds on `hypercomplex-algebra-wave7-codex`
- baseline commit `c2ef74888c3d3629683d1dffa58a368d59d60dad`

Wave 9 baseline:

- builds on `hypercomplex-algebra-wave8-codex`
- baseline commit `94c3f76188042c36f10f3fdcb73fe0b14f3e1b9d`

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
- `prototype_scaffolding.v1.json`: Wave 4 compile-proof and harness matrix for non-public compiler-facing scaffolding
- `prototype_scaffolding.v1.json`: Wave 5 through Wave 9 metadata and differential-selftest harnesses for non-public compiler-facing scaffolding

Validation:

```bash
bash scripts/research/hypercomplex_wave1_gate.sh
bash scripts/research/hypercomplex_wave2_gate.sh
bash scripts/research/hypercomplex_wave3_gate.sh
bash scripts/research/hypercomplex_wave4_gate.sh
bash scripts/research/hypercomplex_wave5_gate.sh
bash scripts/research/hypercomplex_wave6_gate.sh
bash scripts/research/hypercomplex_wave7_gate.sh
bash scripts/research/hypercomplex_wave8_diff_selftest.sh
bash scripts/research/hypercomplex_wave8_gate.sh
bash scripts/research/hypercomplex_wave9_diff_selftest.sh
bash scripts/research/hypercomplex_wave9_gate.sh
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

What Wave 4 adds without changing support claims:

- the first non-public compile-proof harnesses that deliberately exercise
  `Hyper<...>` signatures, struct carriage, and algebra-declaration compile
  paths through the real `compile` pipeline
- a prototype scaffolding manifest that links compile-proof harnesses, runtime
  harnesses, and forbidden-law validation into one bounded internal surface
- a Wave 4 validator that requires Wave 3 to stay green and then compiles the
  new Hyper fixtures through the canonical `resolve_souc.sh` path

What Wave 5 adds without changing support claims:

- a prototype-safe compile-proof harness that carries reassociation-strategy
  metadata through `algebra` declarations and `Hyper<...>` signatures without
  altering runtime or optimizer behavior
- a stronger validation-only contract for the still-missing generic
  distribute/factor law surface, so future prototype work cannot silently
  pretend the rewrite engine exists
- a Wave 5 validator that requires Wave 4 to stay green, compiles the new
  metadata-carriage fixture, and checks that forbidden-law gaps remain fenced by
  research-only touchpoints

What Wave 6 adds without changing support claims:

- an internal-only forbidden-law mask seam that rides beside reassociation
  metadata through checker state, serialized IR algebra metadata, and mini
  e-graph context setup
- a compile-proof fixture for `reassociate: fano_selective` plus `blocked`
  algebra lanes so forbidden-law metadata carriage stays bounded to internal
  compiler surfaces
- a stricter Wave 6 validator that requires Wave 5 to stay green, compiles the
  new fixture, and symbol-checks the required reassociation and forbidden-law
  metadata references instead of trusting path existence alone

What Wave 7 adds without changing support claims:

- a bounded expression-level seam where each recorded hyper multiply site carries
  its own reassociation strategy and forbidden-law profile through the checker
  and serialized epistemic IR
- a compile-proof fixture that exercises actual hyper multiplication expressions,
  not just declarations and signatures, so the law profile seam reaches
  expression recording
- a stricter Wave 7 validator that requires Wave 6 to stay green, compiles the
  new expression fixture, and symbol-checks the expression-level law-profile
  references so future prototype work cannot drift semantically

What Wave 8 adds without changing support claims:

- a bounded registry-vs-fallback baseline-comparison path that compiles a small
  differential harness and checks the new observability seam against the Wave 7
  baseline without changing runtime or optimizer behavior
- a small observability seam where expression-level hyper metadata records the
  source of each law profile so research tooling can tell whether a profile came
  from the algebra registry or fallback derivation
- a stricter Wave 8 validator that requires Wave 7 to stay green, runs the new
  differential self-test, and symbol-checks the source-tag references so future
  prototype work cannot silently drift from advisory metadata into semantic
  claims

What Wave 9 adds without changing support claims:

- a stronger compile-backed baseline-comparison path that pins exact
  registry-vs-fallback law-profile fingerprints so observability drift is
  easier to detect even when the runnable lane remains non-blocking
- a small observability seam where expression-level hyper metadata carries a
  derived law-profile fingerprint through checker bookkeeping, epistemic IR, and
  serialization without changing supported behavior
- a stricter Wave 9 validator that requires Wave 8 to stay green, runs the new
  differential self-test, and symbol-checks the fingerprint references so
  future prototype work cannot silently drift advisory metadata into semantic
  authority

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
- Wave 4 compile-proof harnesses are internal evidence for compiler carriage
  only; they do not imply runtime, optimizer, or ABI support
- Wave 5 metadata-carriage harnesses are still compile-proof only; they do not
  upgrade rewrite-engine, optimizer, runtime, or public maturity claims
- Wave 6 forbidden-law metadata masks are still internal compiler annotations;
  they do not authorize broader rewrites, runtime claims, or public support
- Wave 7 hyper-expression law profiles are still internal compiler bookkeeping;
  they do not create a public semantic contract or authorize new rewrites
- Wave 8 law-profile source tags and registry-vs-fallback baseline comparisons
  are still internal observability only; they do not authorize
  broader rewrites, runtime claims, or public support
- Wave 9 law-profile fingerprints and fingerprint-based baseline comparisons are
  still internal observability only; they do not authorize broader rewrites,
  runtime claims, or public support
