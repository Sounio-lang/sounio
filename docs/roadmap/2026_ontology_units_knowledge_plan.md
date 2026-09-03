<!-- docs:meta
topic_id: repo.docs.roadmap.2026-ontology-units-knowledge-plan
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.roadmap.2026-ontology-units-knowledge-plan
-->

# Sounio Semantic Knowledge Spine Roadmap

Status file: `docs/roadmap/2026_ontology_units_knowledge_status.md`

Short name: Semantic Knowledge Spine.

Goal declaration:

- Name: Sounio Semantic Knowledge Spine.
- Objective: implement the 1->2->3 semantic spine from generated/imported
  ontology facts, through unit/dimensional checking, into `Knowledge<T>` proof
  contexts with static discharge and explicit runtime obligations for dynamic
  evidence.
- Scope boundary: this goal is not complete until imported ontology caches can
  act as ordinary checker authority, unit labels are mirrored through the
  modular unit registry, and dynamic proof obligations lower into the native
  backend guard/trap path. The current green gates are strong intermediate
  rungs, not the final endpoint.

## Vision

Sounio should treat ontology, physical dimensions, and knowledge evidence as
first-class compile-time facts.

The target language experience is:

- ontology classes are usable as semantic types;
- subclass and disjointness facts are checked by the compiler;
- unit and dimension annotations reject incompatible calculations unless an
  explicit conversion is present;
- `Knowledge<T>` can carry a proof context such as ontology, unit, and numeric
  refinement constraints;
- dynamic evidence can still compile, but receives generated runtime checks
  instead of being silently trusted.

The implementation rule for this roadmap is: data may be ingested through
external tooling, but semantic reasoning belongs to Sounio. Python is not part
of the PL core. Current generated ontology ingestion uses a C FFI importer; the
compiler checker remains the authority for subclass, disjointness, unit, and
proof-context decisions.

## Layer 1: Generated Ontology Types

Goal: make `.dontology` bundles available as Sounio ontology blocks, constants,
and semantic type witnesses without modifying the compiler bootstrap path.

TODOs:

- [x] Build a C importer for `.dontology` data.
- [x] Generate `.sio` ontology stubs for the available bundles.
- [x] Generate positive witnesses for subclass acceptance.
- [x] Generate negative witnesses for sibling or disjoint rejection.
- [x] Add a Make target for deterministic regeneration.
- [x] Add a generated artifact freshness gate target for CI use.
- [x] Add a stable public manifest for generated bundles and witness coverage.

Current gate:

```bash
make test-generated-ontology
make test-generated-ontology-manifest
make test-generated-ontology-fresh
```

## Layer 2: Compiler-Visible Ontology Reasoning

Goal: the compiler can use ontology facts while checking ordinary source code.

Subgoals:

- [x] Keep ontology blocks parsed into AST.
- [x] Materialize subclass and disjointness in the checker ontology kernel.
- [x] Bind ontology classes as named semantic types.
- [x] Check call-boundary ontology subsumption.
- [x] Parse `Knowledge<T where { ... }>` proof contexts in source parser.
- [x] Add a Stage1 checker-verdict AST bridge for static field-level ontology
  constraints in `Knowledge<T>`.
- [x] Prove the ordinary parser/AST route retains static field-level ontology
  constraints without source-text matching.
- [x] Prove ordinary Stage1 frontend positive/negative verdicts through the AST
  bridge.
- [x] Prove a transitive subclass witness through the AST bridge.
- [x] Validate static ontology proof contexts in `Knowledge<T>` return types,
  not only parameter types.
- [x] Prove multi-constraint ontology proof contexts are retained and checked
  as conjunctions by the ordinary Stage1 AST bridge.
- [x] Route the same static field-level ontology constraints through the
  boot4-safe checker verdict path without the frontend AST bridge.
- [x] Move the boot4-safe AST validator into dedicated
  `check::knowledge_context` checker source instead of duplicating it inside
  `check::mod`.
- [x] Add an in-place `check::check` validator surface that avoids returning a
  large `Checker` value.
- [x] Retire the unreferenced pure AST `check::check` duplicate after moving
  the boot4-safe validator into `check::knowledge_context`.
- [ ] Move the boot4-safe verdict bridge into the full `check::check` semantic
  implementation.
- [x] Mirror the proof-context parser surface into bootstrap snapshots.
- [x] Add a narrow in-place checker wrapper for proof-context validation.
- [x] Add a scalar `check::check` semantic verdict wrapper that keeps
  allocation and in-place minimal ontology collection inside `check::check`.
- [ ] Repair the `check::mod -> check::check` import/link ABI so the wrapper
  executes from `check_items_verdict_boot4`.
- [x] Prove direct `Checker` table reads in `check::mod` are not a viable
  semantic bridge; the positive witness segfaults after collection.
- [ ] Add a safe accessor/import shape so `check::mod` can request semantic
  `check::check` validation without reading `Checker` fields directly.
- [x] Remove the unreferenced value-style `Checker` proof-context validation
  methods; keep the candidate semantic surface in-place/pointer-shaped.
- [x] Prove importing the scalar semantic verdict wrapper from `check::mod`
  still segfaults, narrowing the fault to executable `check::check` imports.
- [x] Prove an explicit post-collect scalar accessor import from `check::mod`
  into `check::check` still segfaults in the Stage1 positive witness.
- [x] Add a K2 import bridge classifier proving a leaf `check::check` import
  executes and keeping heap, pointer, accessor, semantic, and collector probes
  separated by failure class.
- [x] Extend the K2 bridge classifier to prove `*mut Checker` can cross an
  imported `check::check` boundary when not dereferenced, and that the current
  small-driver failure is dominated by `heap_alloc` before checker init.
- [x] Narrow the small-driver heap failure to the imported `mem::box`/`calloc`
  wrapper; the compiler builtin `heap_alloc` path can allocate a `Checker`.
- [x] Prove standalone `Checker` allocation, init, and imported scalar accessor
  work with builtin `heap_alloc`, while imported `checker_collect_items_mut`
  still segfaults after `init_ok`.
- [x] Prove a standalone imported `check::check` semantic verdict wrapper can
  accept positive and multi-constraint `Knowledge<T>` ontology witnesses and
  reject a negative witness through in-place minimal collection.
- [x] Re-test the ordinary Stage1 `check::mod -> check::check` route after the
  minimal collector: explicit import still segfaults, while unqualified call
  reaches stale behavior and lets the negative witness pass.
- [x] Add a focused `check::mod` diagnostic bridge probe proving a fresh
  unqualified semantic symbol still resolves stale and lets the negative
  witness pass.
- [ ] Repair explicit-import side effects that make pass-through witnesses
  segfault during subsequent checker collection.
- [ ] Add current-source binary coverage once the bootstrap artifact is
  regenerated.
- [x] Add a pre-native `//@ ontology-bundle: "..."` expansion gate that uses
  the C importer to materialize bundle declarations before checker execution.
- [x] Add deterministic pre-native text `.ontocache` reuse for repeated
  ontology-bundle expansion.
- [x] Add a compiler-side raw-source scanner that can see
  `//@ ontology-bundle: "..."` before lexing drops comments and extract the
  bundle path as the first native-loader rung.
- [x] Extend the compiler-side directive scanner into a load plan that verifies
  the referenced `.dontology` bundle exists, has bytes, and rejects malformed
  directive payloads before native ingestion work begins.
- [x] Validate the referenced bundle's deterministic `DONT` v1 envelope in the
  compiler-side load plan, including nonzero payload size and payload-size/file
  size consistency, while keeping JSON ingestion and side-table population as
  future work.
- [x] Add a minimal compiler-side payload summary for `DONT` bundles, proving
  canonical payload fields (`ontology`, `term_count`, `terms`,
  `disjoint_count`, `disjoints`) are present and rejecting valid envelopes with
  invalid payload shape before side-table ingestion.
- [x] Extract the first payload term into the compiler-side load plan,
  including ontology name, first-term CURIE, and numeric-tail witness for the
  SNOMED fixture, without yet claiming full class/parent/disjoint ingestion.
- [x] Extract the first child-parent subclass edge from the compiler-side load
  plan (`SNOMED:404684003 -> SNOMED:138875005`), proving a first subsumption
  fact can be materialized before native side-table/kernel integration.
- [x] Extract the first disjoint pair from the compiler-side load plan
  (`PART:0000003` disjoint `PART:0000012`), proving a first disjointness fact
  can be materialized before native side-table/kernel integration.
- [x] Preserve alphanumeric CURIE-tail bundles in the bounded native cache by
  assigning synthetic positive cache-slot IDs when no numeric tail exists;
  LOINC now writes and reimports a bounded parent edge without changing the
  numeric IDs used by SNOMED/PART/QM/ALG/ChEBI/GO/HPO/PHYS.
- [x] Build a bounded compact side-table prefix inside the compiler-side load
  plan, using primitive scalar/parallel fields for class IDs, parent edges,
  and disjoint pairs with explicit truncation, before checker-kernel side-table
  integration.
- [x] Add query helpers over the bounded compact side-table prefix for direct
  subclass edges, bounded subclass-chain lookup, and symmetric disjoint lookup,
  without yet claiming checker-kernel ingestion.
- [x] Stage kernel-shaped rows from the bounded prefix: class slots plus
  subclass/disjoint axiom rows indexed like `OntologyKernel` materialization,
  with equivalent query helpers, without yet mutating `Checker.ontology_kernel`.
- [x] Materialize a bounded `OntologyKernel`-like fact surface from those staged
  rows, including reflexive subclass facts, transitive subclass closure, and
  symmetric disjoint facts, without yet mutating `Checker.ontology_kernel`.
- [x] Persist those bounded staged kernel facts into a stable native
  side-table `.ontocache` artifact generated by Sounio itself, with an explicit
  authority boundary that the artifact is not the live checker kernel.
- [x] Reimport bounded native side-table `.ontocache` artifacts generated by
  Sounio and rehydrate staged kernel rows so the imported cache answers the
  same bounded materialized subclass/disjoint queries as the original load
  plan.
- [x] Add a source-level `//@ ontology-side-table-cache: "..."` directive so a
  second source file can import a bounded native `.ontocache` generated from a
  different source file and answer the same bounded materialized queries.
- [x] Add a checker-shaped semantic verdict surface over the imported cache
  directive: proven bounded subclass/disjoint queries return `0`, while
  rejected subsumption/disjoint queries return `152`.
- [x] Add a `check::mod` diagnostic wrapper backed by a check-owned native
  side-table cache reader, proving bounded imported `.ontocache` verdicts are
  visible at the checker module boundary without yet mutating
  `Checker.ontology_kernel`.
- [x] Add a focused `check::check` live-kernel hydration probe that builds a
  real `Checker`, populates `Checker.ontology_kernel` from bounded imported
  `.ontocache` slots and subclass/disjoint axioms, materializes the kernel,
  and answers verdicts through `ontology_kernel_is_subclass` and
  `ontology_kernel_is_disjoint`.
- [ ] Route the focused live-kernel cache verdict through `check::mod` without
  reintroducing explicit-import side effects in the ordinary Stage1 checker
  path.
- [x] Add a candidate `items + source_path` sourcecheck hydrator in
  `check::check` and classify its current blocker: it parses the source and
  resolves the cache/indexes, but imported `collect_items` still segfaults
  before `collect_ok`.
- [x] Add a `check::mod`-local sourcecheck candidate and classify the
  pointer-owned hydration blocker: it parses, resolves cache indexes, reaches
  `collect_ok` and `check_ok`, then blocks while hydrating the pointer-owned
  `Checker.ontology_kernel`.
- [x] Add a fresh-symbol `check::mod` sourcecheck cache-verdict bridge that
  runs ordinary sourcecheck first, then accepts the bounded positive subclass
  verdict and rejects the bounded reverse subclass verdict without explicit
  `check::check` imports.
- [ ] Hydrate the same pointer-owned ordinary `Checker.ontology_kernel` from an
  imported `.ontocache` after sourcecheck, instead of consulting the
  check-owned cache verdict surface after sourcecheck.
- [x] Add a fresh-symbol same-Checker hydration classifier proving the current
  same-Checker blocker is now after `hydrate_ok`: ordinary sourcecheck and
  pointer-owned hydration complete, but the positive kernel verdict still
  rejects from the same `Checker.ontology_kernel`.
- [x] Add a second same-Checker copy classifier proving that hydrating a local
  cache-derived checker kernel before copying it to the ordinary checker also
  reaches `copy_ok`, but still rejects the positive same-Checker kernel verdict.
- [x] Add a third same-Checker pointer-query classifier proving that querying
  the copied ordinary checker kernel through a pointer helper reaches `copy_ok`
  and then remains classified before a query verdict.
- [x] Add a same-Checker precheck classifier proving the value-threaded path
  reaches `collect_ok` and `hydrate_ok`, then remains classified when reading
  `class_count` from the collected checker's hydrated ontology kernel.
- [x] Add a stable `check::mod` same-sourcecheck sidecar for imported bounded
  `.ontocache` facts: after ordinary sourcecheck it accepts SNOMED subclass
  and PART subclass/disjoint verdicts while rejecting reverse verdicts.
- [x] Use that stable `check::mod` sidecar as a focused imported-cache
  `Knowledge<T>` proof-context authority: after ordinary sourcecheck it accepts
  bounded SNOMED and PART `field subclass_of Target` proof contexts from
  `.ontocache`, rejects a reverse SNOMED proof context, and rejects a disjoint
  PART proof context without mutating `Checker.ontology_kernel`.
- [x] Generalize that imported-cache `Knowledge<T>` sidecar resolver from a
  fixed SNOMED/PART name table to generic `PREFIX_<numeric-id>` names plus
  explicit `cache_slotN` names, so the proof-context bridge is keyed by the
  numeric IDs carried by `.ontocache` instead of ontology-specific symbol
  spelling.
- [x] Expose the imported-cache `Knowledge<T>` sidecar through a boot4-style
  source-path verdict API,
  `check_items_verdict_boot4_with_ontocache_sidecar(items, source_path)`, so
  the next ordinary frontend integration can call one checker-owned verdict
  surface instead of a diagnostic-only probe.
- [x] Route modular frontend checking for sources with
  `//@ ontology-side-table-cache: "..."` through that boot4-style sidecar API,
  then run the ordinary `check::knowledge_context` value-only validator so
  imported-cache ontology proof contexts can compose with static numeric and
  unit proof-context constraints in one `Knowledge<T>` surface.
- [x] Add a same-Checker direct shadow-field classifier proving that adding
  small cache fields directly to the ordinary `Checker` reaches `hydrate_ok`
  after sourcecheck, but the pointer surface still reads `shadow_count=0` and
  returns the empty-shadow reject verdict.
- [ ] Promote the stable sidecar into an actual ordinary checker authority
  surface or replace direct same-Checker ontology-kernel field mutation with an
  explicit merge/offset representation. Bounded `.ontocache` kernel rows
  currently assume local class slots starting at zero and are not yet a stable
  `Checker.ontology_kernel` mutation surface after `collect_items`.
- [x] Wire `self-hosted/check/ontology_side_table_cache.sio` into the
  bootstrap source concatenation and focused Knowledge bootstrap bundle before
  `check.sio`, so the modular source surface for the ontology cache is no
  longer omitted from bootstrap packaging.
- [x] Route the focused Knowledge bootstrap script through the canonical
  `bin/souc` wrapper by default, so `check`/`run` mode uses the same interface
  as the rest of the local acceptance gates instead of failing early on the
  richer `artifacts/omega` CLI shape.
- [x] Synchronize the focused Knowledge bootstrap's hand-built AST literals
  with the current parser/checker structs for `Expr`, `ExprList`,
  `KnowledgeTypeInfo`, `PolicyRequirementInfo`, and `Item`, eliminating the
  local `missing field in struct literal` failure family from that bundle.
- [x] Advance the focused Knowledge bootstrap bundle from source typecheck
  failure to `check` pass by adding bootstrap shims, removing tuple/boolean
  inference hazards from checker helpers, and filtering unrelated hypercomplex
  law-profile probes out of the narrow Knowledge gate.
- [x] Move the first focused Knowledge bootstrap type-wrapper witnesses onto a
  pointer-shaped checker lowering surface, so `Policy<T>`, `Contest<T>`,
  `Robust<T>`, shorthand `Knowledge<T>`, and the empty-policy
  `prove_robust` witness execute before the remaining runtime blocker.
- [x] Move the focused epistemic policy/contest/robustness/manifest witnesses
  through B15 onto pointer-shaped checker helpers: explicit `Contest<T>`,
  fail-closed plain/annotation checks, requirement satisfaction, robustness
  proof recording, validation manifest recording, and the focused metadata
  side-table facts now execute before the remaining runtime blocker.
- [x] Move the focused full-arity contest, contest-witness,
  decision-certificate, and deferral-certificate witnesses through B36 onto
  pointer-shaped checker helpers: full `Contest<T, Family, Policy>` metadata,
  scalar/rich contest witness projection, fail-closed contest-witness
  diagnostics, `Admissible<T>` decision certificates, `Deferred<T>`
  certificates, deferral-reason witness metadata, and transition wrapper
  lowering now execute before the remaining runtime blocker.
- [x] Move the focused transition action/monitoring witnesses through B43 onto
  pointer-shaped checker helpers: `commit_alternative` transition plans,
  transition-reason witnesses, monitoring policy wrappers,
  observed-transition metadata, and rollback-certificate metadata now execute
  in the focused bootstrap bundle.
- [x] Repair the focused Knowledge bootstrap runtime path: `bash
  scripts/bootstrap/run_knowledge_bootstrap_tests.sh` now reports
  `check exit: 0`, `run exit: 0`, and `Knowledge bootstrap tests: 43/43
  passed`.
- [ ] Wire imported `.ontocache` hydration into ordinary source-file
  check/compile execution instead of only the focused `check::check` probe.
- [x] Add checker-visible runtime proof-obligation counting for dynamic
  `Knowledge<T where {...}>` numeric field constraints, so dynamic evidence is
  no longer silently indistinguishable from fully discharged static evidence.
- [x] Add a pre-native executable runtime guard expansion for the first dynamic
  numeric lower-bound slice; generated Sounio helpers assert the proof
  obligation at runtime and then lower the proof-context type to `Knowledge<T>`
  for the existing backend.
- [x] Extend pre-native runtime guard expansion to conjunctive numeric
  lower-bound proof contexts such as `{ age >= 18, glucose >= 126 }`.
- [x] Extend pre-native runtime guard expansion to unit-suffixed numeric
  lower-bound proof contexts such as `{ amount >= 500 <mg> }` by comparing a
  dimensionless ratio against `1.0` after materializing a same-unit threshold.
- [x] Add current-source runtime guard coverage for internal validation-data
  unit labels such as `{ glucose >= 126.0 <mg_dL> }`, with the same
  no-clinical-authority boundary as the metadata/unit-label gates.
- [x] Add deterministic pre-native text `.guardcache` reuse for repeated
  `Knowledge<T>` runtime guard expansion, keyed by source bytes and expander
  bytes.
- [x] Broaden the pre-native runtime guard expander beyond lower-bound-only
  checks by generating executable upper-bound guards such as
  `{ score <= 100 }`, with positive and violating dynamic witnesses.
- [x] Add executable equality guards such as `{ repeats == 3 }`, matching the
  parser/checker proof-context surface for numeric equality constraints.
- [x] Preserve stable pre-native runtime guard diagnostics in the expanded
  source for representative lower-bound, conjunctive, upper-bound, equality,
  named-unit, and internal-label constraints without adding `IO` effects to
  generated guard helpers.
- [x] Add an optional deterministic runtime-guard diagnostic manifest artifact
  for pre-native guard expansion, keyed by the same input/expander hashes as
  the `.guardcache` path and listing guarded constraints as structured
  `type`, `field`, `op`, `threshold`, `unit`, and `constraint` TSV columns for
  audit.
- [x] Prove pre-native generated `Knowledge<T>` guard assertions lower through
  the existing native backend `assert`/trap path for satisfying and violating
  witnesses, with the violating native binary exiting 1. This is native
  execution coverage for generated guards, not direct native lowering of
  checker-visible proof obligations.
- [x] Bridge checker-visible runtime obligations to pre-native generated
  guards for the first lower-bound positive/reject pair by proving the original
  sources report nonzero obligations and the expanded sources report zero
  obligations after guard insertion and proof-context lowering.
- [x] Extend that runtime-obligation-to-guard bridge to call-boundary and
  assignment sites, with positive/reject witnesses that drain checker-visible
  obligations to zero after expansion and lower through the native `assert`/trap
  path.
- [x] Extend obligation-drain and native `assert`/trap coverage across the
  supported pre-native runtime guard families: conjunctive lower bounds,
  upper bounds, equality, unit-suffixed thresholds, and internal validation-data
  unit labels.
- [x] Add an opt-in `bin/souc` launcher bridge,
  `SOUNIO_KNOWLEDGE_RUNTIME_GUARDS=1`, for compile/build/run so the pre-native
  generated guard path can be invoked without manually calling the expander.
- [x] Add a source-level `//@ knowledge-runtime-guards` directive that enables
  the same launcher bridge for compile/build/run without requiring an
  environment variable.
- [x] Add a compiler-side raw-source scanner for
  `//@ knowledge-runtime-guards` that classifies pre-native expansion intent
  before ordinary lexing drops comments, and keeps direct native proof-obligation
  lowering explicitly marked as not ready.
- [x] Add a compiler-side `Knowledge<T>` runtime guard lowering-plan surface
  that joins directive intent, parser/checker semantic acceptance, and
  checker-visible runtime obligation counts while preserving the first
  obligation payload (`site`, `type`, `field`, `op`, `value`, `unit`) and
  mapping it to a staged backend-guard row (`compare_opcode`,
  `threshold_kind`, `trap_exit_code`) while leaving emitted backend guards
  explicitly at zero until direct native lowering exists.
- [x] Extend that lowering-plan staging from a single representative row to a
  real staged backend-guard row count for all checker-visible obligations in a
  conjunctive dynamic proof context, while still keeping emitted backend guards
  at zero until direct native lowering exists.
- [ ] Lower checker-visible runtime proof obligations into native backend
  guards/traps.
- [ ] Add `//@ ontology-bundle: "..."` as a compile-time bundle directive.
- [ ] Decide whether the directive calls the C importer through FFI or consumes
  a compact Sounio-native cache format.
- [ ] Hydrate the live checker-kernel path from imported native side-table
  `.ontocache` artifacts during actual compile/check execution.

Current gate:

```bash
make test-knowledge-context-phase2
make test-ontology-bundle-directive-native-scan
make test-ontology-cache-frontend-composition
make test-knowledge-unit-constraints
make test-knowledge-numeric-constraints
make test-knowledge-composite
make test-knowledge-context-static
```

## Layer 3: Unit Types and Dimensional Analysis

Goal: numeric types carry dimensions and units at compile time.

Target surface:

```sio
unit Mass = kg
unit Length = m
unit Time = s
unit Velocity = Length / Time
unit Energy = Mass * Length * Length / (Time * Time)

fn kinetic_energy(m: f64<Mass>, v: f64<Velocity>) -> f64<Energy> {
    0.5 * m * v * v
}
```

TODOs:

- [x] Parse base unit declarations.
- [x] Parse one-step derived unit expressions such as `unit velocity = m / s`
  in the current-source `lean_single` path.
- [x] Parse chained derived unit expressions such as
  `unit acceleration = m / s / s` in the current-source `lean_single` path.
- [x] Mirror chained derived unit expression source support into the modular
  parser/checker AST path.
- [x] Add an executable modular parser AST gate for chained derived units.
- [x] Add an executable modular checker gate for chained derived units.
- [x] Parse named numeric literal unit suffixes such as `200.0<mg>` and
  `300<mg>` in the current-source `lean_single` path.
- [x] Normalize registered unit dimensions to exponent vectors for builtin
  units.
- [x] Reject incompatible addition/subtraction.
- [x] Prove same-unit division cancels to a dimensionless scalar.
- [x] Preserve registered named dimensions for non-cancelling
  multiplication/division results in the current-source `lean_single` path.
- [x] Parse first-rung `f64<UnitExpr>` annotations in the current-source
  `lean_single` path, where `UnitExpr` is a registered unit identifier chain
  joined by `*` and `/`.
- [x] Preserve units through scalar multiplication.
- [x] Reject incompatible unit-typed function arguments.
- [x] Accept compatible unit casts and reject incompatible unit casts.
- [x] Require explicit conversion for same-dimension unit systems such as J and
  eV in the current-source `lean_single` path.
- [x] Connect LOINC and ChEBI-linked internal dimension-label metadata as
  validation data, not as checker authority.
- [x] Register validation-data internal dimension labels (`mg_dL`, `mmol_L`, `U_L`,
  `mm_h`) in the current-source `lean_single` unit table so metadata-backed
  labels participate as internal current-source unit identifiers with no
  built-in conversion factors. This does not prove clinical
  correctness, conversion safety, dosing safety, UCUM conformance, LOINC
  conformance, ChEBI conformance, or regulatory data-exchange readiness.
- [x] Mirror internal dimension-label registration into the modular
  `UnitRegistry`, proving `Knowledge<T>` unit proof contexts accept matching
  `mg_dL`, `mmol_L`, `U_L`, and `mm_h` labels and reject incompatible
  mass-concentration vs amount-concentration labels without making clinical,
  conversion, UCUM, LOINC, ChEBI, dosing, or regulatory-exchange claims.
- [x] Add a focused Unit Types Phase 1 gate.

Current gate:

```bash
make test-unit-types
make test-unit-types-derived
make test-unit-types-clinical-current-source
make test-ontology-unit-metadata
```

## Static Spine Umbrella Gate

Current gate:

```bash
make test-semantic-knowledge-spine
make test-ontology-cache-frontend-composition
```

This is the current executable 1->2->3 surface: generated ontology stubs and
typed bridges, pre-native ontology-bundle directive expansion with deterministic
text `.ontocache` reuse, unit/dimensional analysis, ontology-linked unit
metadata as validation data, static `Knowledge<T>` proof contexts, and
checker-visible runtime proof obligations for dynamic `Knowledge<T>` values,
plus pre-native runtime guard expansion for numeric lower-bound, upper-bound,
and equality comparison slices.
The umbrella now also runs the focused frontend-composition gate: the modular
frontend composes imported-cache ontology proof contexts with ordinary
value-only numeric/unit proof-context validation for files that declare
`//@ ontology-side-table-cache: "..."`, and the same gate proves a PART
imported-cache subclass proof context plus a PART disjoint rejection proof
context through that frontend path. It also proves additional numeric-ID
bundle shapes: QM accepts a subclass proof context and rejects the reverse
subclass proof context, while ALG, ChEBI, GO, HPO, and PHYS accept a generic
slot3-to-slot0 subclass proof context and reject the reverse proof context
through the same frontend path. LOINC now uses synthetic positive cache-slot
IDs for alphanumeric CURIE tails and proves its first bounded parent edge with
`cache_slot4 subclass_of cache_slot5` plus reverse rejection through the same
frontend path.
The K2
`check::check` bridge classifier is optional via
`bash scripts/ci/semantic_knowledge_spine_gate.sh --with-k2-classifier` and is
tracked as a separate modular-checker bridge blocker. This gate does not claim
runtime checks or native compiler `//@ ontology-bundle` live checker-kernel
side-table loading. The native `.ontocache` coverage is currently bounded
side-table artifact write/read rehydration plus source-level cache directive
import from a second file, checker-shaped cache verdicts, focused
`check::check` live-kernel hydration, and same-Checker blocker classifiers,
including direct shadow fields that still read empty through the ordinary
checker pointer and a precheck value-threaded classifier that reaches
`hydrate_ok` before failing at collected-checker kernel-state readback. It is
not yet live ordinary checker-kernel hydration.

## Layer 4: `Knowledge<T>` Proof Contexts

Goal: `Knowledge<T>` becomes a light dependent-type bridge for scientific
evidence.

Target surface:

```sio
fn treat(p: Knowledge<Patient where {
    diagnosis subclass_of Diabetes,
    fasting_glucose >= 126.0<mg_dL>,
    age >= 18
}>) -> i32 {
    1
}
```

TODOs:

- [x] Add AST storage for proof constraints.
- [x] Parse `field subclass_of OntologyClass`.
- [x] Parse numeric comparisons and optional unit suffixes.
- [x] Add checker-source support for static ontology constraints against struct
  field types.
- [x] Add a Stage1 checker-verdict AST gate for the same positive/negative
  surface.
- [x] Prove ordinary parser/AST retention through a focused Stage1 probe.
- [x] Prove ordinary Stage1 frontend acceptance/rejection without source-text
  preflight.
- [x] Prove `Knowledge<T where {...}>` proof contexts on return types with
  positive and disjoint negative witnesses.
- [x] Prove multiple static ontology constraints in one `Knowledge<T where
  {...}>` context with positive and negative witnesses.
- [x] Prove the boot4-safe `check::mod` verdict wrapper through the ordinary
  Stage1 verdict path.
- [x] Own the current boot4-safe proof-context validator in
  `check::knowledge_context`, imported by `check::mod`, with no local
  `check_mod_ast_*` duplicate.
- [x] Prove standalone imported `check::check` semantic support for static
  ontology proof contexts using the K2 semantic probe.
- [x] Check static unit constraints against the unit dimension checker for
  `Knowledge<T where { field >= value <unit> }>` witnesses.
- [x] Prove matching and incompatible derived-unit constraints through the
  ordinary `check::knowledge_context` semantic bridge.
- [x] Check static numeric proof-context constraints for numeric field/value
  shape through the ordinary bridge.
- [x] Prove static numeric threshold satisfaction when the struct field type is
  a refinement that implies the `Knowledge<T>` constraint.
- [x] Reject static numeric threshold constraints when the field refinement is
  too weak to imply the `Knowledge<T>` constraint.
- [x] Prove ontology, derived-unit, and numeric-refinement constraints compose
  as a conjunction inside one `Knowledge<T where {...}>` context.
- [x] Reject a composite `Knowledge<T>` context when any one of ontology, unit,
  or numeric-refinement constraints fails.
- [x] Add a static umbrella `Knowledge<T>` gate covering ontology, units,
  numeric refinements, and composite proof contexts.
- [x] Parse `Knowledge { ... }` value literals through the modular parser path
  used by focused K2 probes.
- [x] Check literal `Knowledge { value: Struct { ... } }` return values against
  numeric proof-context thresholds when the constrained field value is a
  compile-time integer or float literal.
- [x] Check literal local `let` and `var` bindings annotated as
  `Knowledge<Struct where {...}>` against numeric proof-context thresholds.
- [x] Check literal `Knowledge { value: Struct { ... } }` call arguments
  against `Knowledge<Struct where {...}>` parameter proof contexts.
- [x] Revalidate literal assignments into local variables annotated as
  `Knowledge<Struct where {...}>`.
- [x] Check unit-suffixed static value thresholds such as
  `field >= 1.0 <acceleration>` against literal `Knowledge` values.
- [x] Classify nonliteral constrained field values as the runtime-check gap
  rather than claiming they are statically proven.
- [ ] Prove the full `check::check` semantic support through the ordinary
  Stage1 verdict path after the `check::mod -> check::check` boot4 bridge is
  narrowed.
- [x] Extend static value-expression checking beyond direct return/local/call
  and simple assignment literals to simple generated `Knowledge<T>` values
  whose function body directly returns a literal `Knowledge` value, including
  direct parameter references supplied by literal call arguments.
- [x] Add a narrow numeric const-eval path for simple generated `Knowledge<T>`
  constructors whose constrained field is a `+`, `-`, `*`, or `/` expression
  over numeric literals and literal-backed parameters.
- [x] Resolve simple local `let`/`var` bindings inside generated
  `Knowledge<T>` constructors when those bindings are const-evaluable by the
  same narrow numeric path.
- [x] Resolve simple linear `=` assignments inside generated `Knowledge<T>`
  constructors before the direct return when the assigned expression remains
  const-evaluable by the same narrow numeric path.
- [x] Resolve simple linear compound assignments (`+=`, `-=`, `*=`, `/=`)
  inside generated `Knowledge<T>` constructors when the current local value and
  right-hand side are const-evaluable by the same narrow numeric path.
- [x] Evaluate narrow numeric helper calls inside generated `Knowledge<T>`
  constructors when the helper has a direct `return` expression and the helper
  call arguments are const-evaluable.
- [x] Resolve simple helper-local `let`/`var` bindings inside narrow numeric
  helper calls used by generated `Knowledge<T>` constructors.
- [x] Resolve simple helper-local linear assignments inside narrow numeric
  helper calls used by generated `Knowledge<T>` constructors.
- [x] Resolve literal-boolean `if true`/`if false` branches in generated
  `Knowledge<T>` constructors when the selected branch directly returns the
  generated value.
- [x] Resolve literal-boolean branch local flow in generated `Knowledge<T>`
  constructors when the selected branch mutates a tracked local before the
  direct return.
- [x] Resolve simple const-evaluable numeric branch conditions in generated
  `Knowledge<T>` constructors when the condition depends on literals and
  literal-backed parameters.
- [x] Resolve direct-return branch selection in generated `Knowledge<T>`
  constructors when the `if` condition is const-evaluable from literals and
  literal-backed parameters.
- [x] Resolve narrow helper branch selection when a generated `Knowledge<T>`
  constructor calls a numeric helper whose direct return is selected by a
  const-evaluable branch condition.
- [x] Resolve narrow helper-local branch flow when a generated `Knowledge<T>`
  constructor calls a numeric helper that mutates a tracked local through a
  const-evaluable branch before returning it.
- [x] Resolve narrow nested helper calls when a generated `Knowledge<T>`
  constructor calls a helper whose direct return calls another helper with
  literal, caller-parameter-threaded, or const-evaluable arguments.
- [x] Restore float literal value carry in the modular flat parser path so
  `field >= 1.0 <unit>` and values like `0.5` are validated numerically by the
  same static value gate.
- [ ] Extend generated `Knowledge<T>` checking beyond simple direct literal
  constructors, direct parameter-reference constructors, and narrow arithmetic
  expressions/local bindings/linear assignments/helper calls/helper-local
  bindings/helper-local assignments/literal-boolean branches/literal-boolean
  branch local flow/simple const-evaluable branch conditions/direct-return
  const-evaluable branch selection/narrow helper branch selection/narrow
  helper-local branch flow/narrow nested helper calls with caller parameter
  threading to general branch-aware flow and interprocedural constructors.
- [x] Generate a pre-native executable runtime-check slice for dynamic fields:
  numeric proof contexts expand to Sounio `assert(...)` helpers, including
  conjunctive lower-bound constraints, upper-bound constraints, equality
  constraints, unit-suffixed thresholds, and internal validation-data labels
  such as `mg_dL`.
- [ ] Lower dynamic proof obligations into native backend guards/traps instead
  of relying on pre-native source expansion.
- [ ] Broaden dynamic runtime checks beyond the current numeric comparison
  slice once the native guard/trap path exists.
- [ ] Preserve proof evidence or runtime diagnostics explaining which
  constraint was satisfied or failed across the eventual native guard/trap
  path.
- [x] Include the pre-native runtime guard expansion gate in the static spine
  umbrella as the first executable dynamic-check rung.
- [ ] Extend the umbrella `Knowledge<T>` gate to native backend dynamic-check
  generation once guard/trap lowering exists.

Current focused gates:

```bash
make test-knowledge-context-phase2
make test-knowledge-unit-constraints
make test-knowledge-numeric-constraints
make test-knowledge-composite
make test-knowledge-static-values
make test-knowledge-context-static
```

## Layer 5: Federated Reasoning and Caches

Goal: large ontology graphs remain usable without making the compiler slow or
fragile.

TODOs:

- [ ] Store transitive closures in bundle shards.
- [ ] Add query planning across ontology families.
- [ ] Cache compile-time ontology query results.
- [ ] Keep cache invalidation deterministic.
- [ ] Keep generated/cached artifacts auditable.

## Risk Register

- 64-class limit per file: keep generated bundles small or sharded.
- Cross-module ontology persistence: pre-native text `.ontocache` reuse exists,
  and bounded native side-table `.ontocache` write/read rehydration plus a
  source-level cache directive exist; live checker-kernel hydration from those
  native caches remains future work. A direct shadow-field experiment is also
  classified and still reads empty through the same ordinary checker pointer.
- Bootstrap drift: update snapshots only after focused source gates are green.
- Runtime checker debt: separate static proof success from dynamic check
  generation.
- Generated artifact drift: fail CI once regeneration is deterministic enough.
- Scope drift into Python: keep Python limited to dataset preparation or legacy
  tooling; PL ingestion and checking should use C FFI or Sounio-native paths.
