<!-- docs:meta
topic_id: repo.docs.spec.2026-10.00-foundations-and-conventions
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.2026-10.00-foundations-and-conventions
-->

# Foundations and conventions

This source-research edition describes source-language interfaces and records their
implementation boundaries at Sounio revision
`f5b2c5850be9f3d3995719ffcade44a6d851658a`. Its document edition is
`2026-10`; that identifier is independent of compiler release numbers.
The author of Sounio is Demetrios Chiuratto Agourakis.

The reference is organized for language implementers, scientific programmers,
and reviewers. Each subject has language rules, implementation observations,
and evidence references. A referenced test is a reproducible witness to run;
its presence is not a record that it passed during preparation of this edition.

## 1. Scope and authority

`SPEC-001` — `editorial-contract`. **Source identity.** A statement about this edition's implementation
means the named revision, engine, target, compiler artifact, and relevant
library. A later installed binary or an uncommitted workspace is another
implementation state. The edition manifest records the research baseline,
not an executed compiler profile. Receipts also identify the CLI adapter,
effective checker/backend entrypoints, and stage actually reached.

`SPEC-002` — `editorial-contract`. **Contract and observation.** An adopted
language contract expresses a requirement on a program or an implementation.
Its authority comes from the identified adoption or explicit ruling, not from
a compiler satisfying it. A source observation describes what the inspected
implementation does and imposes no new obligation. An implementation defect
does not become a language rule through repetition. Where the intended rule
has not been settled, the text records an open decision and the available
witnesses.

Edition reporting requirements are a separate authority domain. They govern
how this reference and its receipts are written; they do not define accepted
programs or runtime semantics. The editorial conventions in these foundations
and the conformance chapter are authorized by Demetrios Agourakis's direct
session direction of 2026-10-16. They are not represented as language rules
adopted at the research baseline. The dated ADR009 working direction in
section 8 records its own provenance and continues to govern this work.

The standing column in the companion evidence tables has these meanings:

| Standing | Meaning |
|---|---|
| `specified-contract` | An existing adopted language or scientific semantic requirement, with the adopting source or ruling identified. Enforcement and conformance remain separate. |
| `editorial-contract` | A requirement for edition methodology, reporting, or session-authorized working procedure. Its provenance is stated; it does not ratify source-language semantics. |
| `source-observed` | Behavior or representation recovered from the inspected source. This is descriptive, including when the behavior is defective. |
| `library-contract` | A named library API and its preconditions and limits. Library scope alone does not establish adoption as language semantics. |
| `target-contract` | Behavior restricted to a named engine, ABI, target, or runtime route. Target scope alone does not ratify that behavior. |
| `open-decision` | The implementation or design evidence does not establish a single language-wide rule. |

Standing and scope are different dimensions. A library or target rule states
whether it describes current implementation behavior or cites a separately
adopted requirement. Neither a filename, a gate, a table label, nor repeated
implementation behavior constitutes adoption. Source/symbol references must
support the kind of claim attributed to them; path existence alone establishes
neither semantic authority nor conformance.

These labels do not promote the existing public-claim registry. Its `stable`,
`validated_research`, `prototype`, and `stale_conflicting` distinctions continue
to describe the permitted scope of implementation claims.

`SPEC-003` — `editorial-contract`. **Requirement words.** In a specified contract, *shall* and *must*
state an obligation, *shall not* and *must not* prohibit an action, *may*
allows an alternative, and *should* expresses a recommendation. In an
observation, the present tense reports the implementation. In an
`editorial-contract`, requirement words govern reporting or working procedure
only. A descriptive `source-observed` rule does not acquire prescriptive force
through an imperative; an intended obligation is stated separately with its
authority. Neither a rule identifier nor the word *shall* establishes
implementation or enforcement. An example does not establish a more general
requirement than its enclosing rule.

`SPEC-004` — `editorial-contract`. **Resolution of disagreement.** Results
from different stages or implementation profiles are distinguished first.
When an existing adopted contract settles the behavior, a conflicting
implementation result is a defect against that contract; its correction does
not need the same semantic choice to be made again. Where no adopted contract
determines the behavior, alternatives remain an open language decision
requiring the appropriate founder ruling and evidence. Until that decision,
an engine-specific result supports only its scoped observation.

An adopted requirement remains the referent when enforcement fails or a
conformance witness is absent. For example,
`docs/spec/S06_EFFECTS_ROWS.md` section 6.0 records the founder's ruling that
function types carry effects while reporting no conformance test for it.
Missing enforcement is an implementation/coverage gap, not withdrawal of that
rule. Conversely, passing a test does not adopt an unresolved alternative
such as row polymorphism.

The applicable project sources are `FOUNDER_INTENT.md`, `AGENTS.md`,
`docs/internal/concepts/SEMANTIC_LANE_CONTRACT.md`,
`docs/spec/E2E_SPECIFICATION_FRAME.md`, and
`docs/serious-language/public-claim-registry.v1.tsv`.

The August frame and `docs/spec/S17_CONFORMANCE_SUITE.md` are dated research
and decision history. Their counts, feature-absence statements, execution
observations, section statuses, and CI census do not describe this edition
unless re-established at its named baseline. Explicit founder rulings remain
identified individually rather than inferred from those measurements.

The frame's original both-engine promotion criterion and this edition's
profile-scoped evidence have not been reconciled by a new ruling here. This
edition neither waives that criterion nor claims it is satisfied. A scoped
receipt can establish a profile observation without authorizing a
language-wide or `Claim-ready` promotion. Public-claim status and existing
promotion requirements remain separate from semantic adoption and structural
validation of these documents.

## 2. Program and implementation model

A *source unit* is a source file read by the lexer and parser. A *module*
provides a namespace and exported declarations. An import extends the program's
source graph through the resolver. A *program* is the entry source unit together
with the source units, library definitions, and runtime services selected for
its translation and execution.

Translation has distinct stages: source decoding and tokenization, parsing,
name and type checking, effect and ownership checking, lowering, target
emission, and artifact construction. Some specialized routes add expansion,
validation wrappers, GPU translation, or external runtime steps. Reachability
through the selected driver determines which of these stages actually runs.

`SPEC-005` — `editorial-contract`. **Stage evidence.** A successful parser verdict, including its
error state and lexer-capacity checks, establishes syntax acceptance on that
route. A recovered AST or reaching the end of input does not establish that
verdict. A checker result establishes the decisions reached by that checker. An emitted
artifact establishes emission. Execution with an asserted observation
establishes that observation. Each later claim requires its own evidence;
acceptance at an earlier stage does not discharge a later obligation.

The main executable path is rooted in `self-hosted/compiler/main.sio`.
`scripts/lib/resolve_souc.sh` and `scripts/lib/resolve_madaros.sh` determine the
public compiler route. The bootstrap implementation in
`self-hosted/compiler/lean_single.sio` is a separate engine. The IR and target
chapters describe reachable lowering and emission; the existence of another
IR module does not itself place that module on this path.

## 3. Semantic notation

The following notation provides a common vocabulary across chapters. It does
not assert a complete mechanized calculus for the implementation.

| Symbol | Meaning |
|---|---|
| `Γ` | Names, declarations, type variables, and assumptions available to static checking. |
| `Δ` | Ownership or resource-usage assumptions, where the checked feature tracks them. |
| `ε` | A function's declared or inferred effect information. Its concrete representation is chapter-specific. |
| `τ` | A language type. |
| `σ` | An abstract execution state, including relevant storage and runtime state. |
| `v` | A value in the semantic domain of its type. |
| `Γ; Δ ⊢ e : τ ! ε` | Expression `e` has type `τ` and effect information `ε` under the stated assumptions. |
| `⟨e, σ⟩ ⇓ ⟨v, σ′, ω⟩` | Execution terminates with value `v`, state `σ′`, and observable trace `ω`. |

A rule using this notation must state the subset it models. In particular,
type compatibility, refinement implication, confidence ordering, effect
subsumption, and physical-unit equivalence are different relations.

The displayed evaluation judgment covers terminating executions in a stated
subset. It is not a totality, progress, determinism, or compiler-correctness
claim. A model or conformance claim states how it treats divergence, stuck
states, defined traps, resource refusal, external interactions, and
nondeterministic traces. An unmodeled outcome is not silently undefined
behavior or an accepted execution. Chapter 09 records the outstanding formal
correspondence obligations.

`SPEC-006` — `editorial-contract`. **Observable behavior.** An execution claim names the observations
it constrains: returned value, process exit status, output, storage changes,
runtime failure, external interaction, or scientific evidence channel.
Preservation of one observation does not automatically preserve the others.

For a deterministic scalar example, equality of exit status alone is weak:
two programs can exit successfully while printing different values. For a
scientific example, equality of printed central values can coexist with lost
variance, changed provenance, or different parenthesization. Its observation
contract must include the channels that matter to that example.

## 4. Outcomes and diagnostics

The reference distinguishes the following outcomes:

| Outcome | Meaning |
|---|---|
| Accepted | The requested stage completed with its success contract. |
| Ill-formed | The program violates a settled syntactic or static rule. |
| Unsupported | The selected implementation or target refuses a requested feature or operation. |
| Runtime failure | Execution reaches a trap, panic, rejected precondition, or external-service failure. |
| Resource limit | The request exceeds a stated implementation or execution bound. |
| Divergence | A modeled nonterminating computation; a timeout alone does not establish it. |
| Stuck state | No modeled next step or final outcome applies; this is not automatically language-authorized undefined behavior. |
| Unresolved | The available evidence does not settle the language rule or the result. |

`SPEC-007` — `editorial-contract`. **Refusal evidence.** A negative witness must identify the violated
rule and the expected diagnostic or observable refusal. A signal, timeout,
missing dependency, or unrelated parse error cannot establish enforcement of a
type, ownership, unit, or scientific-boundary rule.

An unspecified detail is a documented choice the specification leaves open.
An implementation-defined detail requires a named implementation's account.
Undefined behavior is a language-level category only when a rule explicitly
defines it. This edition does not introduce an implicit permission to discard
semantics merely because a backend lacks a lowering.

## 5. Scientific semantic invariants

`SPEC-008` — `specified-contract`. **Distinct information channels.** Computed value, arithmetic
error, measurement uncertainty, confidence, ignorance, computational provenance,
and physical causality have different meanings. A conversion or lowering that
combines them requires an explicit contract identifying the information lost
or preserved.

`SPEC-009` — `specified-contract`. **Order and precision.** The legality of optimization depends on
the operand domain and its contract. Associativity cannot be assumed for
non-associative algebra or floating arithmetic with observable rounding.
Representing an unsupported precision in a narrower format changes the
semantics and requires an explicit, observable policy. A declaration of an
effect or provenance tag does not create a physical mechanism.

These requirements preserve the founder's existing scientific direction.
They do not claim a universal optimizer proof or a complete enforcement
mechanism. The numerical, epistemic, and execution chapters identify the
implemented checks and remaining interfaces.

## 6. Reading the evidence tables

Each chapter's `*-evidence.tsv` maps a stable rule identifier to a primary source
path, a source symbol, positive and negative witness paths where available,
and the remaining gap. A dash means that the cell has no established reference.
It does not mean that the obligation is unnecessary or that the feature passed.

For an `editorial-contract`, a primary source may be `-` when the gap records
the direct session authorization, date, and reporting/procedural scope. Such a
row does not pretend to cite a baseline repository adoption. The absence of
that repository citation does not suspend the user's authorized working rule.
For an adopted semantic requirement, the adopting source or ruling remains
necessary; an editorial label cannot supply missing semantic adoption.

`SPEC-010` — `editorial-contract`. **Evidence completeness.** A complete implementation-conformance claim for a rule requires
its source identity, relevant positive and negative outcomes, target/engine
profile, observation contract, and execution receipt. Structural validation of
this reference verifies the catalog and paths; it does not run the compiler,
prove the prose, or confer language-wide conformance.

The conformance chapter gives the receipt format, current runner semantics,
limitations of existing cases, and the distinction between a bootstrap fixed
point and source-language conformance.

Most chapter fixtures remain unrun by their source-research authors. The
separate [bounded conformance receipt](15-bounded-conformance-receipt.md)
records Slurm job `14573` on 2026-10-16 and the existing manifest's
`total=16 passed=16 failed=0` result using the committed Linux x86-64 Madaros
prebuilt, SHA-256
`125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f`.
Its compiler was not rebuilt from the baseline source. That receipt applies
to its named inputs, profile, and assertions; it does not validate all prose,
chapter rows, scientific properties, formal models, or target behavior.

## 7. Upstream reference practice

The organization uses the technical practices visible in the
[C++ working draft sources](https://github.com/cplusplus/draft),
[Rust Reference](https://doc.rust-lang.org/reference/),
[F# language specification](https://fsharp.org/specs/language-spec/),
[Koka reference](https://koka-lang.github.io/koka/doc/book.html),
[LLVM Language Reference](https://llvm.org/docs/LangRef.html), and
[MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/).
The useful practices are stable rule labels, explicit grammar conventions,
definitions before use, static and dynamic rules, target boundaries, and
testable conformance obligations. Sounio's actual semantics come from its own
approved contracts and evidence.

## 8. Session direction: ADR009

`SPEC-011` — `editorial-contract`. **Numerical calculation and verification
protocol.** Provenance: direct user instructions from Demetrios Agourakis,
reaffirmed for this specification work on 2026-10-16. This section records the
full current working protocol, including the dated updates below; it does not
claim that this complete policy was a repository ADR adopted at commit
`f5b2c5850be9f3d3995719ffcade44a6d851658a`. Its editorial standing identifies
working-procedure provenance, not permission to disregard the user's rule.

1. Every numerical calculation, recalculation, or verification produced for
   Demetrios uses C++23, F#, Q#, F*, Futhark, Koka, or the MLX framework.
2. Python and Rust are prohibited by default; Julia is explicitly excluded.
   Python is not used as a reference, quick cross-check, or auxiliary numerical
   script except within the MLX authorization in item 7.
3. Theorems and proofs belong in Lean 4. Calculation uses a language or
   framework authorized in item 1; a computation is not relabeled as a proof.
4. The rule also applies to numerical recalculation in peer review of
   third-party articles, not only the user's own manuscripts.
5. When an executable supplement has multiple implementations, the canonical
   agreement references are C++ using exact rationals or another appropriate
   representation, and Sounio. A legacy Python implementation is excluded from
   the supplement and verification note. An MLX implementation authorized by
   item 7 may be included without replacing those canonical references.
6. If item 1 cannot be met, disclose that before executing the calculation;
   do not silently substitute Python outside the MLX exception.
7. MLX is authorized for numerical calculation, recalculation, and verification,
   including its Python interface and the data preparation/conversion and test
   harness needed to execute and verify MLX operations. This exception does
   not authorize independent numerical calculation in Python, NumPy, SciPy,
   or other frameworks. The remaining ADR009 requirements continue to apply.
   Demetrios explicitly authorized this exception on 2026-10-04.
8. Demetrios canonically included Q# among the authorized languages on
   2026-10-06 and explicitly reaffirmed F# in the same authorization. That
   update does not authorize independent Python or Rust numerical
   implementations, include Julia, change Lean 4's proof role, or replace the
   canonical C++ and Sounio references in item 5.

`docs/decisions/adr-009-verified-foreign-reference.md` addresses a separate
repository oracle-class policy. Its earlier language list and authority model
are not substitutes for this complete session direction. A metadata checker
or harness does not become an authorized independent numerical oracle merely
because it prepares or reports a result.

## 9. Semantic lane declaration

```text
Semantic-Lane-ID: language-specification-2026-10
Owner: codex-spec-20261016, under Demetrios Chiuratto Agourakis
Concept-IDs: SOUNIO-PRECISION-PRESERVATION; SOUNIO-NONASSOCIATIVE-ORDER;
  SOUNIO-ZERO-PROVENANCE; SOUNIO-KNOWLEDGE-BOUNDARY-PROVENANCE;
  SOUNIO-SCIENCE-RESEARCH-BOUNDARY
Intent-Preserved: explicit preservation and testing of scientific information
Transformation: specification recovery and evidence indexing; no compiler semantic rewrite
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: source research and structurally checked specification coverage
Claims-Forbidden: complete runtime conformance; general soundness; all-target parity;
  new clinical or empirical validation; ratification of contested semantics
Assumptions: source baseline and external reference identities are stated
Write-Set: docs/spec/2026-10/**; docs/spec/LANGUAGE_SPECIFICATION.md;
  docs/README.md; scripts/docs/validate_language_spec.mjs; review receipts
Read-Set: repository source, tests, gates, formal models, approved concept contracts
Positive-Witness: source references and existing positive fixtures in chapter catalogs
Negative-Witness: existing refusal fixtures; structural-validator mutation checks
Acceptance-Gate: node scripts/docs/validate_language_spec.mjs; docs registry check;
  independent chapter and mathematical review with limitations recorded
Integration-Target: protected main through reviewable changes
Authoritative-Only-If: replacing the earlier reference requires reviewed source
  mappings, structural gates, and explicit integration approval; none for
  changing source-language semantics, because this edition adopts no replacement
  semantics and does not confer new authority on an implementation
Semantic-Authority: adopted contract or dated explicit ruling, referenced for
  the affected rule; source observations do not ratify unresolved alternatives
Implementation-Conformance: separate claim requiring the named profile,
  discriminating executed witnesses, observation contract, and execution receipt
Public-Claim-Status: governed separately by the existing claim registry
Edition-Status: source research; no language-wide certification or conformance promotion
```

The concept meanings and statuses in
`docs/internal/concepts/registry.tsv` remain the authority for their named
contracts. This writing lane introduces no replacement concept or status
promotion.
