<!-- docs:meta
topic_id: repo.docs.spec.2026-10.16-specification-authority-review
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.2026-10.16-specification-authority-review
-->

# Specification authority and conformance review

Review date: 2026-10-16.
Source baseline: `f5b2c5850be9f3d3995719ffcade44a6d851658a`.
Verified baseline tree: `25a06aafc245efec6745e72e2fa49e12c7436ff0`.

Reviewed working-tree drafts:

- `docs/spec/LANGUAGE_SPECIFICATION.md`.
- `docs/spec/2026-10/edition.json`.
- `docs/spec/2026-10/00-foundations-and-conventions.md` and `00-evidence.tsv`.
- `docs/spec/2026-10/09-conformance-diagnostics-and-compatibility.md` and
  `09-evidence.tsv`.

The source baseline identifies the implementation being researched, not the
uncommitted text of these drafts. This review also inspected the relevant
launcher, compiler dispatch, checker entrypoints, suite runner, bounded
conformance runner, CI workflow, historical specification frame/S17, and
existing authority documents. The effects chapter's source findings supplied
concrete cross-checks. No compiler, runtime fixture, gate, numerical oracle,
or Lean build was executed. This report is the only file written for this
review; central documents and catalogs remain owned by the parent lane.

## Assessment

The title's explicit **source research edition**, the manifest's
`standing: source-research`, and its
`compiler_witness_execution: not-refreshed-by-this-edition` are appropriate.
The manifest's source revision and tree match Git. The central text already
distinguishes emission, execution, and formal-model results. Chapter 09
correctly dates the August CI census and does not claim new passing tests.
Those are not findings to reverse.

There is nevertheless a blocking ambiguity in the authority rule if the book
is to serve as the referee for correct versus incorrect implementations.
Additional major corrections concern inherited historical policy, effective
command stages, engine classification, and explicit formal obligations.
The blocker is about **normative use** of the edition; it does not prevent
circulating a clearly labeled source-research draft.

## AR-01 — BLOCKER: semantic authority is conditional on successful enforcement

**Affected locations:** chapter 00, `SPEC-002`, `SPEC-004`, and the semantic-lane
field `Authoritative-Only-If`; central reference, introduction paragraph
beginning “An implementation profile identifies”; manifest, edition standing.

The semantic-lane declaration currently says:

```text
Authoritative-Only-If: settled rule plus appropriate executed conformance evidence;
  founder ruling where semantic alternatives conflict
```

That merges the authority of a requirement with evidence that an implementation
satisfies it. The distinction is operationally necessary: a settled rule must
remain the referent when a compiler violates it or no conformance case has yet
been run. The effects chapter supplies an existing example, rather than a
new philosophical preference. `docs/spec/S06_EFFECTS_ROWS.md`, section 6.0,
records the founder's normative ruling that function types carry effects while
explicitly stating that no conformance test existed for that ruling.
Current `self-hosted/check/check.sio::extract_effect_ids` can still discard
unknown/user names or overflow in function types. Missing enforcement does not
erase the already adopted requirement.

The central introduction compounds this with the blanket sentence “Engine
disagreement requires a semantic decision with its own evidence.” Some
disagreements require a decision; others are defects against an existing
decision, or concern different stages/profiles. A fresh founder ruling is not
needed to decide again that already forbidden information loss is a defect.

**Exact suggested replacement for that introduction sentence:**

> When an existing adopted contract settles the behavior, a conflicting
> implementation result is a defect against that contract. Where no adopted
> contract determines the behavior, the alternatives remain an open language
> decision requiring the appropriate ruling and evidence. Results from
> different stages or implementation profiles must first be distinguished.

**Exact suggested replacement for the lane field:**

```text
Semantic-Authority: an adopted contract or dated explicit ruling, referenced for
  the affected rule; source observations do not ratify unresolved alternatives
Implementation-Conformance: a separate claim requiring the named profile,
  discriminating executed witnesses, observation contract, and execution receipt
Public-Claim-Status: governed separately by the existing claim registry
Edition-Status: source research; no language-wide conformance promotion
```

**Additional correction in `SPEC-002`:** explicitly distinguish source-language
requirements from project/reporting requirements. A `specified-contract` row
citing a workflow policy is not thereby a typing or operational rule. State
that a failed or absent implementation witness affects enforcement/claim
status, not the existence of a separately adopted obligation.

**Acceptance condition:** a reader can classify a violation of S06 section
6.0 without needing another semantic decision, while an unresolved choice
such as open-row polymorphism remains unresolved. The manifest must continue
to advertise source research, rather than imply this review ratified any new
language-wide semantics.

## AR-02 — MAJOR: historical S17/frame material re-enters the authority chain without reconciliation

**Affected locations:** chapter 00, `SPEC-004`, the list of applicable project
sources, and semantic-lane acceptance wording; chapter 09, `CONF-002` and
its S17 paragraph; central appendix on history.

Chapter 09 correctly calls the August engine census historical. However,
chapter 00 simultaneously invokes
`docs/spec/E2E_SPECIFICATION_FRAME.md` as an applicable authority without
stating which dated measurements or promotion rules remain operative.
That frame is explicitly `Garden`, says it is not the specification, repeats
the old S17 epistemic-surface account, and requires both engines to be run
before `Executable`, with nothing `Claim-ready` on one engine. The current
edition instead discusses profile-scoped conformance and “appropriate”
executed evidence without reconciling that inherited promotion criterion.

The stale material is not limited to job counts. S17 section 17.3 says the seed
has no annotation-component machinery and that an annotated `Knowledge` case
can only fail to parse on one engine. At the fixed current baseline,
`self-hosted/compiler/lean_single.sio::scan_type` contains the `Knowledge`
branch that handles selected `epsilon`/Unicode-epsilon metadata, recognizes a
`ValidUntil` shape, and accepts some identifier tags while discarding their
meaning. This is **partial, different handling**, not evidence of full parity;
it directly defeats reuse of the absolute “no annotation machinery” claim as
a current source description. Absence of the modular AST type names in a
single-file compiler is not proof of absence of analogous syntax handling.

**Exact suggested addition immediately after the chapter 00 authority list:**

> The August frame and S17 are dated research and decision history. Their
> source counts, feature-absence statements, execution observations, and CI
> census do not describe this edition unless re-established at its named
> baseline. Existing explicit founder rulings remain identified individually.
> The relationship between the frame's both-engine promotion criterion and
> this edition's profile-scoped evidence is unresolved unless a dated ruling
> explicitly reconciles them. This source-research edition claims neither
> engine-wide nor language-wide conformance by itself.

**Exact suggested expansion of chapter 09's historical paragraph:**

> Both the August engine census and S17's Knowledge-surface descriptions are
> historical. Current seed `scan_type` recognizes selected annotation forms,
> sometimes discarding metadata; this is distinct from modular parsing and
> enforcement. Current workflow source identifies intended routes, and
> completed artifact-specific receipts establish their observed outcomes.

Do not silently discard an existing founder promotion requirement merely to
make the new profile table consistent. Record the policy conflict and seek
the ruling only if a promotion actually depends on changing it.

**Acceptance condition:** no path through the central authority references
turns S17's old zero-machinery claim, old handler-erasure observation, or old
CI census into present semantics; the book states whether its conformance
claims are profile observations or promotions under the older criterion.

## AR-03 — MAJOR: `check`/`compile-fail` aliases do not identify the stage that actually decided refusal

**Affected location:** chapter 09, `CONF-003`, especially “Those modes
therefore establish checker refusal.” This also limits the central reference's
promise that chapter 09 documents actual runner behavior.

The bounded runner really maps both `check-fail` and `compile-fail` to
`("$SOUC_BIN" check "$source")`; that observation is correct. The subsequent
inference about the effective stage is too broad across the allowed profiles:

| Selected route | What the source actually does for `check` |
|---|---|
| Default modular `bin/madaros` | `_run_raw_check` invokes raw `--check`, reaching modular preflight. |
| Explicit `lean_single` through `bin/souc` | The seed `check` branch compiles to a temporary output and removes it; it is not a checker-only implementation. |
| `scripts/ci/souc-native-wrapper.sh` in legacy raw mode | The `check` branch calls `souc_compile` into a temporary artifact; modular raw mode instead invokes `--check`. |
| Rebuilt ontology wrapper | Its generated principal verdict can be parser-only, separate from diagnostic checking and fallback compilation; wrapper identity must not be equated with checker identity. |

The last case follows from
`scripts/ci/build_ontology_validation_souc.sh` and
`scripts/ci/ontology_validation_driver.sio`; the independently prepared
`11-library-formal-review.md` also localizes the synthetic-output fallback.
That report is corroborating context; the builder and driver are the source
authorities. No wrapper execution was performed here.

**Exact suggested replacement for `CONF-003`'s opening paragraph:**

> The requested command, selected adapter, resolved raw engine, and reached
> entrypoint determine the stage actually tested. The bounded manifest's
> `check-fail` and `compile-fail` spellings are command aliases: both request
> `check`. On the default modular check route that requests parse/check
> preflight. A legacy seed adapter can perform full temporary compilation, and
> a validation wrapper can supply a different principal verdict. A nonzero
> result establishes the rule's intended refusal only when the underlying
> stage and identifying diagnostic have been established for that profile.

**Required nearby addition:** include the stage table and state that wrapper
generated messages or copied expected-output annotations are not independent
checker/execution evidence. A requested operation name is not an effective
stage receipt.

The converse false finding must also be avoided: current raw Madaros `run` is
**not** merely compile-only. `compiler/main.sio::run_native_v2_compile_mode`
sets `execute_after_compile` when the original mode is `run` and calls
`native_v2_run_elf_and_exit` after successful emission. Sharing the compile
handler with `build` is not proof that execution is absent. Public wrapper
`compile|build` aliases and raw `--native-compile` versus
`--native-v2-compile` likewise need their particular dispatch, not conclusions
from their spelling.

**Acceptance condition:** a seed emission refusal cannot be counted as a
modular type-checker refusal merely because the manifest mode is
`check-fail`; each report identifies its effective deciding stage.

## AR-04 — MAJOR: engine classification and engine-specific skips are not trustworthy receipts

**Affected locations:** chapter 09, implementation profiles and `CONF-002`;
central reference, introduction; foundations, stage evidence; manifest's
interpretation as an implementation profile.

The generic requirement to name the actual engine is correct, but the book
does not explain that the main suite's own attribution can be wrong.
`scripts/run_sio_test_suite.sh` delegates to
`scripts/dev/run_sio_test_suite.sh`. The latter initializes
`SOUNIO_ENGINE_KIND` by local-artifact existence and one exact spelling of
`SOUNIO_SOUC_ENGINE`, instead of using the launcher's final resolved engine.
Its `requires` filtering repeats additional heuristics.

These are source-derived counterexamples, **not executed outcomes**:

| Configuration | Source-derived attribution problem |
|---|---|
| No executable local `artifacts/self-hosted/madaros`, no explicit override, committed or host Madaros available | Harness selects the `lean_single` banner, although `bin/souc` can resolve Madaros. |
| Local Madaros exists; `SOUNIO_SOUC_ENGINE=lean-single` or `leansingle` | Public launcher accepts these seed aliases, but the banner's comparison recognizes only `lean_single`; local presence can leave the banner naming Madaros. |
| Explicit `SOUNIO_TEST_SOUC_BIN` | Banner says `explicit`; that describes selection origin, not whether the engine is Madaros or the seed. A shebang distinguishes an adapter file, not its underlying semantic engine. |
| `requires: madaros` | Admission depends on `SOUNIO_MADAROS_AVAILABLE`, a declaration of availability, not proof that the selected executable is Madaros. |
| `requires: lean_single` | The filter considers the availability flag and local artifact presence; it does not uniformly apply the launcher's alias/override resolution. |

This affects which witnesses ran, not just the human-readable banner. In
particular, the effects chapter's E246/Mut refusal fixtures carry
`requires: madaros`; their mere presence in a suite selection does not imply
they were exercised by the corresponding checker.

**Exact suggested addition following `CONF-002`:**

> The suite's `harness: engine=...` banner and engine-availability variables
> are observational metadata, not authoritative engine identification. At
> this baseline the banner and `requires` filters use heuristics that differ
> from launcher resolution, including accepted seed aliases and committed or
> host Madaros fallback candidates. An `explicit` label identifies an override,
> not an engine. A conformance receipt must name the resolved raw executable,
> its identity, the CLI adapter, and the effective checker/backend entrypoints,
> and must record the engine-specific skipped cases and their reasons.

The chapter should also distinguish `selection_origin`, `engine_family`,
`artifact_identity`, and `cli_adapter` instead of treating them as one engine
enumeration. The edition manifest records a research baseline, not evidence
that a runtime profile was resolved or executed.

**Acceptance condition:** the source-derived configurations above cannot be
reported as a named-engine conformance run using the banner alone. Repairing
the underlying harness belongs to its separate owning implementation lane.

## AR-05 — MAJOR: the formal-obligation inventory is implicit rather than traceable

**Affected locations:** foundations, semantic notation and outcomes; chapter
09, `CONF-011`; central reference, sections 16–17; edition's interpretation as
a serious specification reference.

The drafts properly disclaim a complete mechanized calculus and require model
and assumptions for a proof claim. They do not, however, enumerate which
semantic correspondence obligations remain unestablished. The distinction is
important for the current defects: function-type effect collection can lose
identities; in-place and by-value checker routes differ; handler lowering has
no operational case; async markers can become ordinary blocks; the Lean
effect handler model preserves an already available payload while masking
metadata. A general instruction to cite a theorem does not identify which
of these production correspondences a proof or witness must address.

The single terminating big-step judgment in chapter 00 also has no companion
statement about divergence, stuck execution, nondeterministic traces, resource
exhaustion, or whether an external boundary is inside the modeled semantics.
The outcomes table distinguishes several refusal categories, but not these
semantic outcomes. This is a scope omission, not evidence of an existing
soundness defect in every rule.

**Exact suggested insertion after chapter 00's notation table:**

> The displayed evaluation judgment describes terminating executions in a
> stated subset. It is not a totality, progress, determinism, or compiler
> correctness claim. Any model or conformance claim must state how divergence,
> stuck states, defined traps, resource refusal, external interactions, and
> nondeterministic behavior are treated. An unmodeled outcome is not silently
> undefined behavior or an accepted execution.

**Exact suggested new inventory for chapter 09 after `CONF-011`:**

| Obligation family | Required scope and correspondence | Current edition status |
|---|---|---|
| Static preservation / safe progress | Named core subset, typed state, ownership/resource assumptions, effects, and permitted trap/refusal outcomes | No whole-language claim established by this edition. |
| Checking-route agreement | In-place versus by-value checks and wrapper/expanded-source paths, with the intended shared rule or explicit profile distinction | Unestablished generally; use chapter-specific positive/negative witnesses. |
| Lowering and optimization refinement | Source/checked program to native IR and target observations, including effects, order, uncertainty, and provenance | No general production correspondence proof established. |
| ABI/runtime correspondence | Actual value representation, lifetime, calls/returns, memory effects, and failure propagation | Target- and route-specific evidence only. |
| Handler/concurrency correspondence | Operation dispatch, continuation ownership, blocking/scheduling, memory visibility, and allowed traces | Operational choices and integration remain incomplete as recorded in chapter 03. |
| Formal model to implementation | Exact model definitions, representation map, trusted assumptions, and a relation to the named production routines | Separate obligation; a model's Lean build is not this relation. |

Add columns or adjacent references for model, production symbol, witness,
receipt, and remaining gap when an obligation is claimed for a particular
subset. This requests an inventory of what remains unknown, **not** immediate
proofs of a newly invented semantics and not a prerequisite that every
reference chapter prove full language soundness.

**Acceptance condition:** readers can identify whether a formal result is
about an abstract metadata calculus, a checking relation, executable code,
or a refinement to production—and can locate the unestablished link.

## AR-06 — MAJOR: the catalogs mix reporting requirements with source observations and unsupported authority pointers

**Affected locations:** chapter 00 standing definitions and `00-evidence.tsv`;
chapter 09 rule prose and `09-evidence.tsv`; central appendix B's promise of
rule-level source traceability.

Two related issues make path existence insufficient as authority validation:

1. At this baseline, `AGENTS.md` does not contain the symbols/headings
   `Reporting requirements` or `ADR009`. Nevertheless, `00-evidence.tsv`
   uses the former for `SPEC-005`, and `09-evidence.tsv` uses those pointers
   for `CONF-001` and `CONF-007`. The session's fuller AGENTS instructions
   and the fixed-baseline repository file are different evidence objects.
2. `09-evidence.tsv` labels `CONF-002`, `CONF-003`, `DIAG-001`, and
   `DIAG-002` `source-observed`, while their prose contains report/diagnostic
   obligations using “shall.” `CONF-008` is labeled `target-contract`, but
   its leading rule is also an evidence-reporting obligation rather than a
   target implementation behavior. Chapter 00's requirement-word convention
   distinguishes these uses, but the individual rows do not.

`docs/decisions/adr-009-verified-foreign-reference.md` is an existing accepted
oracle-policy document, but it is **not an interchangeable substitute** for
the user's current full ADR009 working rule: its oracle-class policy and
older language list do not represent all later Q#/MLX authorizations or the
same prohibition scope. Merely pointing `CONF-007` there would replace one
provenance error with another.

**Exact suggested catalog/prose correction:**

- For `SPEC-005`, use `FOUNDER_INTENT.md`, symbol `Scientific honesty`, for
  the existing stage/evidence distinction; retain a gap for any additional
  editorial obligations not contained there.
- For `CONF-001` and `CONF-007`, identify an actual approved, versioned source
  for the complete reporting rule or current ADR009, respectively. If no such
  fixed-baseline repository source exists, use `source: -`, `symbol: -`, and
  explicitly record the session-authorized policy and its missing repository
  authority reference. Do not invent a symbol in `AGENTS.md`.
- Split the observed workflow/runner behavior from the reporting requirement,
  or expressly identify the latter as an editorial reporting convention with
  its appropriate authority and standing. For `CONF-008`, preserve the
  target-gate observation separately from the general receipt requirement.

**Exact suggested addition to chapter 00's standing convention:**

> Standing and scope are different dimensions. Library and target scope alone
> do not ratify behavior. A rule's text must distinguish an adopted semantic
> requirement, an editorial/reporting requirement, and a descriptive current-
> source observation. A repository source/symbol pair must support the kind
> of claim attributed to it; path existence alone establishes none of those
> forms of authority.

**Acceptance condition:** every referenced source/symbol exists at the
declared baseline and supports the stated obligation or observation; catalog
standing agrees with the prose. The parent catalog validator can enforce the
existence portion, while substantive authority remains a review obligation.

## Checks, exclusions, and handoff

Read-only checks included Git source/tree identity, symbol/call-site searches,
launcher and suite dispatch inspection, CI artifact producer/consumer paths,
the bounded manifest's exact assertions, current seed annotation parsing,
and comparison with S06/S07/S17 and the historical specification frame.
No numerical calculation, test execution, or proof check is reported.

This review deliberately does not turn the following into new findings:

- The book does not need to hide its source-research status or claim completed
  proofs to be a useful specification-recovery reference.
- Chapter 09 correctly identifies the two bounded negative-mode spellings as
  command aliases; the correction is the unjustified stage inference.
- Raw Madaros `run` has a postcompile execution branch in current source;
  calling the same routine as `build` does not erase that branch.
- The manifest's revision/tree values match the inspected baseline. A pending
  parent coverage/catalog integration is not itself a semantic defect.
- An old S17 census is not a fresh measurement, and partial current seed
  annotation handling is not full epistemic enforcement parity.

The parent lane owns changes to the central reference, edition manifest,
foundations/conformance chapters, catalogs, and coverage validator. The
implementation limitations localized here require their own in-scope repair
and discrimination evidence; this review neither changes compiler behavior
nor silently ratifies a new language design.
