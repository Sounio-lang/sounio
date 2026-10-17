<!-- docs:meta
topic_id: repo.docs.spec.2026-10.17-preparation-and-review-receipt
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.2026-10.17-preparation-and-review-receipt
-->

# Preparation and independent review receipt

Prepared 2026-10-16. Research baseline:
`f5b2c5850be9f3d3995719ffcade44a6d851658a`, tree
`25a06aafc245efec6745e72e2fa49e12c7436ff0`.
The edition was assembled in an isolated detached worktree. It does not
change compiler semantics or ratify unresolved language alternatives.

## Research and review boundaries

Six independent source-research fronts covered syntax/modules,
types/generics/ownership, effects/concurrency, epistemic information,
numerics/units/algebra, and execution/memory/targets. Central research covered
packages/tools, ontology, quantum, formal correspondence, and conformance.
The repository coverage map is a census of interfaces, not a claim that
every repository file or library API received a line-by-line audit.

The following technical reviews record the inspected source and findings:

- [Library, package, ontology, quantum, and formal review](11-library-formal-review.md).
- [Syntax/effect/execution cross-review](12-syntax-runtime-review.md).
- [Independent epistemic/numerical-model review](13-epistemic-cross-review.md).
- [Numerical gate and precision cross-review](14-numeric-cross-review.md).
- [Specification authority and conformance review](16-specification-authority-review.md).

These reports precede some corrective edits; their original findings remain
visible. A finding about the draft is not automatically a new compiler bug,
and a source-derived suspected defect is not an executable failure receipt.

## Adjudicated technical findings

| Finding | Edition correction | Remaining implementation obligation |
|---|---|---|
| Ontology validation can use parser-only acceptance, override a failed fallback compile, or print expected annotations without executing an artifact. | `ONT-004` now describes all three routes explicitly; synthetic stdout is not runtime evidence. | Repair and discriminate these routes before using them as semantic or runtime acceptance. |
| QIR serialization reorders late gates; bounded replay ignores intermediate measurement opcodes. | `QNT-005` separates QASM order, QIR order, and replay; no arbitrary-schedule equivalence is claimed. | Enforce a supported schedule or implement its semantics, then compare independently. |
| Local package lookup is convention-based, not manifest-name resolution. | `PKG-001` identifies the resolver and conventional path. Build/check/test command meanings are separated. | Manifest-controlled lookup requires separate implementation and tests. |
| Grammar accepted impl modifiers rejected by the parser, and omitted optional-operand lookahead differences. | Syntax chapter, EBNF, and catalog corrected. | Discriminating parser receipts and any future language decision remain separate. |
| Ordinary logical lowering and a GPU host-expression route do not establish one evaluation contract. | Syntax, effects, and execution chapters preserve this boundary. | Decide only unsettled semantics; enforce already adopted contracts on each named target. |
| Numeric gates do not all enforce the same exit/artifact conditions; `egate2` checks only a leading component. | `NUM-022` and `ALG-011` now describe actual branch-specific assertions and component scope. | Stronger refusal/full-correction gates need independent implementation work. |
| Wide scalar support does not establish wide `Knowledge` or wide uncertainty propagation. | Numeric and epistemic scope exclusions are explicit. | Wide scientific-channel preservation requires format-aware witnesses. |
| Lognormal overrides, absolute small-source pruning, and nominal-nonpositive EISA sqrt are exceptions to generic descriptions. | `EPI-015/021/028/041` and catalogs document these source branches and information-loss boundaries. | Distribution/affinity assumptions, direct-core domain handling, and discriminating numerical regressions remain open. |
| Existing semantic authority was conflated with executed enforcement. | Foundations distinguish adopted contracts, editorial reporting rules, observations, and conformance. | A failing adopted requirement remains a defect; a new ruling is needed only for an unresolved alternative. |
| Runner labels were mistaken for effective stages and engine identity. | Conformance chapter traces adapters and raw entrypoints and separates selection origin, engine, artifact, and CLI adapter. | Receipts must identify what actually ran, not just the harness banner. |

These corrections do not claim that the implementation obligations have been
repaired. Historical implementations and bootstrap routes were intentionally
kept. The old beta6 reference was archived rather than destroyed.

## Orthogonal provider review

Provider responses are advisory. They neither ratify language semantics nor
replace numerical or Lean verification. Operational invocation used the
existing `bin/llm-offload` wrapper from the operational checkout, copied into
an isolated remote scratch directory; that tool is not tracked at this
edition's baseline. No provider credentials are included in these artifacts.

| Task | Provider and actual response identity | Outcome |
|---|---|---|
| Full epistemic/numeric draft review | Kimi K3 through LLM Gateway | Partial response, `finish_reason=length`; not a completed review. |
| Symbolic formulas and numerical-domain assumptions | Moonshot response `aws-bedrock/kimi-k3:global` | Completed, `stop`; covariance, denominator-domain, and moment assumptions clarified. No numerical calculation accepted as an oracle. |
| Same independent symbolic review | xAI fallback requested through `xai-fast`; response model `grok-4.3` | Completed, `stop`; missing-factor allegation rejected with explicit reasoning, valid domain cautions incorporated. |
| Canonical Grok 4.7 attempts | Gateway response `aws-bedrock/grok-4-7:global` | Length-exhausted incomplete responses; direct fallback timed out. Not a review pass. |
| Additional provider attempts | DeepSeek direct, Gemini route, Z.AI direct | Authentication, credit, and service errors respectively; not passes. |
| Publication clarity of the central reference and chapters 07/08/10 | xAI fallback response `grok-4.3` | Completed advisory review; authority and navigation framing clarified; unsupported allegations rejected below. |
| Same publication excerpts | Moonshot response `aws-bedrock/kimi-k3:global` | Initial response truncated; follow-up completed with `stop`. Ontology/QIR safety exclusions, sample naming, and visible status clarified; rejected dispositions are explained below. |
| Outline/checklist decorrelation | Darwin local `coder-next` | Advisory checklist only. Unsupported suggestions, including an invented expansion of MCP, were rejected. No source or numerical validation claim. |

Completed symbolic response hashes:

```text
ee4ff29d0fd986be9a333f6f2d99807c56878576e3810a2986ccb87fdf052768  xai-fallback-symbolic.md
c17f1552137d6810f2c355c6121907449c046925784c8b8aad1e1786c82e91a9  kimi-symbolic.md
8612534b39143eeb8120cc16e92ce4ab328ec492636e27c4d51dae7482b940f7  xai-fallback-publication.md
1f4abf8e6396bb8864314500fd9bcf046cdd578f010ece85adb0b070cbcbb9be  kimi-publication.md
```

The general quadratic-surrogate variance retains coefficient one on its
third-moment contraction. The variance cross-term factor cancels the
surrogate's half; adding another factor two would double-count it. Joint
Gaussianity is stated for the Gaussian reduction, separately from finite
fourth moments for the general formula. Zero covariance, rather than full
independence, suffices for omitting a first-order covariance term. Ratio
linearization is not an existence or accuracy guarantee for exact ratio
moments. These are symbolic review dispositions, not a mechanized theorem or
an executed numerical certificate.

The publication review's suggestion that an unvalidated document can contain
only observations was rejected: adopted contracts retain authority even
when enforcement lacks a receipt. The title does not claim ISO or other
standards-body adoption. The central thematic section numbers preserve old
anchors; they are not nonexistent higher chapter files. The distinction is
made explicit in the reference. GUM fixture names and target-format emission
are not certificates of external-standard conformance.

Kimi's ontology/QIR safety findings were accepted as source-observed hazards
`ONT-H01/H02/H03` and `QNT-H01`. The ontology driver's entry was removed
from the catalog's positive-witness cell; unsafe wrapper outcomes are
explicitly excluded from conformance claims. The quantum chapter warns
against submitting reordered QIR as tape-equivalent and clarifies that
`rng_seed` is a supplied comparison sample, not a generator seed. No emitter
or wrapper repair is claimed. The suggestion to delete the founder's
governing design contract was rejected: project design authority is retained,
while no personal authority is allowed to substitute for scientific evidence.

Provider scope was bounded: full chapter 04/05 input produced only an initial
partial response; the completed two-vendor mathematical review used explicit
symbolic formulas and assumptions. The completed publication reviews read
the central reference and chapters 07/08/10, not every linked chapter.
The independent source-research and cross-review fronts cover the other
chapters. No claim of two-provider line-by-line approval of the entire book
is made. Response hashes above identify returned text, not a scientific
validation certificate. The public-review input hash is
`421b5e5cc62c2a830ba6c02890305cc35d8850f43667c9ca4d0190c282fe0ccd`;
the symbolic-review input hash is
`6ccdd90cb5c6ea856d39e417aed5222d39b52f2fab3a230b8006368e91965ab4`.

## Executed validation

- [Bounded compiler receipt](15-bounded-conformance-receipt.md): Slurm job
  `14573`, 16 cases passed, no failed cases, process exit zero, scheduler
  exit `0:0`; baseline committed Madaros prebuilt, not a source rebuild.
- The integration owner independently read the raw remote receipt, checked
  its bundle hash and compiler verification, and read the 16-case JSON summary.
- Structural documentation validation and validator mutation controls use
  `node scripts/docs/validate_language_spec.mjs` and `--self-test`.
- Documentation metadata and front-door links use
  `node scripts/docs/check_docs_registry.mjs`.
- Root asset deduplication was checked by matching SHA-256 hashes. A focused
  static control retained the run-local ABIDE manifest override before the
  canonical and shared fallbacks. The Hessian staging script's canonical
  input and shell syntax were checked; no scientific workload was executed.

Integration validation on 2026-10-16 passed: 11 chapter/catalog pairs,
332 rule declarations, 450 unique evidence paths, 151 EBNF productions,
287 baseline repository entries, and no structural errors. All 73 validator
self-tests passed, including mutations for missing declarations, invalid
symbols, directory evidence, unsafe paths, broken reference links, and
malformed grammar. JavaScript syntax, documentation registry, changed-shell
syntax, and `git diff --check` passed. The registry's stale-snapshot notice
is informational; its `unvalidated` metadata is retained rather than claiming
whole-document semantic acceptance from these checks.

Structural checks do not prove the prose or execute every catalog witness.
There is no fresh full compiler build, complete conformance corpus,
whole-language soundness proof, Lean build, GPU execution, CI completion,
commit, push, merge, or human acceptance receipt for this edition.

## Remaining investigations and promotion gates

The source-derived implementation issues above are investigations for their
own compiler/library lanes, not blockers to publishing an accurately scoped
reference. They are not assigned fictional repair owners or claimed to have
fresh failure reproductions. Each chapter's decision table and evidence gap
states the relevant source, witness, and next acceptance work. Any later lane
claiming a repair must create its own blocker/ownership record and establish
failing-before/passing-after evidence at the appropriate stage.

Edition integration into protected `main` is a separate checkpoint. The
prepared worktree is detached at the stated baseline. Remote-head identity,
the actual reviewed diff, applicable CI, and explicit publication authority
must be checked before integration; no force push or history rewrite is part
of this preparation.
