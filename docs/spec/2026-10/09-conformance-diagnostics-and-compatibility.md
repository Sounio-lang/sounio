<!-- docs:meta
topic_id: repo.docs.spec.2026-10.09-conformance-diagnostics-and-compatibility
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.2026-10.09-conformance-diagnostics-and-compatibility
-->

# Conformance, diagnostics, and compatibility

Conformance relates a particular implementation to particular language rules.
It requires an identifiable compiler and observable results. This chapter
specifies how to interpret the existing repository contracts and how to report
their evidence. It also records gaps that a fuller conformance suite must close.

The `editorial-contract` rules below are edition reporting conventions
authorized by Demetrios Agourakis's session direction of 2026-10-16. They do
not adopt source-language semantics. `source-observed` rules describe the
baseline runner or workflow and impose no new obligation. Adopted language
rules retain their own authority even when enforcement is missing or a test
fails. Rule identifiers and requirement words do not certify enforcement.

## 1. Implementation profiles

`CONF-001` — `editorial-contract`. A conformance report shall identify the source revision, source
modifications, launcher, resolved compiler artifact and hash, engine, target,
host, library root, command, environment overrides, and result artifacts.
It shall distinguish selection origin, engine family, raw artifact identity,
CLI adapter, effective checker/backend entrypoints, and stage actually reached.
Compiler version text alone is insufficient because several source states can
share a version string.

The public launcher defaults to Madaros through the repository resolvers.
An explicit seed selection or artifact override changes the implementation
being tested. The default committed Linux binary, a source-built Linux
candidate, an installed Darwin ARM64 artifact, and `lean_single` are separate
profiles even when they accept the same source.

`CONF-002` — `editorial-contract`. A result shall be attributed to the engine
actually executed. A wrapper, bootstrap compiler, or runtime oracle shall be
named in the receipt when it contributes to the outcome. An `explicit` label
identifies selection origin, not an engine family. A shebang identifies an
adapter file, not the semantic engine it invokes. The suite banner and
availability variables alone are not authoritative engine identification.

`CONF-013` — `source-observed`. The main suite delegates from
`scripts/run_sio_test_suite.sh` to `scripts/dev/run_sio_test_suite.sh`.
Its `SOUNIO_ENGINE_KIND` banner and engine-specific `requires` filtering use
heuristics that differ from the final launcher resolution:

| Source configuration or mechanism | Current-source limitation |
|---|---|
| No local `artifacts/self-hosted/madaros`, no explicit override | The banner selects `lean_single`, although `bin/souc` can resolve committed or host Madaros. |
| Local Madaros exists and engine selection is `lean-single` or `leansingle` | `bin/souc` accepts both seed aliases, but the banner compares only the spelling `lean_single` and can identify Madaros instead. |
| `SOUNIO_TEST_SOUC_BIN` override | The banner says `explicit`; this does not identify its underlying engine. |
| `requires: madaros` | Admission checks `SOUNIO_MADAROS_AVAILABLE`, not the identity of the selected raw executable. |
| `requires: lean_single` | Filtering considers availability and local-artifact heuristics rather than uniformly applying launcher alias/override resolution. |

These are source-derived attribution gaps, not newly executed configurations.
They affect case admission as well as banners. Under the reporting conventions
above, a receipt records the resolved raw executable/hash, adapter, selection
overrides, and engine-specific skipped cases and reasons. A skipped
Madaros-only E246/Mut fixture is not evidence of checking that boundary on
either engine. The edition manifest records a research baseline, not a
resolved or executed runtime profile.

At this edition's source revision, `.github/workflows/ci.yml` names the main
suite job `Full Test Suite` and supplies `SOUNIO_TEST_SOUC_BIN=/tmp/souc-stage2`
from the native bootstrap artifact. Other jobs build or use Madaros. The nearby
qd128 accuracy step deliberately uses the public launcher instead of that raw
stage2 positional CLI. Consequently, the workflow's job names do not identify
one universal compiler profile. Both the August engine census and the
Knowledge-surface descriptions in `docs/spec/S17_CONFORMANCE_SUITE.md` are
historical. At the current baseline,
`self-hosted/compiler/lean_single.sio::scan_type` recognizes selected epsilon
annotation forms and a `ValidUntil` shape, and consumes some identifier tags
without retaining their meaning. This partial handling is neither full
modular parity nor the absolute absence described by old S17. Current workflow
source identifies intended routes; artifact-specific completed receipts
establish observed outcomes. The foundations retain the unresolved
relationship between the older both-engine promotion criterion and
profile-scoped evidence; this edition does not waive that criterion.

## 2. The bounded executable spine

The manifest `tests/conformance/manifest.v1.tsv` is interpreted by
`scripts/ci/serious_language_conformance_gate.sh`. Its fields are:

| Field | Contract |
|---|---|
| `id` | A case identifier restricted by the runner to letters, digits, dot, underscore, and hyphen. |
| `mode` | `check`, `run`, `check-fail`, or `compile-fail`. |
| `source` | An existing repository-relative source path without the runner's prohibited traversal spelling. |
| `expected_exit` | Zero, a specific numeric exit status, or `nonzero`. |
| `stdout_contains` | A fixed substring required in stdout, or `-` to omit that assertion. |
| `stderr_contains` | A fixed substring required in stderr, or `-` to omit that assertion. |
| `claim_id` | The public claim to which the case contributes. |

`CONF-003` — `source-observed`. The bounded runner invokes the selected
`SOUC_BIN` with `check` for **both** `check-fail` and `compile-fail`. These are
command aliases, not evidence that every adapter implements checker-only
refusal. Effective stages differ by route:

| Selected route | Current-source behavior for a `check` request |
|---|---|
| Default modular `bin/madaros` | `_run_raw_check` invokes raw `--check`, reaching modular preflight. |
| Explicit seed through `bin/souc` | The seed `check` branch compiles to a temporary output and removes it. |
| `scripts/ci/souc-native-wrapper.sh` | Legacy raw mode calls `souc_compile` into a temporary artifact; modular raw mode invokes `--check`. |
| Rebuilt ontology wrapper | `scripts/ci/build_ontology_validation_souc.sh` selects a principal parser-only driver from `scripts/ci/ontology_validation_driver.sio`; diagnostic checking and fallback compilation are separate. |

The ontology wrapper can return an accepting ontology verdict after that
driver succeeds even when fallback compilation fails. Its `run` fallback can
emit copied expected-output annotations rather than execute a compiled
program. Those branches supply neither independent semantic checking nor
execution evidence. Their specific limitations are also recorded in chapter 08
and `11-library-formal-review.md`.

For `run`, the runner invokes the selected compiler's `run` command. For
`check`, it invokes `check`. It captures stdout, stderr, a shell-rendered
command, and exit status for each case. The summary contains pass/fail and a
reason. The artifact directory contains `RESULTS.md`, `summary.v1.tsv`,
`summary.v1.json`, and logs.

`CONF-012` — `editorial-contract`. A stage claim shall identify the requested
command, selected adapter, resolved raw engine, reached entrypoint, and actual
deciding stage. A nonzero result shall establish the intended refusal only
with its stage and identifying diagnostic. A wrapper's generated marker or
copied fixture output is not an independent checker or execution witness.
Neither negative-mode spelling proves that native emission or linking was
attempted, nor that it would refuse the same source on another route.

Conversely, sharing a compiler routine does not prove that `run` is
compile-only: raw Madaros `run` selects
`self-hosted/compiler/main.sio::run_native_v2_compile_mode`, which sets
`execute_after_compile` for that original mode and calls
`native_v2_run_elf_and_exit` after emission. Public wrapper `compile|build`
aliases and raw `--native-compile` versus `--native-v2-compile` are classified
by their particular dispatch, not their spelling.

`CONF-004` — `editorial-contract`. A case without an output assertion shall be described with that
limit. In the current manifest, `core-hello-run` asserts successful exit and
does not assert the hello text. `core-struct-run` asserts the substring `7`,
which is weaker than an exact complete output assertion. These cases are
useful smoke witnesses; their successful result cannot be promoted into an
unasserted output property.

The runner checks both requested streams independently. A diagnostic substring
in stdout is a stdout requirement even if its wording resembles an error
message. A reviewer must inspect the manifest's columns, rather than assume
all diagnostic messages are emitted on stderr.

### Separately executed bounded profile

The [bounded conformance receipt](15-bounded-conformance-receipt.md) records
Slurm job `14573`, completed on 2026-10-16, with gate exit `0` and the existing
manifest result `total=16 passed=16 failed=0`. Its inputs came from baseline
`f5b2c5850be9f3d3995719ffcade44a6d851658a`; the executed compiler was the
materialized committed Linux x86-64 Madaros prebuilt, SHA-256
`125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f`,
with `source-rebuilt=false`. This chapter attributes that result to the
separate execution receipt rather than claiming a new run by its author.

The result covers those manifest cases and their exact exit/substring
assertions. It does not establish current-source compiler reproduction,
general effect soundness, independent numerical/GUM verification, all-target
behavior, a full corpus, or a Lean build. Most chapter-catalog fixtures remain
unrun outside this named subset. Metadata remains `unvalidated` because the
receipt does not certify the chapter prose or every rule.

## 3. Positive and negative obligations

`CONF-005` — `editorial-contract`. A positive case shall force the rule under test to matter.
Declaring a function with an unknown type and never calling it can leave that
type uninterrogated on some routes. Parsing a unit annotation does not test
dimension arithmetic. A scientific value's printed central value does not
test its variance. The witness must reach the relevant checker or runtime use.

`CONF-006` — `editorial-contract`. A negative case shall require the intended refusal class and an
identifying diagnostic or observable marker. Nonzero exit by itself also
includes a crash or missing dependency. A timeout, signal, or unrelated
syntax error is a distinct failure result unless that exact behavior is the
rule's declared contract.

For a refusal that prohibits output production, run the candidate in a fresh
output directory and assert that no executable artifact remains. For a
runtime guard, execute the artifact and assert its failure contract. For a
type rule, include a nearby valid control so that a generally broken compiler
cannot make every negative test appear successful.

The current bounded runner has no per-case timeout or explicit signal
classification. An enclosing runner can impose bounds, but that protection
and its outcomes must be part of the receipt. These are conformance-runner
gaps, not newly promised runtime behavior.

## 4. Scientific and target witnesses

`CONF-007` — `editorial-contract`. A scientific witness shall state the semantics of its oracle and
the channels compared. Value, variance, sensitivity, covariance, confidence,
provenance, and status are separate possible observations. An exact algebraic
comparison, an approximate floating comparison, a statistical test, and a
physical observation have different acceptance conditions.

Numerical verification follows the full current user ADR009 protocol recorded
as `SPEC-011` in [the foundations](00-foundations-and-conventions.md).
Its provenance is session direction reaffirmed on 2026-10-16, including the
earlier MLX and Q#/F# authorizations; it is not claimed to be a complete ADR
adopted at the baseline commit. Canonical multi-implementation agreement uses
C++ and Sounio, and proofs use Lean 4. A metadata script does not replace an
authorized numerical oracle. The older
`docs/decisions/adr-009-verified-foreign-reference.md` oracle-class policy is
not a substitute for this complete user direction.

`CONF-008` — `editorial-contract`. A backend witness shall identify the emitted target and inspect
the artifact before execution. ELF and Mach-O recognition, signing, execution,
exit propagation, and GPU device execution each establish different facts.
A real-GPU claim requires the named runtime/device route and its outputs;
successful PTX or MSL emission is only emission evidence.

Unsupported operation tests should show a specific refusal and absence of a
misleading output artifact. Target-specific wide arithmetic should be compared
to the selected exact or format-aware oracle. A test of one target cannot
establish the same rounding, ABI, or runtime behavior on another target.

## 5. Rule, claim, and evidence maps

The existing public maps serve distinct purposes:

| Map | Purpose |
|---|---|
| `tests/conformance/manifest.v1.tsv` | Selected executable cases and their assertions. |
| `docs/serious-language/spec-evidence-matrix.v1.tsv` | Tracked specification areas, standing, and evidence references. |
| `docs/serious-language/public-claim-registry.v1.tsv` | Permitted public claims and closure boundaries. |
| `docs/serious-language/doc-claim-surface.v1.tsv` | Public, downgraded, internal, and historical document surfaces. |
| `docs/serious-language/claim-line-annotations.v1.tsv` | Exact-line anchors for selected public claims. |
| This edition's `*-evidence.tsv` tables | Rule-level source and witness references recovered during specification research. |

`CONF-009` — `editorial-contract`. A document's rule index shall not promote a public claim merely
because it names a source file or a gate. A gate reference identifies the
procedure. A completed receipt identifies its actual outcome and scope.

`CONF-014` — `source-observed`. The existing
`scripts/ci/serious_language_spec_drift_gate.sh` checks the registered matrix, invokes or consumes
bounded conformance results, checks case/claim outcomes, and verifies evidence
paths. A test path can pass the path-existence branch without being executed
by that branch. A shell gate can pass a shell-syntax check without that gate
being executed. The result must be described accordingly.

The edition validator checks document links, evidence references, stable rule
identifiers, grammar references, and the repository coverage map. Its pass is
a structural documentation result. Compiler semantics require the appropriate
compiler gates and results.

## 6. Bootstrap and proof evidence

`CONF-010` — `editorial-contract`. Bootstrap fixed-point evidence shall name the source and binary
generations compared. A fixed point over `lean_single.sio` is not a fixed
point over Madaros `compiler/main.sio`. Reproducing a compiler binary and
satisfying the source-language specification are independent obligations.

`CONF-011` — `editorial-contract`. A formal verification claim shall name the theorem, formal model,
toolchain, dependencies, assumptions, and completed checker result. A Lean
theorem about a small calculus does not establish that every production
checker or emitter implements that calculus. A mirror program requires a
separate refinement or conformance argument.

For each claimed subset, the report shall identify the model, production
symbols, representation/correspondence relation, trusted assumptions,
witnesses, receipt, and remaining gap for the applicable obligations below.
The inventory records unknown links rather than promising that proofs exist.

| Obligation family | Required scope and correspondence | Current edition boundary |
|---|---|---|
| Static preservation / safe progress | Named core subset, typed state, ownership/resource assumptions, effects, and defined trap/refusal outcomes | No whole-language theorem established by this edition. |
| Checking-route agreement | In-place versus by-value checking, adapters, and expanded-source routes under a shared rule or explicit profile distinction | Unestablished generally; individual chapter witnesses and profiles remain necessary. |
| Lowering and optimization refinement | Source/checked program to native IR and target observations, including effects, order, uncertainty, and provenance | No general production refinement proof established. |
| ABI/runtime correspondence | Value representation, lifetime, calls/returns, memory effects, and failure propagation | Target- and route-specific evidence only. |
| Handler/concurrency correspondence | Operation dispatch, continuation ownership, blocking/scheduling, memory visibility, and allowed traces | Decisions and integration remain incomplete as described in chapter 03. |
| Formal model to implementation | Exact model definitions, representation map, assumptions, and relation to named production routines | A model's Lean build alone is not that relation. |

For example, the metadata-masking `handle` in
`formal/lean4/SounioEffects.lean` preserves an already stored payload; it does
not supply a refinement of the production handler or concurrency paths.
The terminating judgment in the foundations is not a progress or totality
claim. Models covering divergence, stuck states, nondeterministic traces,
resource exhaustion, and external boundaries state those choices explicitly.

As a separate source observation,
`scripts/ci/lean_axiom_report.sh::run_report` builds named packages and runs
selected axiom reports; its failure check includes `sorryAx` and does not
reject every nonstandard axiom. That procedure has not been run by this
chapter or by the bounded manifest receipt. No blanket proof or axiom-free
status follows from cataloging it. This reporting inventory neither chooses
new language semantics nor requires an immediate whole-compiler proof.

## 7. Diagnostics

`DIAG-001` — `editorial-contract`. A diagnostic contract consists of the violated rule, the stage,
the required refusal behavior, and any stable identifying code or wording.
Exact wording is required only when the applicable test or contract pins it.
The existence of an `E...` code does not prove every path producing it reports
the same cause, span, or severity.

The driver in `self-hosted/compiler/main.sio`, checker in
`self-hosted/check/check.sio`, parser recovery, and target refusals contribute
different diagnostic classes. The reference describes the pertinent code and
witness in each subject chapter. It does not invent a globally stable
catalog from the historical maximum error number.

`DIAG-002` — `editorial-contract`. Recovery and continued analysis shall be distinguished from
successful translation. The legacy seed's imported-error policy and the
current driver's error propagation must be inspected on their own routes.
A successful shell wrapper cannot establish that all imported units were
soundly checked.

## 8. Compatibility and change control

`COMPAT-001` — `editorial-contract`. This edition's rule identifiers provide references within the
edition. A semantic revision records the affected rules, the previous and new
behavior, source and library compatibility effects, migration examples, and
positive/negative witnesses. An editorial change preserving meaning need not
claim a compiler semantic change.

A proposed future language rule must be labeled as a candidate until its
semantic decision and acceptance obligations are settled. Intentional
prototype extensions may coexist with a supported core, provided their
status and engine/target scope are visible. The founder's scientific semantic
intentions remain governing inputs to those decisions.

## 9. Reproduction entry points

```sh
node scripts/docs/validate_language_spec.mjs
node scripts/docs/check_docs_registry.mjs
bash scripts/ci/serious_language_conformance_gate.sh
bash scripts/ci/serious_language_spec_drift_gate.sh
bash scripts/ci/serious_language_claim_closure_gate.sh
```

The first two commands check documentation. The latter commands can execute
compiler and evidence checks. Full or stress compiler validation belongs on
the repository's Foundry/Slurm route. Each invocation must retain its own
environment and result artifacts; listing these commands is not an execution
receipt.
