<!-- docs:meta
topic_id: repo.docs.llm-guide.reviews.diagnostics-audit
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.llm-guide.reviews.diagnostics-audit
-->

# Diagnostics and MCP source audit

Scope: documentation and source inspection at
`f5b2c5850be9f3d3995719ffcade44a6d851658a`, 2026-10-16. This audit changes
no compiler, library, MCP implementation, test expectation, or generated
catalog in bulk. It supplies no new compiler execution or numerical result.
The separately authored [bounded conformance receipt](../../spec/2026-10/15-bounded-conformance-receipt.md)
identifies its own executed prebuilt profile; it does not validate this audit.

## Documentation repaired in this lane

- [Guide entry](../README.md): root context through EN/PT reference, cookbook,
  complete atlas, specification, and a bounded compile/repair/verify workflow.
- [Error guide](../error-catalog.md): distinguish parser, checker, backend,
  seed, wrapper, and presentation routes; preserve effect/type/uncertainty
  contracts and reject silent profile substitution.
- [E035](../explanations/E035.md): remove the stale July default-check claim
  and the blanket local-`var`/`Mut` row. State the source-specific builtin
  `Knowledge.value`/E170 boundary instead of generalizing by field name.
- [E249](../explanations/E249.md): identify the historical V0-A reservation,
  current catalog-only modular identity, operation-specific limits, and the
  prohibition on silent `f64` replacement.

The published five-column namespace table is retained as historical discovery
data because the existing identity gate consumes it. Current diagnostic
meaning is determined from source route and message, not that table alone.
Prior validation is in Git history; rewritten guide metadata is unvalidated.

## Remaining contradictions and follow-up acceptance

Every finding below has **source-inspected** evidence, not a fresh failing
runtime receipt. Implementation fixes are outside this documentation lane.
The parent integration lane receives these findings; future owners should
confirm the exact profile and add discriminating tests before implementation.

### DIAG-AUDIT-001: resource E035 contradicts the current effect reporter

Severity: high for repair advice. Source:
[errors_catalog.py](../../../tools/mcp/sounio_mcp/errors_catalog.py),
`ERROR_CATALOG["E035"]`, says “Recursive type rejected.” Current
[check.sio](../../../self-hosted/check/check.sio), `print_error_message(35)`
and `report_effect_error_from`, describe missing effects. The updated
Markdown does not change that separate MCP resource.

Acceptance: independently review the current emitter and a focused effect
refusal, then make the resource explicitly route-aware and read it back.
Preserve source and compiler identities. Do not simply change a title without
checking its suggested fixes and provenance traces.

### DIAG-AUDIT-002: E041 has more than one modular reporting mechanism

Severity: medium for attribution. The resource calls E041 a unit mismatch;
`print_error_message(41)` says ontology subsumption; direct E041 unit-mismatch
prints still exist elsewhere in `check.sio`. The parser's
[types.sio](../../../self-hosted/parser/types.sio) emits E041 for Rust `&mut`.
Even “the modular meaning” is not a unique lookup in this case.

Acceptance: inventory message-table and direct reporters separately and pin
each with a stage/message witness. A code-only resource lookup must expose
the ambiguity rather than recommend one universal fix.

### DIAG-AUDIT-003: historical explanations and remapped identities

Severity: medium for retrieval. The [remap gate](../../../scripts/ci/diagnostic_remap_2170_gate.sh)
maps modular identities to E247/E248/E249/E250 while preserving the seed's
E208/E217/E218/E219 identities. It explicitly treats E249 as catalog-only
after V0-B. The older error guide still described these as live collisions.
Current kernel return checks use E253; historical
[E072](../explanations/E072.md) is not a universal route explanation.

Acceptance: inspect actual routes before any further page change. Preserve
published identity history and expose retired emitters. Do not rewrite every
explanation or renumber diagnostics as a documentation cleanup.

### DIAG-AUDIT-004: generated Markdown is not synchronized with MCP resources

Severity: medium for currentness claims. The committed `explanations/E*.md`
set is separate from the static `ERROR_CATALOG`. Its pages have differing
history and no general checked-in regeneration command was identified in
this inspection. `get_error_doc` renders catalog strings and trace paths;
it does not verify their current line contents or use the Markdown files.
Public launcher dispatch did not expose `souc explain`.

Acceptance: name the real producer before claiming regeneration. Resource
readback verifies rendering only; fresh emitter/test review establishes
meaning. Preserve metadata and add executed evidence only for a matching
profile and exact source. The guide documents this workflow without inventing
a synchronization command.

### DIAG-AUDIT-005: adapter diagnostics can change the apparent failure

Severity: high for automated repair. [check.py](../../../tools/mcp/sounio_mcp/check.py)
and [diagnostics.py](../../../tools/mcp/sounio_mcp/diagnostics.py) can synthesize
E001 for path/resolver/timeout/unclassified failures. Text fallback uses
limited location recognition and default positions. `valid` requires no
diagnostics, including no warnings. [server.py](../../../tools/mcp/sounio_mcp/server.py)
can return `isError: false` with a failed tool payload.

Acceptance: focused tests must distinguish transport/path/timeout failure,
warning-only successful compiler output, and genuine type mismatch. Preserve
raw streams and unknown attribution rather than prescribing type edits for
every E001. Existing schema assertions validate shape, not these semantics.

### DIAG-AUDIT-006: tool status does not establish the deciding stage

Severity: high for acceptance evidence.
[compile.py](../../../tools/mcp/sounio_mcp/compile.py) defines success using
nonempty output and no timeout without requiring return code zero. Its
optimization choices beyond default are provenance metadata only. Target
request admission is not backend support.
[run.py](../../../tools/mcp/sounio_mcp/run.py) checks first and then invokes
`run`; any later nonzero result becomes `runtime_error`, even if compilation
failed before execution.

Acceptance: discriminating tests should cover artifact-plus-nonzero exit,
wrong target artifact, and second-stage compile refusal. Retain the actual
stage, exit status, and artifact identity; do not treat an adapter label as
an execution witness.

### DIAG-AUDIT-007: precheck and run environment/identity can differ

Severity: high for conformance attribution. In `run.py`, `sounio_check` runs
before the call's `env` overrides are applied to the separate `run` command.
`common.py` resolves the compiler from the server environment. The check
response has no resolved raw artifact path/hash or stdlib identity fields.
A user-supplied profile override can therefore produce different precheck
and execution profiles without that difference appearing in normalized
diagnostics.

Acceptance: obtain a separate server-environment receipt and ensure the same
compiler/profile settings at both steps. If unavailable, mark attribution
unknown or profiles separate and repeat CLI check plus run on one known
profile. A future implementation fix needs a test proving the same effective
environment and reporting actual identity; documentation must not invent
fields that do not exist.

### DIAG-AUDIT-008: MCP tests do not implement canonical suite admission

Severity: high for negative-test claims.
[test.py](../../../tools/mcp/sounio_mcp/test.py) reads only the first 20 lines,
does not implement full `requires`/`known-failure`/`typecheck-fail` semantics,
and treats `expect-stdout` as a substring. A check-only case can pass for
absence of parsed diagnostics without exit-zero/no-timeout requirements.
Negative success uses diagnostics and an optional substring, not the full
exit/stage refusal contract.

Acceptance: compare with [the canonical runner](../../../scripts/dev/run_sio_test_suite.sh)
and test warning-only, timeout, wrong-engine, ignored, and negative cases
before claiming parity. Until then use a named canonical gate for its contract.

### DIAG-AUDIT-009: loop benchmark is a template demonstration

Severity: medium for model-performance claims.
[llmloop_recipe.py](../../../tools/mcp/examples/llmloop_recipe.py),
`DeterministicFixtureAgent.revise`, discards the prompt and selects canned
replacement programs from fixture metadata. Its convergence aggregate is
not an arbitrary model's repair rate and does not establish preservation of
the original requirement. The recipe calculates aggregates in Python; it was
not executed here under ADR009.

Acceptance: describe it as a deterministic plumbing demonstration. Any actual
model evaluation needs preserved task behavior, exact prompts/source/profile,
independent assertions, and authorized numerical verification.

## Checks and boundaries

Read source for launchers/resolvers, parser compatibility handling, checker
message/direct reporters, MCP tools/resources, schema, and tests. Documentation
checks validate links, metadata, and the preserved namespace format separately
from compiler behavior. This audit itself claims no new test pass, compiler
build, target execution, external review, or publication.

The inherited task mentions `.claude/AGENT_OFFLOAD_POLICY.md` and
`bin/llm-offload`; neither path was present in this isolated baseline during
inspection. No offload review was invoked in this lane. The parent owns any
broader review/logging and integration checkpoint. The optional
`sounio_verificado` service was not exposed among this session's tools; no
genuinely new Sounio was generated here.
