# Madaros module-binding identity reducer

This fixture family reduces issue #854 without changing checker semantics. It
separates a false privacy diagnostic caused by name-only lookup from genuine
cross-module private access.

The one-call reducer mirrors the E5 test. The root and imported leaf each own a
private `abs_f64`; the leaf's one internal call currently resolves to the
root's earlier table entry and emits one false E175.

The matrix mirrors the public example exactly at the causal level:

```text
eisa::format::abs_f64       1 internal call
eisa::evm::print_i64_dec   17 internal calls
                            -----------------
                            18 false E175 diagnostics
```

The fixture uses the same two duplicate private names and the same `1 + 17`
call topology. The baseline is classified with `madaros check`, before
lowering or runtime can replace the diagnostic exit status. Only after both
reducers check clean does the gate execute them. Different return values then
make a superficial "allow access" change insufficient: the resolved state
must print the exact PASS markers from the module-owned bindings.

Run the classifier against a current-source Madaros:

```bash
MADAROS_RAW_BIN=/path/to/madaros \
  bash scripts/ci/madaros_visibility_context_gate.sh
```

Require the implementing acceptance state with:

```bash
MADAROS_RAW_BIN=/path/to/madaros \
SOUNIO_MADAROS_VISIBILITY_CONTEXT_EXPECT=resolved \
  bash scripts/ci/madaros_visibility_context_gate.sh
```

`MADAROS_RAW_BIN` is mandatory so the gate cannot silently use the checked-in
or otherwise stale compiler. The gate accepts only two coherent
classifications: the exact `1/18` E175 check baseline, or two clean checks
followed by both executable contextual-lookup witnesses. In either state it
requires the canonical E175, E176, and E177 true-private negative controls.
Mixed states, fatal logs, diagnostic-count drift, successful checks that
retain privacy diagnostics, and runtime exits without exact markers are
rejected.

This diagnostic gate is intentionally **not wired into ordinary CI**. The
implementation PR must wire the `EXPECT=resolved` form after it owns the
checker change; the present lane records and classifies the blocker without
making a known failure block unrelated pull requests.

## Exact baseline receipt

The initial classifier run used the `madaros-current-source-f64-lowering`
artifact produced for exact `origin/main` SHA
`e6725240d266353621639419f1c4fd859d90caad` by CI run `29270956296`
(artifact ID `8287540274`). The compiler ELF was 98,413,103 bytes with SHA-256
`7e6203c08b254ea46cbae4094a0109317ac28063e4bb3b84209d51eeef2ae8fa`.

```text
context_state=baseline
runtime_state=not-run-baseline
single_e175=1
matrix_e175=18
true_private_fn=E175
true_private_struct=E176
true_private_enum=E177
```

The same artifact with `SOUNIO_MADAROS_VISIBILITY_CONTEXT_EXPECT=resolved`
exits 1 and emits the blocker ID without attempting reducer runtime. A
synthetic resolved-state harness proves the gate orders clean checks before
the two exact runtime markers; it is a gate self-test, not evidence that the
compiler is fixed. A separate synthetic compiler that accepts every input is
rejected when the E175 negative control returns 0. This pins the state machine
and anti-weakening behavior independently of the current compiler result.

## Blocker handoff

```text
Blocker-ID: BLK-20260713-MADAROS-EISA-E5-VISIBILITY-E175
Status: classified
Severity: B1
Class: compiler-semantics
Owner: Codex root coordinator (implementation dispatch pending)
Lane: Madaros module-binding identity / issue #854
Worktree: /tmp/sounio-issue-854-reducer-20260713
Branch: codex/issue-854-reducer-20260713
Files-Owned: tests/compiler/madaros_visibility_context/*; scripts/ci/madaros_visibility_context_gate.sh
Files-Read-Only: self-hosted/check/check.sio; self-hosted/check/defs.sio; self-hosted/check/mod.sio; self-hosted/ir/lower.sio; EISA witnesses and stdlib
Do-Not-Touch: self-hosted/check/check.sio while PR #814 owns it; canonical compiler artifacts; EISA mathematical thresholds and receipt semantics
Repro: MADAROS_RAW_BIN=<current-source-madaros> bash scripts/ci/madaros_visibility_context_gate.sh
Observed: name-only FnSigTable lookup selects the first same-name declaration even while checking a different defining module
Expected: an unqualified call resolves through the caller module/import context before visibility is evaluated
Acceptance-Gate: MADAROS_RAW_BIN=<current-source-madaros> SOUNIO_MADAROS_VISIBILITY_CONTEXT_EXPECT=resolved bash scripts/ci/madaros_visibility_context_gate.sh
Evidence-Level: E3
Evidence: exact-main CI run 29270956296 artifact 8287540274 plus the baseline receipt in this README
Fallback-Path: none
Legacy-Kept: yes; existing checker, EISA, and negative visibility paths are unchanged
LLM-Offload: not-required (diagnostic fixtures/gate only; no math, clinical, or external claim change)
Next-Action: implement module-aware function binding lookup after PR #814 releases checker ownership
```

## Semantic lane declaration

`SOUNIO-MODULE-BINDING-IDENTITY` is proposed here for the future implementation
lane. It is not yet registered as an executable concept, and this reducer does
not alter its semantics.

```text
Semantic-Lane-ID: issue-854-module-binding-identity-reducer
Owner: Codex issue #854 reducer lane
Concept-IDs: proposed SOUNIO-MODULE-BINDING-IDENTITY
Intent-Preserved: source-level module ownership remains part of symbol identity; privacy is checked after contextual binding resolution
Transformation: none; executable characterization only
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: current-source Madaros reproduces exact 1-call and 18-call E175 name-collision boundaries while true private access remains rejected
Claims-Forbidden: EISA runtime parity; general import/re-export correctness; implementation of module-aware bindings; weakening private visibility
Assumptions: root module is collected before imported leaves; unqualified same-module declarations are in lexical scope
Write-Set: tests/compiler/madaros_visibility_context/*; scripts/ci/madaros_visibility_context_gate.sh
Read-Set: self-hosted/check/check.sio; self-hosted/check/defs.sio; self-hosted/check/mod.sio; self-hosted/compiler/main.sio; EISA fixtures and modules
Positive-Witness: acceptance-only; after clean checks, duplicate_private_single_main.sio and duplicate_private_18_main.sio must execute their exact PASS markers
Negative-Witness: visibility_fn_private_main.sio=E175; visibility_struct_private_main.sio=E176; visibility_enum_private_main.sio=E177
Acceptance-Gate: MADAROS_RAW_BIN=<current-source-madaros> SOUNIO_MADAROS_VISIBILITY_CONTEXT_EXPECT=resolved bash scripts/ci/madaros_visibility_context_gate.sh
Integration-Target: future checker/module-binding implementation lane after PR #814
Authoritative-Only-If: both contextual witnesses execute and all three true-private diagnostics remain exact
```

## Integration receipt

```text
Semantic-Outcome: exact blocker reduced; implementation intentionally deferred
Concept-Status-Before: unregistered proposal
Concept-Status-After: unregistered proposal with executable characterization
Distinctions-Added: same spelling != same binding; binding resolution != visibility authorization
Distinctions-Preserved: private same-module access != private cross-module access; compile success != runtime binding parity
Distinctions-Erased: none
Evidence-Run: baseline classifier against exact-main CI artifact; resolved-state and accept-all harness self-tests
Fallback-Path: none
Legacy-Kept: all existing checker, EISA, and visibility paths
Conflicting-Lanes: PR #814 owns self-hosted/check/check.sio; no overlap in this phase
Ordinary-CI: not wired in this diagnostic lane; required wiring belongs to the implementation PR
Next-Semantic-Interface: module-qualified binding records and caller-context lookup
```
