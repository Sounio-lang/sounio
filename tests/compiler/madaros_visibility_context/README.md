# Madaros module-binding identity reducer

This fixture family reduces issue #854 and validates the narrow checker slice.
It separates a false privacy diagnostic caused by name-only lookup from genuine
cross-module private access, while keeping runtime binding as a separate gate.

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

Require the checker slice, without claiming runtime binding parity, with:

```bash
MADAROS_RAW_BIN=/path/to/madaros \
SOUNIO_MADAROS_VISIBILITY_CONTEXT_EXPECT=checker-resolved \
  bash scripts/ci/madaros_visibility_context_gate.sh
```

Require the full checker plus runtime acceptance state with:

```bash
MADAROS_RAW_BIN=/path/to/madaros \
SOUNIO_MADAROS_VISIBILITY_CONTEXT_EXPECT=resolved \
  bash scripts/ci/madaros_visibility_context_gate.sh
```

`MADAROS_RAW_BIN` is mandatory so the gate cannot silently use the checked-in
or otherwise stale compiler. The gate distinguishes the exact `1/18` E175
baseline, a checker-resolved state that deliberately does not execute runtime,
and the full resolved state. A clean checker must also reject multiple global
same-name candidates with exactly one E137 instead of choosing by collection
order. Every state retains the canonical E175, E176, and E177 true-private
negative controls. Mixed states, fatal logs, diagnostic-count drift, successful
checks that retain privacy diagnostics, and runtime exits without all exact
markers are rejected.

This diagnostic gate is intentionally **not wired into ordinary CI** while the
full `EXPECT=resolved` state remains blocked in runtime binding.

## Exact baseline receipt

The baseline was revalidated against the `madaros-current-source-f64-lowering`
artifact produced for exact `origin/main` SHA
`c9bee0ccdf2bc34663252c2d3e68ea97eb6b83fe` by CI run `29286276017`
(artifact ID `8293421642`). The compiler ELF was 98,415,171 bytes with SHA-256
`f080c73f4a9890d1f947d03f1ba9f6fe4cbffd5c1170d70ed7338e2664f215d9`.

```text
context_state=baseline
runtime_state=not-run-baseline
ambiguous_global=not-run-baseline
single_e175=1
matrix_e175=18
true_private_fn=E175
true_private_struct=E176
true_private_enum=E177
```

The checker implementation was rebuilt from this exact base on Slurm with a
65536 KiB soft stack. The resulting 98,418,936-byte ELF has SHA-256
`a46802988c34661a09eba2d1b8c3c9ec7b0da37b4406507660da701960b6949e`.

```text
context_state=resolved
runtime_state=not-run-checker-slice
ambiguous_global=E137
single_e175=0
matrix_e175=0
true_private_fn=E175
true_private_struct=E176
true_private_enum=E177
```

The same ELF with `EXPECT=resolved` exits 1 because the one-call witness checks
clean but returns `rc=12`: merged runtime IR still binds the leaf call to the
root's same-name body. That is a contained runtime/IR blocker, not a checker
failure and not a full issue-resolution claim.

## Blocker handoff

```text
Blocker-ID: BLK-20260713-MADAROS-EISA-E5-VISIBILITY-E175
Status: partial (checker-resolved; runtime-binding blocked)
Severity: B1
Class: compiler-semantics
Owner: Codex root coordinator
Lane: Madaros module-binding identity / issue #854
Worktree: /tmp/sounio-issue854-contextual-20260713
Branch: codex/issue854-contextual-20260713
Files-Owned: self-hosted/check/check.sio; self-hosted/check/defs.sio; tests/compiler/madaros_visibility_context/*; scripts/ci/madaros_visibility_context_gate.sh
Files-Read-Only: self-hosted/check/mod.sio; self-hosted/compiler/module_frontend.sio; self-hosted/ir/ir.sio; self-hosted/ir/lower.sio; native codegen; EISA witnesses and stdlib
Do-Not-Touch: IR/lowering/codegen without a separately reviewed provenance lane; canonical compiler artifacts; pub-use; EISA mathematical thresholds and receipt semantics
Repro: MADAROS_RAW_BIN=<current-source-madaros> bash scripts/ci/madaros_visibility_context_gate.sh
Observed: name-only FnSigTable lookup selects the first same-name declaration even while checking a different defining module
Expected: same-module function lookup uses `(defining_module_id, name)` before visibility; cross-module fallback succeeds only when globally unique
Acceptance-Gate: MADAROS_RAW_BIN=<current-source-madaros> SOUNIO_MADAROS_VISIBILITY_CONTEXT_EXPECT=checker-resolved bash scripts/ci/madaros_visibility_context_gate.sh
Evidence-Level: E3
Evidence: exact-main CI run 29286276017 artifact 8293421642 plus Slurm source rebuild SHA-256 a46802988c34661a09eba2d1b8c3c9ec7b0da37b4406507660da701960b6949e
Fallback-Path: none
Legacy-Kept: yes; legacy name-only helper remains for unproven callers; IR/lowering/codegen are byte-unchanged
LLM-Offload: not-required (diagnostic fixtures/gate only; no math, clinical, or external claim change)
Next-Action: carry DefId/module provenance through imported function preseed, merge, finalization, and native linkage; remove name fallback only after the forward-import and seed-stub controls pass
```

## Semantic lane declaration

`SOUNIO-MODULE-BINDING-IDENTITY` remains proposed. This lane implements only
the checker-local `(module_id, name)` lookup and its unique-only fallback; it
does not claim end-to-end import or runtime identity.

```text
Semantic-Lane-ID: issue-854-module-binding-identity-reducer
Owner: Codex issue #854 reducer lane
Concept-IDs: proposed SOUNIO-MODULE-BINDING-IDENTITY
Intent-Preserved: source-level module ownership remains part of symbol identity; privacy is checked after contextual binding resolution
Transformation: checker function lookup is local-first and global-unique-only
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: current-source Madaros removes the exact 1-call and 18-call false E175 diagnostics, fails closed on ambiguous global names, and preserves true E175/E176/E177 diagnostics
Claims-Forbidden: runtime-binding parity; EISA runtime parity; general import/re-export correctness; full DefId propagation; weakening private visibility
Assumptions: root module is collected before imported leaves; unqualified same-module declarations are in lexical scope
Write-Set: self-hosted/check/check.sio; self-hosted/check/defs.sio; tests/compiler/madaros_visibility_context/*; scripts/ci/madaros_visibility_context_gate.sh
Read-Set: self-hosted/check/mod.sio; self-hosted/compiler/module_frontend.sio; self-hosted/ir/ir.sio; self-hosted/ir/lower.sio; native codegen; EISA fixtures and modules
Positive-Witness: duplicate_private_single_main.sio and duplicate_private_18_main.sio check clean with zero E175
Negative-Witness: ambiguous_public_main.sio=E137; visibility_fn_private_main.sio=E175; visibility_struct_private_main.sio=E176; visibility_enum_private_main.sio=E177
Acceptance-Gate: MADAROS_RAW_BIN=<current-source-madaros> SOUNIO_MADAROS_VISIBILITY_CONTEXT_EXPECT=checker-resolved bash scripts/ci/madaros_visibility_context_gate.sh
Integration-Target: checker slice now; future DefId/runtime-provenance lane separately
Authoritative-Only-If: both contextual checks stay clean, ambiguity stays E137, and all three true-private diagnostics remain exact
```

## Integration receipt

```text
Semantic-Outcome: checker blocker resolved; runtime-binding blocker contained
Concept-Status-Before: unregistered proposal
Concept-Status-After: unregistered proposal with an executable checker slice
Distinctions-Added: same spelling != same binding; binding resolution != visibility authorization
Distinctions-Preserved: private same-module access != private cross-module access; compile success != runtime binding parity
Distinctions-Erased: none
Evidence-Run: exact-main baseline; Slurm current-source checker rebuild; checker-resolved gate; full runtime rejection; disposable forward-import and seed-stub controls
Fallback-Path: none
Legacy-Kept: legacy name-only checker helper for unproven callers; all IR/lowering/codegen paths
Conflicting-Lanes: none observed at implementation start; PR #814 was already merged
Ordinary-CI: not wired while full runtime acceptance is red
Next-Semantic-Interface: DefId-bearing imported function records and exact merge/linkage identity
```
