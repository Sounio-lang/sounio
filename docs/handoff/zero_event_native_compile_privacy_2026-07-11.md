<!-- docs:meta
topic_id: repo.docs.handoff.zero-event-native-compile-privacy-2026-07-11
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.zero-event-native-compile-privacy-2026-07-11
-->

# Zero-event native compile privacy handoff

## Blocker record

```text
Blocker-ID: BLK-20260711-ZERO-PRIVACY
Status: closed
Severity: B1
Class: compiler-semantics
Owner: fix/zero-event-native-privacy
Lane: zero-event constructor opacity under default native compile
Worktree: /tmp/sounio-zero-privacy
Branch: integrated on neurodyn-docs-registry-batch through ecc5c4018
Files-Owned: self-hosted/compiler/main.sio, scripts/ci/zero_event_native_compile_privacy_gate.sh
Files-Read-Only: stdlib/epistemic/zero_event.sio, scripts/ci/zero_event_gate.sh, scripts/ci/zero_event_native_v2_matrix.sh
Do-Not-Touch: bin/souc, scripts/run_sio_test_suite.sh, zero-event semantic oracle
Repro: ./bin/souc compile tests/compile-fail/zero_event_direct_receipt_construction.sio -o /tmp/zero-event-forged
Observed: the pre-fix native compiler accepted the literal; the rebuilt default compiler rejects it with E176
Expected: check and compile both reject external construction of ZeroReceiptF64 and ErasedZeroF64 with E176
Acceptance-Gate: ZERO_EVENT_SOURCE_ROOT=/workspace/sounio bash scripts/ci/zero_event_native_compile_privacy_gate.sh
Evidence-Level: E3
Evidence: rebuilt /tmp/sounio-zero-privacy/artifacts/madaros-zero-privacy-v3; canonical compile gate PASS on 2026-07-11
Fallback-Path: lean_single remains an explicit semantic oracle; it is not native-v2 evidence
Legacy-Kept: yes; do not remove the visibility-disabled merged checker until import-context parity is proven
LLM-Offload: not-required (compiler visibility semantics, no new math or clinical claim)
Next-Action: none for constructor opacity; track direct private-field reads separately
```

## Review-ready implementation

Commits `f998396bc` and `ecc5c4018` extract the already-authoritative visibility path from
`run_check_mode` into `compiler_visibility_preflight` and calls it before
`module_frontend_compile_imported_to_file`. The internal merged checker remains
unchanged and permissive after that boundary, so the patch does not globally
enable visibility or edit `self-hosted/check/mod.sio`.

The first experiment used `module_frontend_populate_imported_programs` for a
second preflight and was rejected: it consumed a full CPU for more than 90
seconds on the first zero-event probe. The committed implementation reuses the
loader logic from `run_check_mode`; the same negative probe returns E176 in
approximately 0.07 seconds on the rebuilt compiler.

The isolated acceptance gate uses the canonical `compile` command and proved:

- forged `ZeroReceiptF64` and `ErasedZeroF64`: rejected with E176;
- private imported struct/function/enum: rejected with E176/E175/E177;
- public imported struct: native emission succeeds;
- `tests/multimodule/wp_a3/w2_main.sio`: generic native emission succeeds;
- positive zero-event stdlib witness: native emission succeeds;
- EISA zero-flags witness: visibility verdict is zero, then it reaches the
  separately classified native backend `rc=12` frontier.

## Root cause boundary

`run_check_mode` in `self-hosted/compiler/main.sio` calls
`check_modules_verdict_boot4_with_visibility(..., true)`. The imported native
compile paths in `self-hosted/compiler/module_frontend.sio` call
`check_modules_verdict_boot4(...)`, whose wrapper disables visibility.

This is deliberate historical behavior, not a missing boolean by accident.
The merged checker knows each definition's module, but it does not retain enough
per-use import context to distinguish:

1. a public API use from another module;
2. a compiler-internal dependency that was legitimately admitted by the module
   loader; and
3. an external construction or access to a private type.

Changing every native call to
`check_modules_verdict_boot4_with_visibility(..., true)` is therefore only a
diagnostic experiment. Existing comments in `self-hosted/check/mod.sio` record
false E175/E176/E177 failures for legitimate internal dependencies when that
global switch is enabled.

## Required semantic model

Visibility must be checked at the use edge, with both pieces of evidence:

- the defining module and declaration visibility; and
- the importing module's resolved `use` context/export decision.

The native frontend must reject a private struct literal written by a consumer
module even if the loader has discovered the defining module transitively.
Loader reachability is not construction authority. Conversely, a public item
must remain usable, and compiler-internal dependency edges already admitted by
the current frontend must not acquire spurious visibility errors.

The generic-specialization branch needs the same policy. Its merged item list
currently calls `check_items_verdict_boot4`, so fixing only the ordinary
`check_modules_verdict_boot4` fallback leaves a bypass whenever
`spec_last_instantiated()` is true.

## Acceptance matrix

The implementing lane should create
`scripts/ci/zero_event_native_compile_privacy_gate.sh` and prove all rows using
a freshly rebuilt Madaros artifact, never the stale workspace binary.

| Case | Command surface | Required result |
|---|---|---|
| Forged `ZeroReceiptF64` | native `compile` | reject with E176 |
| Forged `ErasedZeroF64` | native `compile` | reject with E176 |
| Private field read | native `compile` | residual language-wide limitation; tracked separately from constructor opacity |
| Public imported struct | `tests/multimodule/visibility_struct_pub_main.sio` | compile succeeds |
| Private imported struct | `tests/multimodule/visibility_struct_private_main.sio` | reject with E176 |
| Private imported function | `tests/multimodule/visibility_fn_private_main.sio` | reject with E175 |
| Private imported enum | `tests/multimodule/visibility_enum_private_main.sio` | reject with E177 |
| Generic imported public API | existing multimodule generic run-pass set | check and compile remain green |
| EISA imported program | `tests/stdlib/eisa/test_eisa_evm_v2.sio` | check and explicit native gate retain their prior classification |

After the rebuilt compiler passes the focused matrix, run the Madaros full gate
through Compiler Foundry/Slurm. Do not run full stress in `/workspace/sounio`.

## Closure rule

Move the two constructor probes from `tests/known_failures/` to
`tests/compile-fail/` only after the canonical compile-based harness rejects
them using the rebuilt default compiler. Then update `zero_event_gate.sh` to use
the compile-fail harness rather than check-only opacity evidence.

The constructor-opacity blocker closes at E3 when the default native compile
path enforces the same constructor boundary as `check` without breaking the
positive import and generic rows. That condition is now met. Direct private
field reads remain a separate compiler-semantics limitation; they do not permit
construction of either receipt type and are outside this blocker.
