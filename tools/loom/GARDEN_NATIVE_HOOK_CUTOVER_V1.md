# Garden Seed: Native Hook Cutover V1

Status: Garden
Date: 2026-08-31
Concept-ID: SOUNIO-LOOM-NATIVE-HOOK-CUTOVER
Parent: action 9044, frozen at `tools/loom/sovereign_material_change.freeze.v2`

## First Phrase

A provider may request an operation, but it cannot choose the language,
dialect, policy, or receipt that makes the request admissible.

## Hypothesis

One Sounio semantic authority can govern the distinct native hook envelopes of
Codex, Claude, Cursor, and Grok without using Python, Rust, or a disposable
language as an oracle. OCaml can normalize and operate the envelopes while the
frozen Sounio executable remains the sole producer of expected decisions.

The cutover becomes real only when the shared coordination runtime no longer
ships the Python bridge, all four provider configurations promote atomically,
failures refuse before execution, and real provider CLIs produce decision
receipts through the same native ingress.

## Ordered Experiment

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> FOUR_PROVIDER_CANARY
-> CLAIM_READY
```

## Semantic Lane Declaration

```text
Semantic-Lane-ID: loom-native-hook-cutover-20260831
Owner: codex-1
Concept-IDs: SOUNIO-LOOM-NATIVE-HOOK-CUTOVER
Intent-Preserved: Sounio produces the first executable representation and every expected hook decision
Transformation: provider-specific native envelopes become hash-bound inputs to frozen Sounio action 9045
Types-Changed: none
Effects-Changed: provider hook ingress becomes deny-by-default and receipt-producing before execution
IR-Changed: none
Claims-Introduced: four native provider dialects can share one Sounio-authoritative OCaml-operated ingress after live canaries pass
Claims-Forbidden: configuration text, OCaml, shell, Python, Rust, provider output, or LLM review can manufacture semantic authority
Assumptions: action 9044 remains frozen; provider CLIs expose documented hook events; runtime hashes are available before promotion
Write-Set: stdlib/coordination/loom_native_hook_cutover_authority.sio, tools/loom/native_hook_cutover_authority_main.sio, tools/loom/src/loom_hook.ml, tools/loom/src/loom.ml, native hook installers and selftests, Cursor and Grok hook configurations
Read-Set: frozen action 9044, existing native Codex and Claude hooks, provider hook documentation, shared runtime installer
Positive-Witness: each of Codex, Claude, Cursor, and Grok reaches the native OCaml ingress and receives a Sounio action-9045 ALLOW receipt
Negative-Witness: removing the Sounio Python-absence rule promotes the unchanged prohibited-bridge witness; the unmodified rule refuses it
Acceptance-Gate: scripts/ci/sounio_loom_native_hook_cutover_selftest.sh and four real provider canaries pass from a fresh shared runtime install
Integration-Target: shared coordination runtime and project provider hook configurations
Authoritative-Only-If: action 9045 is frozen by hash, the OCaml runtime consumes its exact decision, the Python bridge is absent from the package, and all four live canaries pass
```

## Falsifiers

- a provider/dialect mismatch is normalized and admitted;
- Python, Rust, Node, Ruby, shell, awk, or another disposable oracle produces
  an expected hook decision;
- an unavailable or timed-out Sounio authority fails open;
- a pre-tool operation begins before its Sounio decision;
- ALLOW or DENY occurs without a complete receipt;
- any promoted provider configuration names the retired Python bridge;
- a partial four-file promotion survives a failed install;
- a synthetic fixture is reported as a live provider canary; or
- the same witness still refuses after the load-bearing Sounio rule is removed.

This seed authorizes the action-9045 experiment. It is not evidence that the
shared runtime, provider configurations, or live canaries already satisfy it.
