<!-- docs:meta
topic_id: repo.docs.implementation.selfhost-debt-register
authority: repo_only
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.implementation.selfhost-debt-register
-->

# Selfhost Debt Register

## Purpose

This file is the short deferred-debt register for the self-hosted compiler authority program.

## Current deferred debt

- `stabilize_return_agg_x86` fenced missing-SRET path
  - class: fenced unsupported
  - severity: medium
  - note: normal supported path is caller-owned SRET; shared aggregate BSS codegen is no longer emitted here

- `stabilize_return_agg_a64` fenced missing-SRET path
  - class: fenced unsupported
  - severity: medium
  - note: normal supported path is caller-owned SRET; shared aggregate BSS codegen is no longer emitted here

- AArch64 runtime surface expansion
  - class: transitional implementation debt
  - severity: medium
  - note: closure literals now cover runtime-validated capture, aggregate return, closure-expression aggregate return, closure struct return, large import, and `>7` user-parameter closure calls; broader runtime promotion still needs additional cases such as `ref_inner_struct_field`

- `legacy_native_acceptance` non-green baseline
  - class: baseline inherited noise
  - severity: medium
  - note: kept non-blocking because the accepted baseline is not green there yet

- ruleset/branch-protection application
  - class: CI/governance follow-up
  - severity: medium
  - note: repo-local model is explicit; hosted branch protection still needs maintainer application

- hosted artifact attestation
  - class: CI/governance follow-up
  - severity: low
  - note: repo-local provenance plus repo-local reproducible bootstrap is the active trust model; hosted attestation remains optional and should only be added when the repo/plan context supports it cleanly

## Next candidate wave

Highest-value implementation wave after governance hardening:

- eliminate the remaining fenced missing-SRET sites once all callers are proven to supply hidden result buffers
- expand AArch64 runtime coverage beyond the current promoted closure, aggregate-return, and large-import surface
- reclassify additional AArch64 paths from compile-proof to runtime-supported only after runtime tests prove them
