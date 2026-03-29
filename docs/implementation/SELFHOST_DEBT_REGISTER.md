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

- AArch64 closure literals
  - class: transitional implementation debt
  - severity: medium
  - note: runtime validation now exists for the current focused closure surface; `>7` user-parameter closure literals remain explicitly fenced unsupported and broader runtime coverage still needs expansion

- `legacy_native_acceptance` non-green baseline
  - class: baseline inherited noise
  - severity: medium
  - note: kept non-blocking because the accepted baseline is not green there yet

- ruleset/branch-protection application
  - class: CI/governance follow-up
  - severity: medium
  - note: repo-local model is explicit; hosted branch protection still needs maintainer application

## Next candidate wave

Highest-value implementation wave after governance hardening:

- eliminate the remaining fenced missing-SRET sites once all callers are proven to supply hidden result buffers
- expand AArch64 runtime coverage beyond the current focused closure and aggregate-return surface
- reclassify additional AArch64 paths from compile-proof to runtime-supported only after runtime tests prove them
