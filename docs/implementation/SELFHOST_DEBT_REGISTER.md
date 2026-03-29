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

- `stabilize_return_agg_x86` legacy BSS fallback
  - class: fenced unsupported
  - severity: medium
  - note: normal supported path is caller-owned SRET; fallback remains only as fenced legacy code

- `stabilize_return_agg_a64` legacy BSS fallback
  - class: fenced unsupported
  - severity: medium
  - note: normal supported path is caller-owned SRET; fallback remains only as fenced legacy code

- AArch64 closure literals
  - class: transitional implementation debt
  - severity: high
  - note: compile path is explicitly unsupported and expected-fail, not runtime-supported

- `legacy_native_acceptance` non-green baseline
  - class: baseline inherited noise
  - severity: medium
  - note: kept non-blocking because the accepted baseline is not green there yet

- ruleset/branch-protection application
  - class: CI/governance follow-up
  - severity: medium
  - note: repo-local model is explicit; hosted branch protection still needs maintainer application

## Wave 5 candidate

Highest-value implementation wave after governance hardening:

- eliminate fenced aggregate-return fallback debt in normal paths
- implement AArch64 closure literals for real runtime support
- reclassify supported AArch64 paths from compile-proof to runtime-supported only after tests prove it
