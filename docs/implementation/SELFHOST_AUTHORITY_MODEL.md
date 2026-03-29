<!-- docs:meta
topic_id: repo.docs.implementation.selfhost-authority-model
authority: repo_only
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.implementation.selfhost-authority-model
-->

# Selfhost Authority Model

## Purpose

This file is the maintainer-facing overview for the self-hosted compiler authority model.

Use it to answer:

- what is the authoritative selfhost gate?
- how is source↔artifact parity judged?
- which CI checks are intended to be required?
- what is supported on x86 versus AArch64?
- which compiler paths are intentionally fenced?

## Canonical entrypoints

- local authority gate: [`scripts/selfhost/selfhost_authority_gate.sh`](../../scripts/selfhost/selfhost_authority_gate.sh)
- focused ABI/parity gate: [`scripts/selfhost/selfhost_ci_abi_parity_gate.sh`](../../scripts/selfhost/selfhost_ci_abi_parity_gate.sh)
- provenance verification gate: [`scripts/selfhost/selfhost_artifact_provenance_gate.sh`](../../scripts/selfhost/selfhost_artifact_provenance_gate.sh)
- source↔artifact parity gate: [`scripts/selfhost/selfhost_source_artifact_parity_gate.sh`](../../scripts/selfhost/selfhost_source_artifact_parity_gate.sh)
- fixed-point gate: [`scripts/selfhost/selfhost_x86_fixed_point_gate.sh`](../../scripts/selfhost/selfhost_x86_fixed_point_gate.sh)
- artifact promotion entrypoint: [`scripts/selfhost/update_selfhost_artifact.sh`](../../scripts/selfhost/update_selfhost_artifact.sh)
- checked artifact provenance: [`artifacts/self-hosted/souc-self-hosted-x86_64.provenance.json`](../../artifacts/self-hosted/souc-self-hosted-x86_64.provenance.json)
- promotion policy: [`scripts/selfhost/selfhost_promotion_policy.v1.json`](../../scripts/selfhost/selfhost_promotion_policy.v1.json)
- required-checks manifest: [`scripts/selfhost/selfhost_required_checks.v1.json`](../../scripts/selfhost/selfhost_required_checks.v1.json)
- release drift gate: [`scripts/selfhost/selfhost_release_drift_gate.sh`](../../scripts/selfhost/selfhost_release_drift_gate.sh)
- release candidate gate: [`scripts/selfhost/selfhost_release_candidate_gate.sh`](../../scripts/selfhost/selfhost_release_candidate_gate.sh)

## Pass/fail semantics

Blocking selfhost authority checks:

- `promotion_policy`
- `release_drift`
- `fixed_point`
- `fallback_inventory`
- `abi_parity_regressions`
- `aarch64_compile_proof`
- `source_artifact_parity`

Non-blocking legacy surface:

- `legacy_native_acceptance`

Parity is judged by comparing:

1. the accepted checked-in artifact baseline
2. the rebuild-from-source baseline
3. the current head compiler built from source

Current head is acceptable only if it introduces no unresolved failures relative to the accepted artifact baseline.

## Protected-branch model

Target protected branch:

- `main`

Checks intended to be required:

- `Contracts`
- `Selfhost Authority`
- `Selfhost ABI/Parity`
- `Sounio Lint`
- `Website`

Checks intended to stay informational unless the baseline changes:

- `Seed Policy`
- `Rust-Free Proof`
- `LSP Smoke`
- `Full Self-Host (5 generations)`
- `Release Dashboard`

Review expectation:

- require at least one approving review for all PRs
- require maintainer review for PRs touching:
  - `self-hosted/compiler/lean_single.sio`
  - `artifacts/self-hosted/souc-self-hosted-x86_64`
  - `artifacts/self-hosted/souc-self-hosted-x86_64.provenance.json`
  - `scripts/selfhost/`
  - `tests/selfhost/`

## Target support taxonomy

- `x86-runtime-supported`: supported runtime behavior and merge-blocking in ABI/parity coverage
- `aarch64-runtime-supported`: reserved for true AArch64 runtime support; no current cases are admitted
- `aarch64-compile-proof`: code generation accepted as compile-proof only
- `aarch64-explicit-unsupported`: expected-fail paths that must reject with a named diagnostic

The manifests under `tests/selfhost/` are the active taxonomy surface. Unsupported paths must be recorded there as expected-fail coverage instead of succeeding silently.

## Fallback debt status

The legacy aggregate BSS return path is not part of the supported normal path.

Current status:

- `stabilize_return_agg_x86`: unsupported but fenced
- `stabilize_return_agg_a64`: unsupported but fenced

The canonical inventory is produced by:

- [`scripts/selfhost/selfhost_fallback_inventory_gate.sh`](../../scripts/selfhost/selfhost_fallback_inventory_gate.sh)

## Related operator docs

- release train and promotion sequence: [`docs/implementation/SELFHOST_RELEASE_TRAIN.md`](SELFHOST_RELEASE_TRAIN.md)
- deferred debt register: [`docs/implementation/SELFHOST_DEBT_REGISTER.md`](SELFHOST_DEBT_REGISTER.md)
- contributor-facing compiler map: [`docs/implementation/SELF_HOSTED_COMPILER.md`](SELF_HOSTED_COMPILER.md)
