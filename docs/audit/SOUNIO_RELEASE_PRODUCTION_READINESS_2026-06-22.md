<!-- docs:meta
topic_id: repo.docs.audit.sounio-release-production-readiness-2026-06-22
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.sounio-release-production-readiness-2026-06-22
-->

# Sounio Release Production Readiness - 2026-06-22

## Scope

This note defines the broader release gate for Sounio. It
is intentionally stricter than the Madaros executable gate. The public status is: research language (pre-1.0). APIs may change; not recommended for production use.

Madaros readiness answers:

- Is the default compiler path currently green on `main`?
- Are the tracked Madaros production blockers closed?
- Is the compiler-owner PR queue clear enough for the current compiler lane?

Sounio release production readiness answers a larger product question:

- Can the repository safely present Sounio as a broad language distribution
  for production use, including compiler source lineage, stdlib surface, tooling,
  installation, docs, package flow, and public support wording?

As of `origin/main@f9e139a6c4ea7ff5a6e63387c807bbe0578a189c`, the gate outcomes were:

- **Madaros executable gate:** pass, on the gate named below.
- **Broad release gate:** not passed, by the stricter release gate introduced here.

These outcomes are a dated audit. They are not the public maturity status. The public line is: research language (pre-1.0). APIs may change; not recommended for production use.

This is not a regression. It is a higher bar.

## Executable Gate

Use:

```bash
scripts/ci/sounio_release_production_readiness_gate.sh
```

The gate requires:

1. Clean checkout at `origin/main`.
2. Madaros production readiness:
   `scripts/dev/madaros_readiness_status.sh --no-audit --production-ready`.
3. Docs registry and docs consistency checks.
4. Serious-language public claim closure:
   `scripts/ci/serious_language_claim_closure_gate.sh`.
5. No release-critical public product surface remaining
   `prototype`/`downgraded` in
   `docs/serious-language/public-claim-registry.v1.tsv`.

For fast classification without rerunning live gates:

```bash
scripts/ci/sounio_release_production_readiness_gate.sh --skip-live-gates
```

The skip mode is for triage only. It is not sufficient for a release sign-off.
When changing the gate itself on a feature branch, use `--allow-non-main` to
validate the classification logic before merge. Do not use `--allow-non-main`
for release sign-off.

## Release-Critical Blockers

These claim IDs currently block the broad release gate:

The broad `direct_driver` frontier remains `prototype` / `downgraded` and is
explicitly out of scope for this release-support contract. The release contract
requires the narrower `direct_driver.support` claim instead. This does not imply
general direct-driver readiness or stability; it only locks in the named
24-fixture research cohort while broad direct-driver support stays unsupported
for release purposes.

| Claim ID | Current status | Why it blocks broad release |
|---|---|---|
| `binary.source` | `prototype` / `downgraded` | The checked binary still depends on `lean_single` as the source path; modular source-swap parity is not closed. |
| `direct_driver.support` | `validated_research` / `closed` | Bounded 24-fixture direct-driver compile/run support is checked; large-surface direct-driver execution and semantic authority remain explicitly not claimed. |
| `stdlib.surface` | `validated_research` / `closed` | Bounded stdlib support contract is checked; broad all-file stdlib callability remains explicitly not claimed. |
| `tooling.package` | `prototype` / `downgraded` | There is no public package registry/support contract. |
| `tooling.editor` | `prototype` / `downgraded` | Formatter, REPL, and editor tooling are prototype surfaces. |
| `install` | `prototype` / `downgraded` | Installation is repo-artifact based, not a broad supported distribution path. |
| `website.docs` | `prototype` / `downgraded` | Docs are extensive but still require readiness filtering before public support wording. |

Each row may be resolved in one of two honest ways:

1. Close the claim with evidence and move it to `stable` or
   `validated_research` with `closure_status=closed`.
2. Remove the surface from the release support contract and avoid claiming a
   broad language distribution for production use.

The gate intentionally does not count downgraded claims as release-ready simply
because they are honestly downgraded. Honest downgrade is sufficient for public
claim closure; it is not sufficient for a production release claim.

## What Already Passes

The narrower Madaros gate is green on the `main` this note measured:

- issue #356 is closed,
- the tracked BSS/global and self-build blockers are resolved,
- current `main` CI is green,
- the Madaros prebuilt refresh is green,
- compiler-owner overlap is clear,
- PR-resolution queue is contained,
- source-to-ELF semantic gate is green.

Those facts say the named gate passed. They do not say the language is recommended for production use. The public status is: research language (pre-1.0). APIs may change; not recommended for production use.

## Live Validation Result

Running the new gate with live checks on this branch reached this split:

- `madaros-production-ready`: pass.
- `docs-registry`: pass.
- `docs-consistency`: pass.
- `serious-language-claim-closure`: fail.

The claim-closure failure currently bottoms out in
`scripts/ci/serious_language_conformance_gate.sh`, where the bounded
serious-language conformance spine reported `4/16` passing cases. Failing claim
areas included structs, effect diagnostics, observe diagnostics, imports,
generics, GUM/Knowledge execution, ownership diagnostics, and epistemic boundary
diagnostics.

That conformance failure is a release-wide blocker independent of the seven
release-critical `prototype`/`downgraded` product-surface blockers above.

## Acceptance Gates For Closure

Broad release production readiness should not be declared until these are true:

- `scripts/ci/sounio_release_production_readiness_gate.sh` exits `0`.
- `scripts/dev/madaros_readiness_status.sh --no-audit --production-ready`
  exits `0`.
- `scripts/ci/serious_language_claim_closure_gate.sh` exits `0`.
- `scripts/dev/check_docs_registry.sh` exits `0`.
- `scripts/dev/check_docs_consistency.sh` exits `0`.
- Current `main` CI is green.
- Any release-critical claim in
  `docs/serious-language/public-claim-registry.v1.tsv` is either closed at
  `stable`/`validated_research` or intentionally removed from the broad release
  support contract.

## Current recommended public wording

> Status: research language (pre-1.0). APIs may change; not recommended for production use.

That line does not name a compiler version. A gate pass on the 2026-06-22 audit is not a public maturity label. Do not publish a separate sentence that calls the compiler or the language ready for production use.
