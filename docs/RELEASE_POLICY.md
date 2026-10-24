<!-- docs:meta
topic_id: repo.docs.release-policy
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.release-policy
-->

# Sounio Release Policy

## Versioning

The normative rules are in [ADR-018](decisions/adr-018-release-cycle-and-versioning.md). In summary:

- **Toolchain** (`souc`, Madaros, stdlib, runtime): one [Semantic Versioning 2.0.0](https://semver.org/) number, `MAJOR.MINOR.PATCH[-PRE]`, with `CITATION.cff` as its single source of truth. Engine build identifiers such as `madaros 4f1c2e9a` follow the version and never replace it.
- **Language**: editions named by year (`edition = "2026"` in `sounio.toml`). Edition 2026 is the language of 3.0.0. Edition 2027 (ADR-015) arrives as an opt-in in 3.x and becomes the default in 4.0.0.
- **Pre-release identifiers**: `alpha.N` (API unstable), `beta.N` (feature-complete), `rc.N` (frozen). Nothing else. Dated and engine-named tags are frozen history.

| Bump | Allowed changes (per edition) |
|------|-------------------------------|
| **Major** | anything, every break listed under *Breaking* |
| **Minor** | additions; `semantics` changes declared in a fragment; `refusal` (soundness) changes; numerically consistent changes |
| **Patch** | fixes; `numerical-correction` toward an exact oracle; bit-identical or numerically consistent output |

"Numerically consistent" is the epistemic test of ADR-018 §3. Shared provenance components are treated as fully correlated and cancel, so only the declared `numerics` component can absorb a difference. Without that component, the test reduces to exact equality.

## Release Cadence: feature trains

Releases are scoped by feature and bounded by a freeze date:

1. **Kickoff.** Open the milestone (`3.1`, `3.2`, …) in GitHub and in Notion with its scope and **freeze date**.
2. **Freeze.** Cut `release/X.Y` from `main`. Only fixes labelled `backport:X.Y` are cherry-picked onto it, and `main` becomes the next train. Anything not merged by the freeze date moves to the next train; a train never waits.
3. **Release candidate.** Tag `vX.Y.0-rc.1` once the RC gates pass. Soak for **at least 7 days**. Every fix merged into the branch produces `rc.N+1` and restarts the soak.
4. **Stable.** Tag `vX.Y.0` when the soak ends clean and the Zenodo DOI is minted.
5. **Patches.** Tag `vX.Y.Z` from `release/X.Y`. These need the RC gates but no soak.

Manual `Release Gate` runs may set `native-runtime-filter` to narrow the Apple
Self-Host native runtime proof during blocker triage. Filtered runs are
diagnostic evidence only; an unfiltered release gate is still required before a
release can be considered ready.

## Release Checklist

**RC gates**, all on the head of `release/X.Y`:

- [ ] Unfiltered `Release Gate` (`.github/workflows/release-gate.yml`) green
- [ ] Zero open issues labelled `release-blocker`
- [ ] `artifacts/stdlib/stdlib_reliability_status.v1.json` regenerated on this head and reporting `status_summary=pass`
- [ ] Science pipeline gates pass: `fmri` and `darwin_pbpk` lanes green
- [ ] Compatibility check (ADR-018 §3) passes against the previous release's baseline, with every failure covered by a fragment of the right kind (from 3.0.1 on)
- [ ] `CHANGELOG.md` section assembled with `scripts/release/changelog_assemble.sh`
- [ ] `CITATION.cff` `version` and `date-released` match the tag
- [ ] Language specification (`docs/spec/LANGUAGE_SPECIFICATION.md`) and `docs/guide/MINIMUM_VIABLE_SOUNIO.md` reflect the edition shipped
- [ ] No `//@ ignore` test regressions introduced since the last release

**Stable gates**, in addition:

- [ ] Soak of at least 7 days since the last `rc.N`, with no new `release-blocker`
- [ ] `souc --version` reports the same version as `CITATION.cff`
- [ ] Zenodo version DOI minted and recorded in `CITATION.cff`
- [ ] GitHub Release body equal to the CHANGELOG section
- [ ] Compatibility baseline recorded for the next release

## Artifact Policy

### Shipped Binaries

| Artifact | Path | Purpose |
|----------|------|---------|
| Native compiler (default) | `artifacts/omega/souc-bin/souc-linux-x86_64` | **Shipped engine**: self-hosted native-v2 (Madaros) x86-64 backend |
| GPU compiler | `artifacts/omega/souc-bin/souc-linux-x86_64-gpu` | PTX/CUDA codegen |
| Wrapper (legacy path) | `bin/souc` | A bash wrapper that routes to Madaros — **not** a Cranelift artifact and not an artifact at all. Development/reference entry point; not the shipped engine |

> **The Cranelift row was removed rather than demoted — measured 2026-08-27.** It
> read "Cranelift JIT backend"; `bin/souc` is a 318-line bash script whose
> `--version` reports `elf=bin/madaros-linux-x86_64`, and no binary in this tree
> has a Cranelift backend compiled in. Demoting the claim from "default" to
> "legacy" left an object that does not exist still named in the Artifact Policy.

### Signing and Provenance

- SHA-256 checksums: `*.sha256` alongside each binary
- Signature files: `*.sig`
- Release provenance: `artifacts/omega/souc_release_provenance*.json`

### Zenodo

Each stable release is archived on Zenodo with a version-specific DOI. The concept DOI (`10.5281/zenodo.18190065`) resolves to the latest version.

## Support Model

Sounio is a **research language**. Support expectations:

- **Bug reports**: Accepted via GitHub Issues. No response-time SLA.
- **Security issues**: Report via `SECURITY.md`. Best-effort response.
- **Feature requests**: Welcomed but prioritized by research program alignment.
- **Maintained releases**: the current minor receives every fix. The previous minor receives only `numerical-correction` and `security` fixes, until the next minor ships, so that work citing it is corrected rather than merely superseded. Older releases receive nothing.
- **Commercial support**: Not available.

## Deprecation

The stdlib uses the `.disabled` file extension convention for deprecated modules. A file renamed from `foo.sio` to `foo.sio.disabled` is:

1. Excluded from `souc check` and stdlib gates
2. Retained in the repository for reference
3. Eligible for permanent removal after two releases

Currently 120 files use this convention.

## Version Metadata Locations

| File | Field | Role |
|------|-------|------|
| `CITATION.cff` | `version`, `date-released`, version DOI | **Authoritative** |
| `CHANGELOG.md` | Section header | Release history, assembled from `changelog.d/` |
| `changelog.d/` | One fragment per PR | Unreleased changes |
| souc binary | `--version` output | Must equal `CITATION.cff` at a release tag |
