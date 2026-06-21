<!-- docs:meta
topic_id: repo.docs.release-policy
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.release-policy
-->

# Sounio Release Policy

## Versioning

Sounio follows [Semantic Versioning 2.0.0](https://semver.org/) with pre-release tags:

| Tag | Meaning | Gate Requirement |
|-----|---------|------------------|
| `alpha` | Feature-incomplete, API unstable | `souc check` passes on canonical fixtures |
| `beta` | Feature-complete, API may change | stdlib reliability gate pass (0 fail) |
| `rc` | Release candidate, API frozen | All CI gates green + manual review |
| *(none)* | Stable release | RC validated + Zenodo DOI minted |

**Current version:** See `CITATION.cff` (single source of truth for version metadata).

The checked-in JIT artifact (`bin/souc`) is not rebuilt for every changelog entry. Its `--version` output may trail the source version by one or two pre-release increments. The discrepancy is documented in `README.md`.

## Release Cadence

Releases are **event-driven**, not calendar-driven. A release is cut when:

1. A significant feature milestone is reached (new stdlib domain, compiler backend, paper submission)
2. A critical bug fix warrants immediate distribution
3. The release dashboard job in `.github/workflows/release-gate.yml` reports
   green, including zero open issues labeled `release-blocker`

There is no fixed cadence. Research languages evolve in bursts aligned with paper deadlines and sprint completions.

Manual `Release Gate` runs may set `native-runtime-filter` to narrow the Apple
Self-Host native runtime proof during blocker triage. Filtered runs are
diagnostic evidence only; an unfiltered release gate is still required before a
release can be considered ready.

## Release Checklist

Before tagging a release:

- [ ] All stdlib reliability gates pass: `artifacts/stdlib/stdlib_reliability_status.v1.json` reports `status_summary=pass`
- [ ] Science pipeline gates pass: `fmri` and `darwin_pbpk` lanes green
- [ ] `CHANGELOG.md` has an entry for the new version with date
- [ ] `CITATION.cff` version and date-released match the new tag
- [ ] Language specification (`docs/spec/LANGUAGE_SPECIFICATION.md`) is current
- [ ] `docs/guide/MINIMUM_VIABLE_SOUNIO.md` reflects the validated feature set
- [ ] No `//@ ignore` test regressions introduced since last release
- [ ] JIT artifact version discrepancy (if any) documented in README

## Artifact Policy

### Shipped Binaries

| Artifact | Path | Purpose |
|----------|------|---------|
| JIT compiler | `bin/souc` | Default: Cranelift JIT backend |
| GPU compiler | `artifacts/omega/souc-bin/souc-linux-x86_64-gpu` | PTX/CUDA codegen |
| Native compiler | `artifacts/omega/souc-bin/souc-linux-x86_64` | Full native x86-64 backend |

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
- **Backports**: Not provided. Users are expected to track the latest release.
- **Commercial support**: Not available.

## Deprecation

The stdlib uses the `.disabled` file extension convention for deprecated modules. A file renamed from `foo.sio` to `foo.sio.disabled` is:

1. Excluded from `souc check` and stdlib gates
2. Retained in the repository for reference
3. Eligible for permanent removal after two releases

Currently 120 files use this convention.

## Version Metadata Locations

To avoid drift, version is maintained in exactly one authoritative location:

| File | Field | Role |
|------|-------|------|
| `CITATION.cff` | `version`, `date-released` | **Authoritative** |
| `CHANGELOG.md` | Section header | Release history |
| `README.md` | Prose note | Documents binary/source discrepancy |
| souc binary | `--version` output | Build-time snapshot (may lag) |
