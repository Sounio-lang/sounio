<!-- docs:meta
topic_id: repo.docs.decisions.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.readme
-->

# Architecture Decision Records (ADRs)

Decision records for the Sounio compiler and language. Each ADR captures a
decision that was already made and validated — not a proposal.

## Format

```
ADR-NNN: Title
Status: accepted | experimental | superseded | deprecated
Date: YYYY-MM-DD
Superseded by: ADR-NNN (optional, only when status=superseded)
Context: why the decision was needed
Decision: what was chosen
Consequences: what follows
Grounded in: commit, artifact, or empirical result
```

## Status Convention

- `accepted`: current decision of record
- `experimental`: evidence-backed direction still under active validation and
  not yet the default long-term rule
- `superseded`: kept for lineage, but replaced by a later ADR
- `deprecated`: still historically relevant, but no longer recommended and not
  yet replaced by a single successor

## Maintenance Rules

- ADR numbering is append-only.
- Existing ADRs should not be rewritten to change history; new decisions should
  be captured in new ADRs.
- When an ADR becomes superseded, update its status and add `Superseded by`.
- The index below is the scan surface for current lifecycle state and should be
  kept in sync with the file headers.

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| 001 | Bundle-as-authority | accepted | 2026-03-30 |
| 002 | Truth layers are independent | accepted | 2026-03-30 |
| 003 | Wrapper provenance preserved | accepted | 2026-03-30 |
| 004 | Capacity guards over silent corruption | accepted | 2026-03-30 |
| 005 | `algebra` keyword is compiler infrastructure | accepted | 2026-03-30 |
| 006 | Self-hosting fixed-point as trust anchor | accepted | 2026-03-30 |
| 007 | Madaros second-order compilation | experimental | 2026-07-12 |

## Files

- [ADR-001](./adr-001-bundle-as-authority.md)
- [ADR-002](./adr-002-truth-layers-independent.md)
- [ADR-003](./adr-003-wrapper-provenance-preserved.md)
- [ADR-004](./adr-004-capacity-guards-over-silent-corruption.md)
- [ADR-005](./adr-005-algebra-keyword-compiler-infrastructure.md)
- [ADR-006](./adr-006-fixed-point-trust-anchor.md)
- [ADR-007](./adr-007-second-order-compilation.md)

## Related Docs

- [compiler-maturity-blueprint.md](../architecture/compiler-maturity-blueprint.md)
- [truth-layers.md](../architecture/truth-layers.md)
- [module-closure-truth.md](../architecture/module-closure-truth.md)
- [scientific-core.md](../architecture/scientific-core.md)
