<!-- docs:meta
topic_id: repo.docs.decisions.adr-006-fixed-point-trust-anchor
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-006-fixed-point-trust-anchor
-->

# ADR-006: Self-Hosting Fixed-Point as Trust Anchor

**Status**: accepted
**Date**: 2026-03-30

## Context

On 2026-03-23, lean_single.sio achieved self-compilation fixed point:
gen2 == gen3 bit-identical (md5=7b91e249, 230KB). The bootstrap chain:
boot4_a1 → gen1 → gen2 → gen3. gen1 is compiled by the old artifact (different
BSS layout). gen2 is compiled by M6-era code. gen3 is compiled by M6-era code
compiled by M6-era code. gen2 == gen3 means the compiler, when compiling itself,
produces a binary that compiles itself identically.

This is the strongest available proof that the compiler is self-consistent:
it is a fixed point of its own compilation function.

## Decision

The gen2==gen3 fixed-point check is the **trust anchor** for compiler changes.

- `artifacts/bootstrap/boot4.elf` is NOT updated until gen2==gen3 passes.
- Commit sequence: modify .sio first → validate fixed point → only then copy
  gen2.elf to artifact → commit .sio + .elf together.
- Any change to boot4.sio that breaks the fixed point is a regression, not a
  feature, regardless of what it improves.
- `git checkout artifacts/bootstrap/boot4.elf` is the instant rollback path.

## Consequences

- Every boot4.sio change requires the full chain: boot4 → gen1 → gen2 → gen3 →
  diff gen2 gen3.
- The fixed-point check is the M6 validation gate V1 (the most important gate).
- CI or validation scripts must run the chain, not just compile once.
- The artifact in `artifacts/bootstrap/boot4.elf` is a curated trust anchor,
  not a build cache — it changes only on verified fixed points.
- gen1 ≠ gen2 is expected and normal (different compiler compiled it). Only
  gen2 == gen3 matters.

## Grounded in

- 2026-03-23 fixed point: md5=7b91e249, 230KB, lean_single.sio
- Bootstrap chain: `bootstrap/boot4.sio` → `artifacts/bootstrap/boot4.elf`
- M6 validation gate V1: `scripts/ci/m6_closure_v2_validation.sh` checks 1-3
- Earlier fixed point: boot4_a1 chain md5=2c108e21, 210KB (2026-03-23)
