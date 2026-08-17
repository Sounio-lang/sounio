<!-- docs:meta
topic_id: repo.docs.research.particle-exp20-deferred-2026-08-06
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.particle-exp20-deferred-2026-08-06
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Particle EXP20 — deferred (2026-08-06)

**Status:** DEFERRED (honesty halt)  
**Depends:** EXP17–19 Z/W/H stdlib amp ledgers (Madaros+lean green)

## Why not ship an EXP20 amp leaf today

1. **Top-channel (gg→tt̄ / similar)** needs an explicit, bounded flux × phase-space
   normalisation and NWA comparator. Inventing `|M|² · s/(12π)` would reopen the
   scalar/vector `4π` vs `12π` disagreement already logged on EXP16/19.
2. **Public `particle_physics::mod` re-export wiring** is blocked by module-resolution
   behaviour (even established exports failed a minimal consumer probe) — that is a
   compiler leaf, not a physics leaf.
3. Optical theorem / unitarity remain **NonUnitary non-claims** (Lean N×NWA leaf).

## What is green (no new claim)

```bash
bash scripts/research/particle_exp17_zwh_ledger_gate.sh
# PARTICLE_EXP17_GATE_OK  (lean_single + Madaros)
```

Ratios remain construction gaps: Z≈13.952395, W≈3.486637, H≈0.652209.

## Next when unblocked

- Compiler: public `mod` export resolution for `particle_physics`.
- Science: EXP20 top-channel with published flux/PS derivation + math-review
  **before** any continuum/NWA ratio claim.
