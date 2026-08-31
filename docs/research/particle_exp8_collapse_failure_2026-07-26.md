<!-- docs:meta
topic_id: repo.docs.research.particle-exp8-collapse-failure-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.particle-exp8-collapse-failure-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Particle EXP8 — DeficitCollapse failure surface + SchemeTension

**Date:** 2026-07-26  
**Orthography:** EN-UK  
**Status:** construction vertical (`PARTICLE_EXP8_OK`, 20/20)  
**Source:** `examples/particle_physics/exp8_deficit_collapse_failure.sio`  
**Gate:** `scripts/ci/particle_exp8_collapse_gate.sh`

---

## Why this is deeper than EXP6

EXP6 shows pure Breit-Wigner ξ-collapse. That identity cannot fail for fixed-width BW.
EXP8 introduces **objects that go red**:

| Scheme | At ξ=2 (Z) | Status |
|---|---|---|
| Fixed width | residual ~1e-16 | **HOLDS** |
| Running Γ(s)=Γ₀·s/M² | residual ~0.018 | **FAILS_WIDTH_SCHEME** |
| Interference α=0.4, m₂=M_Z+10 | residual > tol at ξ=1 | **FAILS_INTERFERENCE** |
| Interference α=0 | residual ~0 | **HOLDS** (control) |

## Types

```
DeficitCollapse { status, residual, scheme, xi, d_scheme, d_analytic }
SchemeTension   { scheme_a/b, val_a/b, var_a/b, gap_sigma, consistent }
```

Status: `0 HOLDS | 1 FAILS_WIDTH_SCHEME | 2 FAILS_INTERFERENCE`.

## SchemeTension snapshot

| Pair | gap σ | consistent |
|---|---:|---:|
| fixed vs running (deficit @ ξ=2) | ~6.0 | 0 |
| fixed vs interference @ ξ=1 | ~7.9 | 0 |
| M_W tree pred vs G_F pred | ~88 | 0 |
| fixed vs fixed | 0 | 1 |

Non-isomorphism (printed): `SchemeTension_is_not_EpistemicTension_vs_measurement`
— scheme gap is between two predictions, not pred vs PDG.

## Non-claims

- Not a full SM lineshape generator.
- Running-width and two-pole toys are **falsifiers of pure-BW universality**, not precision fits.
- No BSM Z' discovery claim.

## Reproduce

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
./bin/souc run examples/particle_physics/exp8_deficit_collapse_failure.sio
bash scripts/ci/particle_exp8_collapse_gate.sh
```

## AI disclosure

Assembled under human direction (2026-07-26). GAIDeT-ICMJE 2025.
