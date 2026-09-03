<!-- docs:meta
topic_id: repo.docs.research.particle-exp10-approx-algebra-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.particle-exp10-approx-algebra-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Particle EXP10 — Approximation effect algebra

**Date:** 2026-07-26  
**Source:** `examples/particle_physics/exp10_approx_effect_algebra.sio`  
**Gate:** `scripts/ci/particle_exp10_approx_algebra_gate.sh`  
**Status:** lean 30/30 full; Madaros algebra+core (physics import residual)

---

## Claim

Compiler effects `NonUnitary`, `Perturbative`, `NarrowWidthApproximation` already
propagate. EXP10 adds a **runtime algebra** of layers, residuals, and tension:

| Law | Stack | Note |
|---|---|---|
| L1 | singletons | always legal |
| L2 | {NWA, Pert} | e.g. H→bb NWA + LO α_s |
| L3 | {NU, Pert} | LO + unstable line |
| L4 | {NU, NWA} | typed-legal, **physically in tension** |
| L5 | triple | L4 + LO residual |

Combined residual: √(Σ r_i²) over active layers.

## Measured residuals

| Species | Γ/M (NWA residual) |
|---|---:|
| Z | ~0.027 |
| top | ~0.008 |
| H | ~3.3e-5 |

Triple stack combined residual ~1.04 (dominated by unitarity deficit=1 at pole).

## Live effect stack (compiler-enforced)

`main` and `run_effect_stack` declare all three effects and call:

- `alpha_s_lo_ep` (Perturbative)
- `z_partial_width_nwa_ep` (NWA)
- `nu_approx` / peak (NonUnitary)
- `h_bb_width_nwa_ep` (NWA+Pert)

## Types

```
ApproxStack { layers, residual_nu, residual_nwa, residual_pert, legal }
```

## Engine note

Madaros full run: algebra laws + effect-core (acks + NonUnitary) green.
Domain C (`alpha_s_lo` / NWA width Epistemic methods) may fail under Madaros
imported-module residual — gate accepts `PARTICLE_EXP10_PARTIAL_OK`.

## Non-claims

- Runtime layer tags are not a second effect system; compiler bits remain authority.
- `approx_pert_residual_lo_default = 0.30` is a construction scale, not a fit.
- L4 tension is physical honesty, not a type error.

## AI disclosure

Human direction 2026-07-26. GAIDeT-ICMJE 2025.
