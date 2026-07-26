<!-- docs:meta
topic_id: repo.docs.research.particle-exp7-gum-transfer-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.particle-exp7-gum-transfer-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Particle EXP7 — GUM ξ + EXP3↔EXP6 transfer

**Date:** 2026-07-26  
**Orthography:** EN-UK  
**Status:** construction vertical (`PARTICLE_EXP7_OK`, 30/30)  
**Source:** `examples/particle_physics/exp7_gum_xi_tension_transfer.sio`  
**Gate:** `scripts/ci/particle_exp7_gum_transfer_gate.sh`

---

## Two objects (both shipped)

### A — GUM-propagated reduced variable

Stdlib:

| Function | Meaning |
|---|---|
| `nu_reduced_xi_ep(s, M, Γ)` | ξ with Var from mass GUM |
| `nu_deficit_analytic_xi_ep(ξ)` | d=1/(1+ξ²) with Var chain rule |
| `nu_unitarity_threshold_ep(M, Γ, t)` | √s thr as Epistemic |

Measured (Z, lean_single):

| Quantity | Central | Var |
|---|---:|---:|
| ξ(pole) | 0 | >0 |
| ξ=1 | 1 | >0 |
| d(ξ=1) | 0.5 | >0 |
| d(pole) | 1 | ~0 (∂d/∂ξ=0) |
| thr 1% | ~102.85 GeV | >0 |

### B — Transfer receipt (not isomorphism)

**Claim:** the EXP3 M_W EpistemicTension ladder (tree → ρ → G_F) can improve
pull while the EXP6 universal ξ-shape of NonUnitary deficit remains intact.

| Construction | M_W pred | pull |
|---|---:|---:|
| tree | 79.954 | −34.35 |
| on-shell ρ | 80.301 | −6.17 |
| G_F Sirlin | 80.362 | −1.18 |

At matched ξ, residual |d_Z − d_W| ~ 10⁻¹⁶. Analytic d(ξ) at construction
masses (self-consistent s(ξ; M_pred, Γ)) is construction-invariant (= 0.5 at ξ=1).

**Non-isomorphism (printed):** `EpistemicTension_is_not_NonUnitary_deficit` —
shared receipt dashboard only; no type identification.

JSON: `particle.exp7.transfer_tension_xi.v1`

## Reproduce

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
./bin/souc run examples/particle_physics/exp7_gum_xi_tension_transfer.sio
bash scripts/ci/particle_exp7_gum_transfer_gate.sh
```

## AI disclosure

Assembled under human direction (2026-07-26). GAIDeT-ICMJE 2025.
