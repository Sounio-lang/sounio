<!-- docs:meta
topic_id: repo.docs.research.particle-exp6-universal-xi-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.particle-exp6-universal-xi-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Particle EXP6 — Universal reduced-variable NonUnitary deficit

**Date:** 2026-07-26  
**Orthography:** EN-UK  
**Status:** construction vertical (`PARTICLE_EXP6_OK`, 41/41)  
**Source:** `examples/particle_physics/exp6_universal_deficit_xi.sio`  
**Gate:** `scripts/ci/particle_exp6_universal_xi_gate.sh`  
**Stdlib:** `nu_reduced_xi`, `nu_deficit_analytic_xi`, `nu_s_from_xi` in `nonunitary.sio`

---

## Claim (executable, not a journal paper)

For pure Breit-Wigner NonUnitary propagators, the unitarity deficit

```
d = (M·Γ)² / [(s − M²)² + (M·Γ)²]
```

depends only on the reduced variable

```
ξ = (s − M²) / (M · Γ)
```

via the universal curve

```
d(ξ) = 1 / (1 + ξ²)
```

Therefore **Z, W, H, and t** evaluated at the **same ξ** must share the same deficit
(within numerical noise). That is the construction object: a cross-species collapse
receipt, not a BSM claim.

## Measured snapshot (lean_single + Madaros, 2026-07-26)

| ξ | d analytic | d_Z | d_W |
|---:|---:|---:|---:|
| −1 | 0.5 | 0.5 | 0.5 |
| 0 | 1.0 | 1.0 | 1.0 |
| 2 | 0.2 | 0.2 | 0.2 |

Cross-species residual `|d_Z − d_W|` at matched ξ is ~10⁻¹⁶.

## Novelty surface

1. **Universal ξ object** in stdlib (`nu_reduced_xi` / `nu_deficit_analytic_xi` / `nu_s_from_xi`).  
2. **Four-species scan** (Z, W, H, t) on one ξ grid.  
3. **Cross-species JSON residual** (`particle.exp6.cross_species_xi.v1`).  
4. **Madaros full run green** on this vertical (no local peak workaround needed).

## Non-claims

- No statement that real higher-order widths preserve exact collapse.  
- No isomorphism with sedenion zero-divisors (see EXP5 dual receipt).  
- No BSM / new particle.

## Reproduce

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
./bin/souc run examples/particle_physics/exp6_universal_deficit_xi.sio
bash scripts/ci/particle_exp6_universal_xi_gate.sh
```

## AI disclosure

Assembled under human direction (2026-07-26). GAIDeT-ICMJE 2025.
