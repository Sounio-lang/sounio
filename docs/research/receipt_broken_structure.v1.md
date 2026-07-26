<!-- docs:meta
topic_id: repo.docs.research.receipt-broken-structure.v1
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipt-broken-structure.v1
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Receipt schema — `sounio.broken_structure.v1`

**Date:** 2026-07-25  
**Orthography:** EN-UK  
**Status:** construction contract (not a paper claim)

---

## Purpose

One **machine-readable receipt shape** for two domains that both require
*typed honesty about singular / non-unitary structure*:

| Domain tag | Coordinate | Proximity ∈ [0,1] | Substrate |
|---|---|---|---|
| `qft_unstable` | √s (GeV) | NonUnitary deficit | `particle_physics::nonunitary` |
| `sedenion_zd` | path parameter t ∈ [0,1] toward ZD pair | 1/(1+dist) to annihilation | `algebra::sedenion` |

## Schema (JSON object)

```json
{
  "schema": "sounio.broken_structure.v1",
  "domain": "qft_unstable | sedenion_zd | ossm_silence",
  "species_or_fiber": "string",
  "coordinate": "sqrt_s | path_t | fiber_param | time",
  "effect_or_type": "NonUnitary | ZeroDivisorProximity | ...",
  "points": [
    { "label": "string", "coord": 0.0, "proximity": 0.0 }
  ],
  "non_isomorphism": "required disclaimer string when dual-gated"
}
```

### Field rules

- `schema` must be exactly `sounio.broken_structure.v1`
- `proximity` ∈ [0, 1] (gate-enforced)
- For `qft_unstable`: pole point proximity ≈ 1; far points < pole
- For `sedenion_zd`: at path endpoint on canonical ZD, proximity ≈ 1; product norm ≈ 0
- Dual gate requires **both** domains present and the **non_isomorphism** disclaimer

## Explicit non-isomorphism (mandatory)

**Breit–Wigner NonUnitary deficit is not a sedenion zero-divisor.**

They share only:

1. A **receipt geometry** (coordinate → proximity).  
2. A **refusal to treat singular structure as ordinary** (effect / algebraic fact).  
3. A **construction discipline** (falsifiers + gate).

They do **not** share:

- The same algebra  
- The same measure  
- Any claim of physical identity  

Violating this section is a claim error, not a style preference.

## Producers

| Domain | Producer |
|---|---|
| QFT | `examples/particle_physics/exp4_unstable_spectrum.sio`, dual `exp5_*` |
| 𝕊 ZD | `examples/particle_physics/exp5_broken_structure_dual.sio` |

## Gate

```bash
bash scripts/ci/particle_broken_structure_dual_gate.sh
```

## AI disclosure

Assembled under human direction (2026-07-25). GAIDeT-ICMJE 2025.
