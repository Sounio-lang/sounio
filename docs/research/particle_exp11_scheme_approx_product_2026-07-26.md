<!-- docs:meta
topic_id: repo.docs.research.particle-exp11-scheme-approx-product-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.particle-exp11-scheme-approx-product-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Particle EXP11 — Scheme × Approx product surface

**Date:** 2026-07-26  
**Source:** `examples/particle_physics/exp11_scheme_approx_product.sio`  
**Gate:** `scripts/ci/particle_exp11_scheme_approx_gate.sh`  
**Depends:** EXP8 (DeficitCollapse) + EXP10 layer semantics (residuals inlined; no approx_effects import for Madaros thin-link)

---

## Claim

Honest analysis is a **matrix**, not a scalar residual:

```
cell_residual = √( collapse_residual² + approx_combined² )
fails         = 1  iff  collapse status ≠ HOLDS
```

L4 (NU∧NWA tension) can be true on a **holding** scheme (fixed-width BW).

## Sample cells (Z, measured)

| Scheme | Approx | fails | tension | cell_res order |
|--------|--------|:-----:|:-------:|----------------|
| fixed | empty | 0 | 0 | smallest |
| fixed | triple | 0 | 1 | larger (approx only) |
| running | empty | 1 | 0 | collapse drives |
| running | triple | 1 | 1 | largest |
| interf α=0 | NU | 0 | 0 | control holds |
| interf α≠0 | NU/triple | 1 | 0/1 | fails |

nfail_cells in the scanned set = **4**.

## Type

```
SchemeApproxCell { scheme, collapse_status, collapse_residual, xi,
  approx_layers, approx_combined, approx_tension, cell_residual, fails }
nu_scheme_approx_cell(...)
```

## Non-claims

Not a full theory uncertainty budget; product of construction toys from EXP8/10.

## AI disclosure

Human direction 2026-07-26. GAIDeT-ICMJE 2025.
