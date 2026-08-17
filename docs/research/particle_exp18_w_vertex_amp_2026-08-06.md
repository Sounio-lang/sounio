<!-- docs:meta
topic_id: repo.docs.research.particle-exp18-w-vertex-amp-2026-08-06
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.particle-exp18-w-vertex-amp-2026-08-06
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Particle EXP18 — W amp→σ via `cc_w_leptonic_amplitude_nu`

**Date:** 2026-08-06  
**Source:** `examples/particle_physics/exp18_w_vertex_amp_to_xsec.sio`  
**Gate:** `scripts/research/particle_exp18_w_vertex_amp_gate.sh`  
**Receipt:** `examples/particle_physics/results/exp18_w_vertex_amp_to_xsec.json`  
**Depends:** EXP15 (W honesty band) + E175 trilogy (Madaros amp import)

---

## Claim

W twin of EXP14's Z restore: continuum cross-section from stdlib amplitude

```
|M|² = (g²/2)² · |D_W|² = (g⁴/4) · |D_W|²
σ(s) = |M|²(s) · s / (12π)
```

with `g` from `coupling_g()` (GUM) via `cc_w_leptonic_amplitude_nu`, kept
distinct from the local leptonic NWA peak toy (same as EXP15).

Measured (lean_single + Madaros, 2026-08-06):

| Quantity | Value |
|---|---:|
| `σ_from_amp` | 2.40×10⁻⁴ GeV⁻² |
| `σ_peak` | 6.9×10⁻⁵ GeV⁻² |
| `σ_from_amp / σ_peak` | **3.486637** (EXP15 local-num: 3.486629) |
| honesty band | `(2, 6)` |

Tiny ratio drift vs EXP15 is GUM on `g` vs inlined PDG scalar — not a bug.

## Pillars (5/5)

| ID | Check |
|---|---|
| P1 | `|M|²` pole in `(1e-9, 1e-3)` |
| P2 | `σ` in `(1e-6, 1e-2)` |
| P3 | peak finite and `ratio ∈ (2, 6)` |
| P4 | `σ(1.01 M_W²) < σ(M_W²)` |
| P5 | `Var(|M|²) > 0` |

## Non-claims

- Not a full eν→μν helicity amplitude.
- Not NLO / PDF.
- Does not migrate EXP15 (thin local-num leaf remains a regression witness).
- H Yukawa stdlib amplitude is EXP19 (`h_bb_yukawa_amplitude_nu`).

## Reproduce

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
bash scripts/research/particle_exp18_w_vertex_amp_gate.sh
# expect: PARTICLE_EXP18_GATE_OK
```
