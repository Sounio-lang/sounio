<!-- docs:meta
topic_id: repo.docs.research.particle-exp15-w-amp-to-xsec-2026-08-05
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.particle-exp15-w-amp-to-xsec-2026-08-05
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Particle EXP15 — W observable from amplitude (`amp → σ`)

**Date:** 2026-08-05  
**Source:** `examples/particle_physics/exp15_w_amp_to_xsec.sio`  
**Gate:** `scripts/research/particle_exp15_w_amp_xsec_gate.sh`  
**Receipt:** `examples/particle_physics/results/exp15_w_amp_to_xsec.json`  
**Depends:** EXP14 (Z amp→σ pattern) + `nu_w_propagator`

---

## Claim

The W twin of EXP14: continuum cross-section from the Breit-Wigner propagator
with a charged-current numerator

```
num = (g²/2)² = g⁴/4
|M|² = num · |D_W|²
σ(s) = |M|²(s) · s / (12π)
```

kept distinct from the leptonic NWA peak toy

```
σ_peak = 12π · Γ_eν · Γ_μν / (M_W² · Γ_W²)
Γ_ℓν   = Br(W→ℓν) · Γ_W ,   Br ≈ 0.1086,   Γ_W = 2.085 GeV
```

Measured at the W pole (lean_single + Madaros, 2026-08-05):

| Quantity | Value |
|---|---:|
| `|M|²` | 1.4×10⁻⁶ GeV⁻⁴ (print may round) |
| `σ_from_amp` | 2.40×10⁻⁴ GeV⁻² |
| `σ_peak` | 6.9×10⁻⁵ GeV⁻² |
| `σ_from_amp / σ_peak` | **3.486629** |
| `σ(1.01 M_W²)` | 2.11×10⁻⁴ GeV⁻² (< pole) |

Honesty band for the ratio: `(2, 6)`. Equality is **not** claimed.

## Pillars (5/5)

| ID | Check |
|---|---|
| P1 | `|M|²` at pole in `(1e-9, 1e-3)` |
| P2 | `σ_from_amp` in `(1e-6, 1e-2)` |
| P3 | peak finite and `ratio ∈ (2, 6)` |
| P4 | `σ(1.01 M_W²) < σ(M_W²)` |
| P5 | `Var(|M|²) > 0` (from `M_W` in propagator) |

## Engine note

| Surface | Status |
|---|---|
| lean_single **run** | green (`PARTICLE_EXP15_OK`, 5/5) |
| Madaros **run** | green (EXP12-thin graph: `mass_w` + `nu_w_propagator`) |
| Gate | `PARTICLE_EXP15_GATE_OK` |

Same lean_single selective-import discipline as EXP14: PDG-central `g = 0.629773`
inlined; GUM via `Var(cX) = c² Var(X)` on the propagator.

## Non-claims

- Not a full eν→μν helicity amplitude (angular structure omitted).
- Not NLO / QCD / PDF.
- EXP18 restores the same continuum path via `cc_w_leptonic_amplitude_nu`
  (`coupling_g` GUM); this leaf keeps the inlined-`g` thin graph.
- Does not equate continuum BW to the partial-width NWA peak.
- Not a BSM bound.
- No `eemm_w_peak_xsec_nu` in stdlib — peak is a local PDG toy.

## Reproduce

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
bash scripts/research/particle_exp15_w_amp_xsec_gate.sh
# expect: PARTICLE_EXP15_GATE_OK
```

## LLM-offload

`math-review` via xai on a four-question extract (receipt
`examples/particle_physics/results/exp15_math_review_offload.txt`).
Canonical `.claude/llm_offload_log.md` append deferred (shared-file claim).

## AI disclosure

Human direction 2026-08-05. GAIDeT-ICMJE 2025.
