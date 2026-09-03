<!-- docs:meta
topic_id: repo.docs.research.particle-exp16-h-amp-to-xsec-2026-08-05
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.particle-exp16-h-amp-to-xsec-2026-08-05
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Particle EXP16 — Higgs observable from amplitude (`amp → σ`)

**Date:** 2026-08-05  
**Source:** `examples/particle_physics/exp16_h_amp_to_xsec.sio`  
**Gate:** `scripts/research/particle_exp16_h_amp_xsec_gate.sh`  
**Receipt:** `examples/particle_physics/results/exp16_h_amp_to_xsec.json`  
**Depends:** EXP14/15 pattern + `nu_higgs_propagator`

---

## Claim

The Higgs twin of the Z/W amp→σ leaves: continuum cross-section from the
Breit-Wigner propagator with a Yukawa-bb numerator

```
y_b = √2 m_b / v
num = y_b⁴
|M|² = num · |D_H|²
σ(s) = |M|²(s) · s / (12π)
```

kept distinct from the bb NWA peak toy

```
σ_peak = 12π · Γ_bb² / (M_H² · Γ_H²)
Γ_bb   = Br(H→bb) · Γ_H ,   Br ≈ 0.5824,   Γ_H = 0.00407 GeV
```

Measured at the H pole (lean_single + Madaros, 2026-08-05):

| Quantity | Value |
|---|---:|
| `σ_from_amp` | 5.32e-4 GeV⁻² |
| `σ_peak` | 8.16e-4 GeV⁻² |
| `σ_from_amp / σ_peak` | **0.652209** |
| off-pole | ξ = 2 (not Δs/s = 1% — vacuous for Γ/M ~ 3×10⁻⁵) |

Honesty band for the ratio: `(0.3, 2)`. Continuum sits **below** NWA for this
Yukawa toy; equality is **not** claimed.

## Pillars (5/5)

| ID | Check |
|---|---|
| P1 | `|M|²` at pole in `(1e-9, 1e-3)` |
| P2 | `σ_from_amp` in `(1e-6, 1e-2)` |
| P3 | peak finite and `ratio ∈ (0.3, 2)` |
| P4 | `σ(ξ=2) < σ(pole)` |
| P5 | `Var(|M|²) > 0` (from `M_H` in propagator) |

## Engine note

| Surface | Status |
|---|---|
| lean_single **run** | green (`PARTICLE_EXP16_OK`, 5/5) |
| Madaros **run** | green (`mass_h` + `nu_higgs_propagator`) |
| Gate | `PARTICLE_EXP16_GATE_OK` |

Closes the thin-graph trio **Z / W / H** (EXP14 / EXP15 / EXP16).
EXP19 restores the same H continuum path via `h_bb_yukawa_amplitude_nu`
(stdlib Yukawa + GUM `mass_bottom`); this leaf stays the thin local-num witness.

## Non-claims

- Not a full H→bb¯ helicity amplitude; not gg→H / VBF production.
- Not NLO / QCD / PDF / interference with continuum bb.
- Does not equate continuum BW to the partial-width NWA peak.
- Not a BSM bound.

## Reproduce

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
bash scripts/research/particle_exp16_h_amp_xsec_gate.sh
# expect: PARTICLE_EXP16_GATE_OK
```

## LLM-offload

`math-review` via xai (receipt `examples/particle_physics/results/exp16_math_review_offload.txt`).
Canonical log append deferred (shared-file claim).

## AI disclosure

Human direction 2026-08-05. GAIDeT-ICMJE 2025.
