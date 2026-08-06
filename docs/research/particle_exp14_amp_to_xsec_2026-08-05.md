<!-- docs:meta
topic_id: repo.docs.research.particle-exp14-amp-to-xsec-2026-08-05
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.particle-exp14-amp-to-xsec-2026-08-05
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Particle EXP14 — Observable from amplitude (`amp → σ`)

**Date:** 2026-08-05  
**Source:** `examples/particle_physics/exp14_amp_to_xsec.sio`  
**Gate:** `scripts/research/particle_exp14_amp_xsec_gate.sh`  
**Receipt:** `examples/particle_physics/results/exp14_amp_to_xsec.json`  
**Depends:** EXP13 (amplitude honesty) + EXP12 peak path (Madaros-safe NonUnitary)

---

## Claim

The physical observable after a Madaros-safe `|M|²` is the continuum cross-section

```
σ(s) = |M|²(s) · s / (12π)
```

Honest analysis keeps this distinct from the NWA peak-widths formula

```
σ_peak = 12π · Γ_ee · Γ_μμ / (M_Z² · Γ_Z²)
```

Measured at the Z pole (lean_single + Madaros, 2026-08-05):

| Quantity | Value |
|---|---:|
| `|M|²` | 3.249461×10⁻⁷ GeV⁻⁴ |
| `σ_from_amp` | 7.2×10⁻⁵ GeV⁻² |
| `σ_peak` (PDG partial widths) | 5.2×10⁻⁶ GeV⁻² |
| `σ_from_amp / σ_peak` | **13.952363** |
| `σ(1.01 M_Z²)` | 6.4×10⁻⁵ GeV⁻² (< pole) |

The ratio is a construction gap, not a bug: continuum BW with `g⁴` couplings versus
partial-width NWA. EXP14 asserts the gap sits in `[10, 20]`, and does **not** claim
equality.

Under lean_single, a separate probe showed `σ_from_amp` matches
`eemm_z_total_cross_section` at ratio `1.000` (not on the dual-engine gate — that
import pulls `amplitude`/`vertex`/`spinor`; see Non-claims).

## Pillars (5/5)

| ID | Check |
|---|---|
| P1 | `|M|²` at pole in `(1e-9, 1e-4)` |
| P2 | `σ_from_amp` in `(1e-6, 1e-3)` |
| P3 | peak finite and `ratio ∈ (10, 20)` |
| P4 | `σ(1.01 M_Z²) < σ(M_Z²)` |
| P5 | `Var(|M|²) > 0` |

## Engine note

| Surface | Status |
|---|---|
| lean_single **run** | green (`PARTICLE_EXP14_OK`, 5/5) |
| Madaros **run** | green — imports `nonunitary_amp::eemm_z_amplitude_nu` (E175 closed 2026-08-05/06) |
| Gate | `PARTICLE_EXP14_GATE_OK` (both engines) |
| Imported `eemm_z_amplitude_nu` under Madaros | **closed** — private `extern "C"` sqrt stubs removed from particle_physics; see `particle_e175_amp_import_2026-08-05.md` / trilogy |

## Non-claims

- Not a full NLO / ISR / beamstrahlung cross-section.
- Does not equate continuum BW to the partial-width NWA peak.
- Not a BSM bound.

## Reproduce

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
bash scripts/research/particle_exp14_amp_xsec_gate.sh
# expect: PARTICLE_EXP14_GATE_OK
```

## LLM-offload

`math-review` via xai/Grok 4.3 on a four-question extract: **Q1–Q4 all [OK]**
(conversion identity; NWA≠continuum honesty band; GUM exact-multiplier rule; no overclaim).
Canonical log row blocked by active claim on `.claude/llm_offload_log.md` (CS6 lane);
raw receipt: `examples/particle_physics/results/exp14_math_review_offload.txt`.

## AI disclosure

Human direction 2026-08-05. GAIDeT-ICMJE 2025.
