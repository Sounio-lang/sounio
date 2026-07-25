<!-- docs:meta
topic_id: repo.docs.research.particle-exp123-vertical-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.particle-exp123-vertical-2026-07-25
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Particle physics vertical — EXP1+2+3 (executable)

**Date:** 2026-07-25  
**Orthography:** EN-UK  
**Status:** `PARTICLE_EXP123_OK` (51/51 checks under lean_single)  
**Source:** `examples/particle_physics/exp123_z_metrology_nonunitary_ew.sio`  
**Gate:** `scripts/ci/particle_exp123_gate.sh`  
**Stdlib:** `m_w_prediction_ep_radiative`, `m_w_consistency_pull_radiative` in `ew_precision.sio`

---

## Purpose

Construction-first: three real experiments on `stdlib/particle_physics` with
GUM / effects discipline, where novelty can grow without paper-first pressure.

| Exp | Physics | Novelty surface |
|---|---|---|
| **1** | Γ(Z→ee) metrology + uncertainty budget + confidence gate | GUM provenance to observable; budget of PDG sources |
| **2** | Non-unitarity at Z pole: deficit(s), peak σ with `NonUnitary`, deficit vs √s curve | Effect-typed unstable intermediate; compiler-enforced honesty |
| **3** | EW tension: tree vs radiative (Δρ) M_W, pull collapse, S/T/U, Δρ | Tension as first-class numeric object; Δρ as construction step |

---

## Reproduce

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
export SOUNIO_SOUC_ENGINE=lean_single   # recommended for this package

./bin/souc run examples/particle_physics/exp123_z_metrology_nonunitary_ew.sio
# expect: PARTICLE_EXP123_OK  (51 PASS)

bash scripts/ci/particle_exp123_gate.sh
# expect: PARTICLE_EXP123_GATE_OK
```

**Engine note:** Madaros may block `examples/` → scientific modules under
science-boundary preflight (`E-SRB-002`). lean_single is the validated path for
this vertical. Forcing Madaros requires an approved science-boundary receipt.

---

## Measured snapshot (lean_single, 2026-07-25)

### EXP1 — Z metrology

| Quantity | Value |
|---|---|
| Γ(Z→ee) | 0.08341 GeV |
| σ(Γ) | 0.00098 GeV |
| confidence | 846 |
| budget Var(M_Z) | 3.32e-11 |
| budget Var(sin²θ_W) | 3.98e-12 |
| **dominant source** | **M_Z** |
| gate @ 800 | pass |
| gate @ 9999 | fail (correct) |

### EXP2 — NonUnitary + deficit vs √s

| Quantity | Value |
|---|---|
| deficit at pole | **1.000** |
| deficit far / high | 7.5e-4 / 6e-6 |
| peak σ (GeV⁻²) | ~5e-6 with Var > 0 |
| unitarity threshold √s (1%) | ~102.85 GeV (> M_Z) |
| deficit mid (s=1.01 s_pole) | 0.882 |

Deficit scan receipt (√s → deficit):

| √s (GeV) | deficit |
|---:|---:|
| 89.19 (M_Z − 2) | 0.285 |
| 91.19 (pole) | **1.000** |
| 91.64 (mid) | 0.882 |
| 96.19 (M_Z + 5) | 0.056 |
| 102.85 (1% thr) | 0.010 |

`main` declares `with NonUnitary` — peak path cannot hide the effect.

### EXP3 — EW tension (tree vs radiative Δρ)

| Quantity | Value |
|---|---|
| M_W pred (tree GUM) | 79.954 ± 0.0028 GeV |
| M_W pred (radiative, ρ = 1+Δρ) | **80.301 ± 0.0031 GeV** |
| M_W PDG direct | 80.377 ± 0.012 GeV |
| **pull tree** | **≈ −34.35** |
| **pull radiative** | **≈ −6.17** |
| S,T,U measured | 0.05±0.11, 0.09±0.14, −0.01±0.11 |
| Δρ (top, QCD-corrected) | 0.0087 ± 0.0010 |
| a_μ Schwinger | 0.001161 |

**On-shell construction:**  
`M_W = M_Z √[(1 − sin²θ_W)(1 + Δρ)]`  
(Δα is **not** folded into this form — it belongs in the G_F-based Δr relation.)

**Honest note:** leading Δρ lifts M_W by ~0.35 GeV and collapses pull from −34σ to
−6σ. Residual pull is **by construction** — full Δr / higher orders still open.
No BSM claim. The novelty is the **typed tension object that improves under
honest radiative construction**.

---

## Novelty (construction, not paper)

1. **Runnable metrology budget** for a textbook width — who owns the variance.  
2. **Compiler-enforced NonUnitary** on a Z-pole observable path + deficit curve.  
3. **Tree vs radiative pull** as one executable tension dashboard (Δρ step).

None of these require a journal. All of them are **objects that exist** when
the vertical is green.

---

## Next construction (if the vertical holds)

- Full G_F-based Δr M_W prediction (close residual ~6σ).  
- Extend gate to Madaros with science-boundary allowlist for this example.  
- Optional: export deficit-vs-√s as a machine-readable receipt (JSON).

## AI disclosure

Vertical assembled under human direction (2026-07-25). GAIDeT-ICMJE 2025.
