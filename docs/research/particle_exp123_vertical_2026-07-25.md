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
**Status:** `PARTICLE_EXP123_OK` (58/58 checks under lean_single)  
**Source:** `examples/particle_physics/exp123_z_metrology_nonunitary_ew.sio`  
**Gate:** `scripts/ci/particle_exp123_gate.sh`  
**JSON receipt:** `examples/particle_physics/results/exp123_deficit_curve.json`  
**Stdlib:** tree / on-shell-Δρ / G_F-Sirlin Δr M_W APIs in `ew_precision.sio`

---

## Purpose

Construction-first: three real experiments on `stdlib/particle_physics` with
GUM / effects discipline, where novelty can grow without paper-first pressure.

| Exp | Physics | Novelty surface |
|---|---|---|
| **1** | Γ(Z→ee) metrology + uncertainty budget + confidence gate | GUM provenance to observable; budget of PDG sources |
| **2** | Non-unitarity at Z pole: deficit(s), peak σ with `NonUnitary`, deficit vs √s **JSON** | Effect-typed unstable intermediate; machine-readable curve |
| **3** | EW tension ladder: tree → Δρ → G_F-Δr pulls, S/T/U | Tension as first-class object that improves under honest construction |

---

## Reproduce

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
export SOUNIO_SOUC_ENGINE=lean_single   # full run path

./bin/souc run examples/particle_physics/exp123_z_metrology_nonunitary_ew.sio
# expect: PARTICLE_EXP123_OK  (58 PASS)

bash scripts/ci/particle_exp123_gate.sh
# expect: PARTICLE_EXP123_GATE_OK
# also writes examples/particle_physics/results/exp123_deficit_curve.json
# and checks Madaros science-boundary: no E-SRB-002 for this example
```

**Engine note:**

| Surface | Status |
|---|---|
| lean_single **run** | **green** (58/58, gate path) |
| Madaros **check** | **green** (no E008/E137 on this vertical) |
| Madaros science-boundary | **no E-SRB-002** (`research` → `scientific-package-candidate` allowlist) |
| Madaros **run**/native lower | **SEGV** in `lower_array` on imported IR — compiler residual, not claimed |

Madaros typecheck fixes for this vertical: (1) `stdlib/complex/lib.sio` splits
the sixth `extern "C"` (`atan2`) into a second block — Madaros drops symbols past
the fifth in one block; (2) local helpers renamed `chk`→`pass_if`, `near`→`within`
(name collisions under multi-module Madaros typecheck).

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

### EXP2 — NonUnitary + deficit vs √s + JSON

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

Machine-readable: `EXP2_DEFICIT_JSON {...}` on stdout; gate writes
`examples/particle_physics/results/exp123_deficit_curve.json`
(schema `particle.exp123.deficit_curve.v1`).

`main` declares `with NonUnitary` — peak path cannot hide the effect.

### EXP3 — EW tension ladder

| Construction | M_W (GeV) | pull vs PDG 80.377±0.012 |
|---|---:|---:|
| tree (GUM) | 79.954 ± 0.0028 | **−34.35** |
| on-shell ρ = 1+Δρ | 80.301 ± 0.0031 | **−6.17** |
| **G_F Sirlin Δr** | **80.362 ± 0.0037** | **−1.18** |
| PDG direct | 80.377 ± 0.012 | — |

| Other | Value |
|---|---|
| S,T,U measured | 0.05±0.11, 0.09±0.14, −0.01±0.11 |
| Δρ (top, QCD-corrected) | 0.0087 ± 0.0010 |
| a_μ Schwinger | 0.001161 |
| tension flag | **CONSISTENT** (|pull_GF| < 2) |

**G_F construction (Sirlin):**

```
A0 = π α / (√2 G_F)
Δα = Δα_lep + Δα_had^(5) + Δα_top
Δr = Δα − (c²/s²) Δρ + Δr_rem
M_W² = (M_Z²/2) [1 + √(1 − 4 A0 / (M_Z² (1−Δr)))]
```

with self-consistent on-shell `s²_W = 1 − M_W²/M_Z²`.

`Δr_rem = 0.0075 + (α/4π) ln(m_H/100)` is the O(10⁻²) pure-weak remainder
scale when Δα already includes hadronic VP — **not** fitted to M_W^PDG.

**Honest note:** pull ladder −34 → −6 → **−1.2** under successive honest steps.
Residual ≲2σ is construction-consistent; full two-loop / higher orders remain
open. No BSM claim.

---

## Novelty (construction, not paper)

1. **Runnable metrology budget** for a textbook width — who owns the variance.  
2. **Compiler-enforced NonUnitary** + deficit curve JSON receipt.  
3. **Pull ladder** tree → Δρ → G_F-Δr as one executable tension dashboard.  
4. **Science-boundary allowlist** for research → scientific-package-candidate.

None of these require a journal. All of them are **objects that exist** when
the vertical is green.

---

## Next construction (if the vertical holds)

- Higher-order Δr / two-loop residue (optional tightening below 1σ).  
- Madaros typecheck of `particle_physics` (E008 residuals) — separate from boundary.  
- Promote ring status when package extraction lands.

## AI disclosure

Vertical assembled under human direction (2026-07-25). GAIDeT-ICMJE 2025.
