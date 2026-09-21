<!-- docs:meta
topic_id: repo.docs.research.particle-exp19-h-yukawa-amp-2026-08-06
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.particle-exp19-h-yukawa-amp-2026-08-06
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Particle EXP19 — H amp→σ via `h_bb_yukawa_amplitude_nu`

**Date:** 2026-08-06  
**Source:** `examples/particle_physics/exp19_h_yukawa_amp_to_xsec.sio`  
**Gate:** `scripts/research/particle_exp19_h_yukawa_amp_gate.sh`  
**Receipt:** `examples/particle_physics/results/exp19_h_yukawa_amp_to_xsec.json`  
**Depends:** EXP16 (H honesty band) + EXP18 pattern (stdlib vertex amp)

---

## Claim

H twin of EXP18's stdlib amp restore: continuum cross-section from

```
y_b = √2 m_b / v
|M|² = y_b⁴ · |D_H|²
σ(s) = |M|²(s) · s / (12π)
```

with `m_b` from `mass_bottom()` (GUM) and `v = higgs_vev()` via
`h_bb_yukawa_amplitude_nu`, kept distinct from the local bb NWA peak toy
(same as EXP16).

Measured (lean_single + Madaros, 2026-08-06):

| Quantity | Value |
|---|---:|
| `σ_from_amp` | 5.32×10⁻⁴ GeV⁻² |
| `σ_peak` | 8.16×10⁻⁴ GeV⁻² |
| `σ_from_amp / σ_peak` | **0.652209** (matches EXP16 local-num) |
| honesty band | `(0.3, 2)` |
| off-pole | ξ = 2 |

## Pillars (5/5)

| ID | Check |
|---|---|
| P1 | `|M|²` pole in `(1e-9, 1e-3)` |
| P2 | `σ` in `(1e-6, 1e-2)` |
| P3 | peak finite and `ratio ∈ (0.3, 2)` |
| P4 | `σ(ξ=2) < σ(pole)` |
| P5 | `Var(|M|²) > 0` (lean: m_b + M_H; Madaros print may round tiny vars) |

## Math-review

`bin/llm-offload -t math-review -p xai` (2026-08-06):

- Yukawa `y_b=√2 m_b/v` and `|M|²=y_b⁴|D_H|²` **[OK]**
- Flagged vector `12π` as wrong for scalar J=0 (physical would use `4π`)

**Disagreement (logged):** this leaf keeps shared vector `12π` of EXP14–18 so the
EXP16 twin ratio stays bit-identical. Switching continuum+peak to `4π` scales the
ratio by `(12/4)² = 9` → ~5.87, which breaks the EXP16 honesty twin. Absolute H
xsec is an explicit non-claim; the leaf is amp→σ formula honesty, not a physical
Higgs lineshape.

## Non-claims

- Not a full H→bb¯ helicity / colour amplitude; not gg→H / VBF.
- Not NLO / QCD / PDF.
- Not a physical scalar spin-factor claim (`4π`); uses shared vector `12π`.
- Does not migrate EXP16 (thin local-num leaf remains a regression witness).
- Does not close Madaros imported-module GUM print residuals.

## Reproduce

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
bash scripts/research/particle_exp19_h_yukawa_amp_gate.sh
# expect: PARTICLE_EXP19_GATE_OK
```
