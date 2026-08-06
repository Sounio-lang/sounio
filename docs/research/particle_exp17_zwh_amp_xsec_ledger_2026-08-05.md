<!-- docs:meta
topic_id: repo.docs.research.particle-exp17-zwh-amp-xsec-ledger-2026-08-05
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.particle-exp17-zwh-amp-xsec-ledger-2026-08-05
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Particle EXP17 — Z / W / H amp→σ joint ledger

**Date:** 2026-08-05  
**Source:** `examples/particle_physics/exp17_zwh_amp_xsec_ledger.sio`  
**Gate:** `scripts/research/particle_exp17_zwh_ledger_gate.sh`  
**Receipt:** `examples/particle_physics/results/exp17_zwh_amp_xsec_ledger.json`  
**Depends:** EXP14 (Z) + EXP15 (W) + EXP16 (H)

---

## Claim

One ledger jointly asserts the three continuum amp→σ honesty leaves. If any
species regresses, the vertical fails.

| Species | num | peak toy | honesty band | measured ratio |
|---|---|---|---|---:|
| Z | NC `g⁴ c² / cos⁴θ_W` | local NWA (= EXP14 peak formula) | (10, 20) | 13.952363 |
| W | CC `(g²/2)²` | Br·Γ_W leptonic | (2, 6) | 3.486629 |
| H | Yukawa `y_b⁴` | Br·Γ_H bb | (0.3, 2) | 0.652209 |

Additional joint checks:

- **ordering:** `ratio_Z > ratio_W > ratio_H` (construction-gap hierarchy, not a fit)
- **shape:** each species has `σ_off < σ_pole` (H at ξ=2; Z/W at s→1.01 s)

## Pillars (5/5, bits = 31)

| ID | Check |
|---|---|
| P1 | Z band + Var > 0 |
| P2 | W band + Var > 0 |
| P3 | H band + Var > 0 |
| P4 | `ratio_Z > ratio_W > ratio_H` |
| P5 | off-pole drop for Z, W, and H |

## Engine note

| Surface | Status |
|---|---|
| lean_single **run** | green (`PARTICLE_EXP17_OK`, 5/5, bits=31) |
| Madaros **run** | green (three masses + three propagators; local NWA peaks) |
| Gate | `PARTICLE_EXP17_GATE_OK` |

## Non-claims

- Ratios are construction gaps, not PDG fits or collider observables.
- Not a full matrix-element / NLO / PDF ledger.
- EXP14 now imports Madaros-safe `eemm_z_amplitude_nu` (E175 closed); EXP17
  still uses local NC scalars for the joint thin graph (IR size), not a regression.

## Reproduce

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
bash scripts/research/particle_exp17_zwh_ledger_gate.sh
# expect: PARTICLE_EXP17_GATE_OK
```

## LLM-offload

`math-review` via xai (receipt `examples/particle_physics/results/exp17_math_review_offload.txt`).
Canonical log append deferred (shared-file claim).

## AI disclosure

Human direction 2026-08-05. GAIDeT-ICMJE 2025.
