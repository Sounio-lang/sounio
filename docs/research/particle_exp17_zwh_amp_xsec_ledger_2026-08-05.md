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

# Particle EXP17 — Z / W / H amp→σ joint ledger (stdlib)

**Date:** 2026-08-05 (stdlib migrate 2026-08-06)  
**Source:** `examples/particle_physics/exp17_zwh_amp_xsec_ledger.sio`  
**Gate:** `scripts/research/particle_exp17_zwh_ledger_gate.sh`  
**Receipt:** `examples/particle_physics/results/exp17_zwh_amp_xsec_ledger.json`  
**Depends:** EXP14/18/19 stdlib amplitudes + EXP15/16 honesty bands

---

## Claim

One ledger jointly asserts the three continuum amp→σ honesty leaves via
`nonunitary_amp` (schema **v2**). If any species regresses, the vertical fails.

| Species | source | peak toy | honesty band | measured ratio |
|---|---|---|---|---:|
| Z | `eemm_z_amplitude_nu` | local NWA | (10, 20) | 13.952395 |
| W | `cc_w_leptonic_amplitude_nu` | Br·Γ_W leptonic | (2, 6) | **3.486637** |
| H | `h_bb_yukawa_amplitude_nu` | Br·Γ_H bb | (0.3, 2) | 0.652209 |

W ratio drifts vs EXP15 local-num (3.486629) by GUM on `coupling_g` — same as EXP18.

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
| Madaros **run** | green (shipped ELF post-#1627 promote) |
| Gate | `PARTICLE_EXP17_GATE_OK` |

## Math-review (v2 migrate)

`bin/llm-offload -t math-review -p xai` (2026-08-06):

- Item 3 (ordering = construction) **[OK]**
- Items 1–2 flagged optical-theorem / 4π concerns — **disagreement logged**:
  this migrate only redirects W/H continuum through the same `nonunitary_amp`
  leaves already reviewed in EXP18/19; EXP16/19 both use shared vector `12π`
  (scalar `4π` twin break already documented on EXP19). Absolute unitarity /
  optical theorem is an explicit NonUnitary construction non-claim.

## Non-claims

- Ratios are construction gaps, not PDG fits or collider observables.
- Not a full matrix-element / NLO / PDF ledger.
- Thin EXP14–16 leaves remain local-num regression witnesses.
- H keeps shared vector `12π` prefactor (EXP16/19 twin); scalar `4π` not claimed.
- Does not claim optical-theorem / unitarity closure (NonUnitary effect).

## Reproduce

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
bash scripts/research/particle_exp17_zwh_ledger_gate.sh
# expect: PARTICLE_EXP17_GATE_OK
```
