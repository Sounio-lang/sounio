<!-- docs:meta
topic_id: repo.docs.research.particle-exp9-engine-joint-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.particle-exp9-engine-joint-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Particle EXP9 — Joint (M,Γ) GUM + EngineDisagreement

**Date:** 2026-07-26  
**Source:** `examples/particle_physics/exp9_engine_joint_gum.sio`  
**Gate:** `scripts/ci/particle_exp9_engine_joint_gate.sh` (dual-engine)

---

## A — Joint GUM

Stdlib:

- `width_z()` / `width_w()` Epistemics in `sm_params`
- `nu_reduced_xi_joint_ep(s, M, Γ, ρ)`
- `nu_reduced_xi_joint_uncorr_ep`
- `nu_unitarity_threshold_joint_ep`

Measured (ξ=1, Z):

| Budget | Var(ξ) |
|---|---:|
| mass-only | ~3e-6 |
| joint uncorr | ~4e-6 |
| joint ρ=0.3 | ~5e-6 |
| **expand** joint/mass | **~1.29** |

Thr 1% joint Var ≫ mass-only Var (Γ owns much of threshold uncertainty).

## B — EngineDisagreement

Gate runs **lean_single** and **Madaros**, compares metrics.

| Quantity | lean | Madaros | agrees? |
|---|---:|---:|---|
| deficit pole | 1.0 | 1.0 | yes |
| peak local | ~5e-6 | ~5e-6 | yes |
| **peak stdlib imported** | ~5e-6 | **0** | **no (witness)** |
| joint ξ | 1.0 | 1.0 | yes |

JSON: `particle.exp9.engine_disagreement.v1`  
Non-isomorphism: `EngineDisagreement_is_not_physics_scheme_tension`

This turns the Madaros imported-peak residual into an **executable platform receipt**, not a comment.

## Reproduce

```bash
bash scripts/ci/particle_exp9_engine_joint_gate.sh
```

## AI disclosure

Human direction 2026-07-26. GAIDeT-ICMJE 2025.
