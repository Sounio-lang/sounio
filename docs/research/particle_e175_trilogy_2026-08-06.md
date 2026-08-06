<!-- docs:meta
topic_id: repo.docs.research.particle-e175-trilogy-2026-08-06
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.particle-e175-trilogy-2026-08-06
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Particle E175 trilogy — extern sweep + #1627 + EXP14 restore (2026-08-06)

## Scope

| Step | Deliverable |
|---|---|
| **1** | Drop remaining private `extern "C"` libm stubs across `stdlib/particle_physics/*.sio` (same class as #1661 lorentz/vertex). |
| **2** | Madaros #1627: do not resolve foreign private `extern "C"` stubs for native builtins (`prefer_module` + visibility allow-list). |
| **3** | EXP14 imports `nonunitary_amp::eemm_z_amplitude_nu` again (dual-engine green). |

## Evidence (shipped Madaros + stdlib)

```bash
bash scripts/research/particle_e175_amp_import_gate.sh
# → PARTICLE_E175_TRILOGY_GATE_OK
```

Witnesses: `complex`+`lorentz`, `complex`+`propagator`, EXP13, EXP14.

## #1627 checker note

Source changes in `self-hosted/check/{defs,check}.sio`. A negative control with a
deliberate private `extern "C" { fn sqrt; }` still E175 under the **shipped**
binary until Madaros is rebuilt and the ELF is promoted. Stdlib sweep alone
closes the particle import path without waiting for that rebuild.

## Non-claims

- Does not rewrite every stdlib crate off `extern "C"` — particle_physics only.
- Does not claim EXP17 uses `eemm_z_amplitude_nu` (still thin local NC scalars).
