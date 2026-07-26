<!-- docs:meta
topic_id: repo.docs.handoff.particle-madaros-exp-residuals-2026-07-26
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.particle-madaros-exp-residuals-2026-07-26
-->

# Madaros residuals 1-2-3 (continued)

**Date:** 2026-07-26

## 1 — LO/NWA Epistemic under Madaros (partially closed)

- New module `approx_effects_gum.sio` with free-fn `*_gum` APIs.
- Thin vertical `exp10_approx_physics.sio` Madaros **8/8**.
- Gate requires `PARTICLE_EXP10_MADAROS_PHYSICS_OK`.
- Original `approx_effects` untouched (lean full EXP10 30/30).
- **2026-07-26 closeout:** `ep_gate` / `ep_require_conf` / `ep_is_credible` now compare
  confidence via `ep_i64_ge(field, k)` call-arg boundary. Madaros multimodule mis-branches
  on direct `if e.confidence >= k` even when returning the same field is correct.
  Witness: `tests/multimodule/madaros_ep_gate_*.sio` + `scripts/ci/madaros_ep_gate_imported_gate.sh`.
  Full EXP123 under Madaros: **58/58** after this fix (gates 111/113 were the fail).

## 2 — Peak under full EXP123 IR (narrowed)

- Minimal imported `eemm_z_peak_xsec_nu` is OK under Madaros (not always zero).
- Full EXP123 still uses **local peak body** as defence-in-depth (imported peak forced only for NonUnitary effect).
- Core path exercises imported peak successfully.

## 3 — Drop workarounds (not yet)

Local peak and thin physics vertical remain. Compiler residual: i64 field-if mis-branch
in imported native (stdlib workaround only; true fix is native codegen).

## Main regression note

Merge of `research/particle-exp123-20260725` into main reintroduced vertex imports and broke Madaros full EXP123 SEGV path; this lane restores the Madaros-safe vertical.

## CI note (2026-07-26): arity-13 stack

`scripts/ci/madaros_imported_call_arity_13_gate.sh` default soft stack raised
131072 → 524288 KiB. FO GUM multi-channel growth made 128 MiB insufficient on
GitHub runners (SEGV / call-arg scratch overflow). Measured: 262144 passes;
131072 fails. Contracts LoRA sync for `variance_covariance_blindness.sio` (β10).
