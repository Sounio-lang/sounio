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

## 2 — Peak under full EXP123 IR (still open)

- Imported Epistemic **and** f64 returns from `nonunitary` can zero under large main IR.
- Main-module **fully inlined** peak works; EXP123 restored Madaros-safe (chain_z, free-fn, local peak).
- Expected checks **58** (lean-compatible EXP3 without EpistemicTension API).

## 3 — Drop workarounds (not yet)

Local peak and thin physics vertical remain. Compiler: imported return ABI under large multimodule IR.

## Main regression note

Merge of `research/particle-exp123-20260725` into main reintroduced vertex imports and broke Madaros full EXP123 SEGV path; this lane restores the Madaros-safe vertical.
