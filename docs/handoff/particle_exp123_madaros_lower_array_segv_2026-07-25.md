<!-- docs:meta
topic_id: repo.docs.handoff.particle-exp123-madaros-lower-array-segv-2026-07-25
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.particle-exp123-madaros-lower-array-segv-2026-07-25
-->

# Blocker — Madaros SEGV on full EXP123 native lower

**Blocker-ID:** `BLK-20260725-madaros-exp123-lower-array-segv`  
**Severity:** medium (default-engine run of *full* vertical)  
**Class:** compiler / imported multimodule native lower  
**Date:** 2026-07-25  
**Owner:** compiler lane (imported IR / `lower_array`)  
**Worktree:** separate from particle science (do not mix with #1461 physics)

---

## What works (N4 partial close)

| Surface | Status |
|---|---|
| Madaros **check** full EXP123 | green |
| Madaros **run** `exp123_madaros_core.sio` | **green** (11/11) |
| Madaros **run** module combos (chain_z + nonunitary + ew_precision) | green |
| lean_single full EXP123 | green (62/62) |

## What fails

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
SOUNIO_SOUC_ENGINE=madaros ./bin/souc run \
  examples/particle_physics/exp123_z_metrology_nonunitary_ew.sio
```

Dies after:

```text
imported_compile: typecheck ok
imported_compile: lower_begin
lower_array: seed_begin
Segmentation fault
```

Also: full `exp4_unstable_spectrum.sio` Madaros run SEGV (larger EXP2-like scan body).

## Root causes isolated (particle-side)

1. **Nested Epistemic method chains** on struct fields  
   (`nu.amp_sq.scale(x)`) → SEGV when the containing function is in the module.  
   **Mitigation landed:** `nonunitary.sio` rewritten to `ep_scale` / `ep_mul` free functions.

2. **vertex → spinor → complex import graph**  
   Loading `particle_physics::vertex` from `nonunitary` or full `epistemic_chain` pulls a graph that SEGVs or thin-link fails.  
   **Mitigation landed:**  
   - `nonunitary_amp.sio` for vertex-backed amplitudes  
   - `epistemic_chain_z.sio` Madaros-safe Z metrology (no vertex)

3. **Residual:** full EXP123 / EXP4 **main IR size / scan body** still SEGVs with only 6 modules loaded — not explained by (1)(2) alone. Minimal core of same modules runs.

## Minimal repro (still open)

```bash
# Fails (full vertical)
SOUNIO_SOUC_ENGINE=madaros ./bin/souc run \
  examples/particle_physics/exp123_z_metrology_nonunitary_ew.sio

# Passes (core)
SOUNIO_SOUC_ENGINE=madaros ./bin/souc run \
  examples/particle_physics/exp123_madaros_core.sio
```

## Acceptance gate (compiler fix)

- Madaros run of **full** `exp123_z_metrology_nonunitary_ew.sio` prints `PARTICLE_EXP123_OK`  
- Or documented compiler limit with IR function-count ceiling if intentional

## Non-goals

- Do not reintroduce vertex into nonunitary core without a Madaros lower fix.  
- Do not claim full EXP123 Madaros-run green until acceptance above.

## AI disclosure

Handoff under human direction (2026-07-25).
