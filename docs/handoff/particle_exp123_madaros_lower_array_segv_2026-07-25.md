<!-- docs:meta
topic_id: repo.docs.handoff.particle-exp123-madaros-lower-array-segv-2026-07-25
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.particle-exp123-madaros-lower-array-segv-2026-07-25
-->

# Blocker — Madaros full EXP123 native lower / peak ABI

**Blocker-ID:** `BLK-20260725-madaros-exp123-lower-array-segv`  
**Severity:** medium → **partially closed** (particle mitigation)  
**Class:** compiler / imported multimodule native lower  
**Date:** 2026-07-25  
**Owner:** compiler lane (imported IR / `lower_array` residual)  
**Worktree:** `codex/madaros-exp123-lower-array-segv-20260725`

---

## Status (2026-07-25 closeout lane)

| Surface | Status |
|---|---|
| Madaros **check** full EXP123 | green |
| Madaros **run** `exp123_madaros_core.sio` | **green** (11/11) |
| Madaros **run** full `exp123_z_metrology_nonunitary_ew.sio` | **green** (62/62) after particle mitigations |
| lean_single full EXP123 | green (62/62) |
| Gate `scripts/ci/particle_exp123_gate.sh` | lean full + Madaros check + core + **full** |

### Closed (particle-side mitigations)

1. **SEGV at `lower_array: seed_begin`** on full EXP123  
   - Split EXP2 into `exp2_basic_deficits` / `exp2_peak_approx` / `exp2_scan_receipt`  
   - Free-function Epistemic access (`ep_val` / `ep_variance` / `ep_confidence`) instead of nested `prop.amp_sq.val()`

2. **Peak zero under full Madaros IR** after SEGV closed  
   - Imported `eemm_z_peak_xsec_nu(...)` returns val=var=0 under full EXP123 IR even with correct args  
   - Same formula as `exp2_peak_xsec_local` in the **main module** returns σ≈5e-6 and positive GUM var  
   - Core vertical still uses the imported path successfully (ABI works at smaller IR)  
   - Mitigation: local peak body + still call stdlib entry for `NonUnitary` effect enforcement

### Residual for true compiler fix (not claiming closed)

- Imported multimodule call to `eemm_z_peak_xsec_nu` under large main IR returns zero `Epistemic` (not early-return logic; removing early return did not help).  
- Nested method chains on struct fields (`nu.amp_sq.scale`) and vertex→spinor graphs remain known Madaros hazards (already mitigated in stdlib).  
- `exp4_unstable_spectrum.sio` full Madaros run not re-validated in this lane.

## Acceptance (particle vertical)

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
SOUNIO_SOUC_ENGINE=madaros ./bin/souc run \
  examples/particle_physics/exp123_z_metrology_nonunitary_ew.sio
# → PARTICLE_EXP123_OK / PASS 62
bash scripts/ci/particle_exp123_gate.sh
# → PARTICLE_EXP123_GATE_OK including Madaros full
```

## Acceptance (compiler true fix — still open)

- Full EXP123 can use **only** imported `eemm_z_peak_xsec_nu` (drop `exp2_peak_xsec_local`) under Madaros and still print peak ≈ 5e-6 with var > 0.  
- Or document intentional IR/function-count ceilings with a reproducible diagnostic.

## Non-goals

- Do not reintroduce vertex into nonunitary core without a Madaros lower fix.  
- Do not remove the local peak until the imported path is proven under full IR.

## AI disclosure

Handoff under human direction (2026-07-25).

## Follow-up (2026-07-26 novelty lane)

- **EXP4** full Madaros run closed via same split/free-fn/local-peak pattern (`exp4_unstable_spectrum.sio`).
- **EXP6** universal ξ vertical is Madaros full green without peak ABI (no `eemm_z_peak_xsec_nu` call).
- Imported peak residual under full EXP123 IR remains open for a true compiler fix.

