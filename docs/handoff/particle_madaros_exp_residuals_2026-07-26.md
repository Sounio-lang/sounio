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

## 2 — Peak under full EXP123 IR (**closed**)

- Full EXP123 now uses **only** imported `eemm_z_peak_xsec_nu` (local peak body dropped 2026-07-26).
- Madaros full vertical: **58/58** after drop; lean gate green.
- EXP12 residual-closure ledger jointly asserts peak + gate + collapse + product.

## 3 — Drop workarounds (partial)

| Workaround | Status |
|---|---|
| Local peak body | **dropped** |
| `ep_gate` field-if | stdlib `ep_i64_ge` (still needs native fix) |
| `pb_is_credible` / `ck_is_credible` | same call-arg pattern |
| Thin EXP10 physics vertical | remains (IR size) |
| Vertex/amplitude on full EXP123 | **closed** EXP13 dual-engine (2026-08-05) — drop private `extern "C" sqrt/sinh/cosh` in `lorentz`/`vertex` (false E175 vs `complex` builtins); see `docs/research/particle_e175_amp_import_2026-08-05.md` |


Compiler residual: i64 field-if mis-branch in imported native codegen.

## Main regression note

Merge of `research/particle-exp123-20260725` into main reintroduced vertex imports and broke Madaros full EXP123 SEGV path; this lane restores the Madaros-safe vertical.

## CI note (2026-07-26): arity-13 stack

`scripts/ci/madaros_imported_call_arity_13_gate.sh` default soft stack raised
131072 → 524288 KiB. FO GUM multi-channel growth made 128 MiB insufficient on
GitHub runners (SEGV / call-arg scratch overflow). Measured: 262144 passes;
131072 fails. Contracts LoRA sync for `variance_covariance_blindness.sio` (β10).

## 4 — Native field-if residual (forensic, 2026-07-26)

Witness: `tests/multimodule/madaros_field_if_i64_{leaf,main}.sio`  
Gate: `scripts/ci/madaros_field_if_i64_gate.sh`  
Audit: `docs/audit/MADAROS_FIELD_IF_I64_2026-07-26.md`

Madaros imported multimodule:

- `return e.confidence` / `+ 0` → **846 OK**
- `if e.confidence >= m` / `let c; if c >= m` → **0 wrong**
- `e.confidence - m` → **pointer-scale garbage**
- `ge(e.confidence, m)` call-arg → **1 OK**

Native fix blocked this session by active claims on `self-hosted/native/**` and
`self-hosted/ir/lower.sio`. Stdlib workarounds remain until `MADAROS_FIELD_IF_I64_FIXED`.

## Field-if closeout

**CLOSED** #1511 + workaround drop (direct `e.confidence >= k` restored).

## 5 — E175 trilogy (2026-08-06)

- **stdlib:** remaining private `extern "C"` sqrt/exp/log/… stubs removed from
  `particle_physics/*.sio` (detector/fitting use `math::libm` for pow/fabs).
- **checker (#1627):** `prefer_module` skips foreign private extern stubs;
  visibility treats native-builtin externs as always visible (needs Madaros rebuild
  to take effect in the shipped ELF).
- **EXP14:** restores `eemm_z_amplitude_nu` import; dual-engine green.
- Gate: `scripts/research/particle_e175_amp_import_gate.sh` → `PARTICLE_E175_TRILOGY_GATE_OK`.
- Note: `docs/research/particle_e175_trilogy_2026-08-06.md`.

## 6 — #1627 verify + EXP17 amp (2026-08-06)

- Tip Madaros rebuild verifies #1627 fixture green.
- EXP17 Z continuum uses `eemm_z_amplitude_nu` (shipped Madaros dual-engine OK).

## 6b — #1627 promote (2026-08-06)

- Root cause of promote E035: `sm_params` thin `Epistemic::measured` accessors
  lacked `with Mut, Div, Panic`. Fixed in `sm_params.sio`.
- Promoted tip ELF → `bin/madaros-linux-x86_64` (sha256 `f9ddba96…5189e`).
- Gates green: #1627 shipped, E175 trilogy, EXP17/18/19.
- Issue #1627 closable.

## 7 — EXP18 W vertex amp (2026-08-06)

- `nonunitary_amp::cc_w_leptonic_amplitude_nu` — `(g⁴/4)·|D_W|²` via `coupling_g`.
- `examples/particle_physics/exp18_w_vertex_amp_to_xsec.sio` dual-engine green
  (ratio ≈ 3.4866, band (2, 6)).
- Gate: `scripts/research/particle_exp18_w_vertex_amp_gate.sh`.

## 8 — EXP19 H Yukawa amp (2026-08-06)

- `nonunitary_amp::h_bb_yukawa_amplitude_nu` — `y_b⁴·|D_H|²`, `y_b=√2 m_b/v`
  via `mass_bottom` + `higgs_vev`.
- Ratio **0.652209**, band (0.3, 2); matches EXP16. Shared vector `12π`
  kept (math-review noted scalar `4π`; twin ratio preserved — see EXP19 doc).
- Gate: `scripts/research/particle_exp19_h_yukawa_amp_gate.sh` (Madaros retry×3).
- Completes stdlib amp restore trio Z/W/H (EXP14/18/19); thin EXP14–16 stay.

## 9 — EXP17 stdlib ZWH ledger (2026-08-06)

- EXP17 W/H now use `cc_w_leptonic_amplitude_nu` + `h_bb_yukawa_amplitude_nu`
  (schema v2). Ratios Z/W/H: 13.952395 / 3.486637 / 0.652209.
- Thin EXP14–16 remain witnesses. Gate: `particle_exp17_zwh_ledger_gate.sh`.

