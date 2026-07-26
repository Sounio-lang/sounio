# Handoff — Madaros i64 field-if residual

**For:** native/codegen owner  
**Blocker class:** Madaros imported multimodule ABI / branch condition  
**Evidence:** `docs/audit/MADAROS_FIELD_IF_I64_2026-07-26.md`  
**Witness gate:** `scripts/ci/madaros_field_if_i64_gate.sh`  

## Ask

Implement native fix so `if e.confidence >= m` returns 1 for conf=846, m=800
in `tests/multimodule/madaros_field_if_i64_main.sio`, then gate prints
`MADAROS_FIELD_IF_I64_FIXED`.

## Already shipped workarounds (do not regress)

- `stdlib/epistemic/knowledge.sio` → `ep_i64_ge`
- `stdlib/epistemic/knightian.sio` → `pb_i64_ge`
- `stdlib/epistemic/composed_effects.sio` → `ck_i64_ge`

## Do not

- Delete workarounds until FIXED marker is green on CI with current-source Madaros.
- Conflate with peak ABI (peak residual closed #1492).
