<!-- docs:meta
topic_id: repo.docs.audit.soir-capacity-lockstep-census-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: empryo-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.soir-capacity-lockstep-census-2026-08-19
-->

# Semantic Declaration — SOIR capacity lockstep census

**Filed:** 2026-08-19 · **Lane:** empryo-1 · **Status:** census recorded, NOT unified

## Semantic declaration (required before self-hosted changes)

A capacity ceiling that silently drops is a lie, and a lie is exactly what this
language exists to make impossible. The SOIR deserializer
(`self-hosted/ir/serialize.sio`) dimensioned its function table by a
hand-written literal (`[IrFunction; 1024]`) instead of the governing constant
(`IR_MAX_FUNCS`). When #1897 raised `IR_MAX_FUNCS` 8192 → 16384 in `ir.sio`,
`lower.sio`, and `codegen_x86_linux.sio`, it did not touch `serialize.sio` —
so the deserializer's literal fell out of lockstep. Five founder PRs
(#869 #870 #881 #883 #885, open since 2026-07-14) died on the resulting E002
and sat blocked for five weeks.

This change links the SOIR deserializer to `IR_MAX_FUNCS` and adds the missing
bounds check with a detectable refusal. It does NOT and CANNOT claim that
capacity is unified across the tree — the census below lists the literals that
remain unlinked. The census says how many are left; claiming unification while
any remain is itself the lie this work forbids.

## What was fixed (this change)

| Site | Before | After |
|---|---|---|
| `serialize.sio:4465` `deserialize_ir_module` functions | `[IrFunction; 1024]`, loop `while i < fn_count` with NO bounds check (out-of-bounds write on >1024 fns) | `[IrFunction; 16384]` matching `IR_MAX_FUNCS`; `fn_count` refused if `< 0 || > IR_MAX_FUNCS` |
| `serialize.sio` string_table | `[Name; 4096]`, no bounds check | literal matches `IR_MAX_STRINGS`; `string_count` refused if out of range |
| `serialize.sio:646` `deserialize_ir_function` instrs | stale inline `instrs: [IrInstr; 4096]` (removed by #1649), missing `region`/`float_reg_bits` (E046) | arena region via `ir_instr_arena_alloc(instr_count)`, per-slot `ir_arena_store`; `instr_count` refused if `> IR_MAX_INSTRS`; alloc failure refused |
| `dce.sio` refusal | `dce_run_impl` returned empty stats SILENTLY when refusing a function above `DCE_MAX_INSTRS` | `pub var DCE_REFUSAL_COUNT` incremented on every refusal — the refusal is now detectable |

## Census — per-function / per-instruction literals NOT yet linked

Method: grep for sites that dimension something PER FUNCTION or PER
INSTRUCTION, and compare against the constant that should govern them. A
literal that happens to equal the constant today is still a defect if it is
hand-written, because it will not rise next time. Classified:

**FIXED here** — `serialize.sio` functions/string/instrs (above).

**Production, unbounded, same defect class — needs a follow-up:**
- `module_loader.sio:3587` `built_functions: [IrFunction; 256]`, written at
  `built_functions[gi]` for `gi < final_module.fn_count` with NO bounds check
  (write at :3702). Not in the Madaros build closure (frontend/thin-link path).

**Separate backends, own capacity tiers — verify against their own governing
constant, not IR_MAX_FUNCS:**
- `native/wide.sio` / `native/wide_driver.sio` `fn_offsets: [i64; 1024]`
- `native/codegen.sio` `fn_offsets: [i64; 256]` (`finalize_elf64_shared`)
- `compiler/render_native_compile_driver_lean.sio` `fn_offsets: [i64; 512]`
- `compiler/render_native_compile_driver_f64.sio` `fn_offsets: [i64; 256]`

**Frozen bootstrap — deliberately pinned, do not raise:**
- `bootstrap/bootstrap_v0.sio` `fn_offsets: [i64; 512]`, `IR_MAX_INSTRS = 512`

**Test fixtures — small by construction, low risk, still hand-written:**
- `compiler/main.sio:6642` (+3) `funcs: [IrFunction; 1024]` (tco tests)
- `ir/inline.sio:1348` (+4) `funcs: [IrFunction; 1024]` (inl tests)
- `ir/tailcall.sio:1505/1709` `funcs: [IrFunction; 1024]` (tco tests)
- `ir/closure.sio:458` `lifted_functions: [IrFunction; 64]`
- `native/test_phase1.sio` `fn_offsets: [i64; 64]`

## Claims-Forbidden

Capacity is NOT unified. This change fixes the SOIR deserializer lockstep site
and makes the DCE refusal detectable. The census above lists the remaining
unlinked literals by class. Do not assert unification until each production
site is linked to its governing constant with a detectable refusal.

## Related

- #1897 — raised IR_MAX_FUNCS in ir.sio/lower.sio/codegen_x86_linux.sio, missed serialize.sio
- #1649 — moved IrFunction.instrs to an arena region (deserialize_ir_function was not updated)
- `042c29be53` — dce_run_impl refuses functions above DCE_MAX_INSTRS (dce.sio:818-821)
- `MADAROS_IR_CAPACITY_OBJECT_DESIGN_DISPATCH_2026-08-18.md` — the IrCapacities object
