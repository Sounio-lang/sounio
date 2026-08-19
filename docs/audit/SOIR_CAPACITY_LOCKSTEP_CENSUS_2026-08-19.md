<!-- docs:meta
topic_id: repo.docs.audit.soir-capacity-lockstep-census-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli5 (assumed #1947; measured on worktree /workspace/.wt/ir-capacity-1947)
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.soir-capacity-lockstep-census-2026-08-19
-->

# Semantic Declaration — SOIR capacity lockstep census

**Filed:** 2026-08-19 · **Lane:** grok-cli5 on existing PR #1947 · **Status:** census recorded, NOT unified

## Semantic declaration (required 6/6)

- **Concept-IDs:** proposed SOUNIO-IR-CAPACITY-COUPLING
- **Intent-Preserved:** a capacity ceiling that silently drops is a lie; truncated liveness is a wrong analysis, not a weaker one; main's refuse-function choice (`042c29be53`) is not converted into a panic on every compile
- **Transformation:** IrCapacities object makes the three runtime couplings checkable; invariant 2 at runtime is `dce_liveness_slots > 0`; SOIR deserializer function/string tables are gate-bound to `IR_MAX_FUNCS` / `IR_MAX_STRINGS`; oversized deserialize increments `SOIR_DESER_REFUSAL_COUNT`
- **Claims-Introduced:** the production default validates; a violating `region_slots` is refused; an oversized SOIR `fn_count` is refused detectably; the handwritten `[IrFunction; 16384]` in `serialize.sio` cannot drift from `IR_MAX_FUNCS` without the coherence gate failing
- **Claims-Forbidden:** capacity is NOT unified across the tree; a literal that happens to equal 16384 today is still a defect if it is not gate-bound; DCE does not analyse functions above 8192 instructions; raising `dce_liveness_slots` to 16384 is not claimed
- **Authoritative-Only-If:** coherence gate prints `serialize_functions=<IR_MAX_FUNCS> soir_deser_refusal=detectable coherent=pass`; one valid `souc compile` of `docs/audit/repro/ir_cap_hello.sio` exits 0; the invalid probe run exits non-zero with no PASS line

## What was fixed (this change)

| Site | Before | After |
|---|---|---|
| `ir.sio` header invariant 2 | advertised `dce_liveness_slots >= max_instructions` while default is 8192 < 16384 | header matches the chosen runtime invariant (`> 0`) and cites refuse-function |
| `serialize.sio` `deserialize_ir_module` functions | `[IrFunction; 1024]` then a handwritten `[IrFunction; 16384]` with silent `ir_empty_module()` | literal still 16384 (Sounio cannot index arrays by a `let`); **gate-bound** to `IR_MAX_FUNCS`; `SOIR_DESER_REFUSAL_COUNT` incremented before empty return |
| `serialize.sio` string_table | `[Name; 4096]`, silent refuse | gate-bound to `IR_MAX_STRINGS`; same detectable counter |
| `serialize.sio` `deserialize_ir_function` instrs | refuse returned empty function with no counter | increments `SOIR_DESER_REFUSAL_COUNT` (count and alloc-fail paths) |
| `dce.sio` refusal | empty stats, silent | `DCE_REFUSAL_COUNT` (already on the branch) |
| `test_ir.sio` T30 | defined twice (same class as #1695) | defined once; T32 feeds `fn_count = IR_MAX_FUNCS+1` and asserts the counter moved |

## Census — per-function / per-instruction literals NOT yet linked

Method: search `self-hosted/**/*.sio` for sites that dimension something PER FUNCTION or PER INSTRUCTION, and compare against the constant that should govern them. A literal that happens to equal the constant today is still a defect if it is hand-written, because it will not rise next time. Measured 2026-08-19 on this worktree.

**Gate-bound this turn (still a handwritten literal; the gate is the bind):**
- `serialize.sio` `var functions: [IrFunction; 16384]` must equal `IR_MAX_FUNCS`
- `serialize.sio` `var string_table: [Name; 4096]` must equal `IR_MAX_STRINGS`
- `ir.sio` `IrModule.functions: [IrFunction; 16384]` must equal `IR_MAX_FUNCS`

**Production, coincidentally-16384, NOT gate-bound — 16 sites:**

| Count | Site | Literal | Governing constant it should track |
|---:|---|---|---|
| 1 | `ir/normalize.sio:257` | `[IrFunction; 16384]` | `IR_MAX_FUNCS` |
| 2 | `ir/lower.sio:689,694` | `elem_kinds: [i64; 16384]` / `[0; 16384]` | per-function lowering width |
| 11 | `native/{reloc,elf,elf_bulk,frame,codegen_x86_linux}.sio` | `fn_offsets: [i64; 16384]` | `IR_MAX_FUNCS` / `backend_offset_slots` |
| 1 | `native/elf.sio:113` | `name_offsets: [i64; 16384]` | `IR_MAX_FUNCS` |
| 1 | `hlir/ir.sio:1139` | `[HlirInstr; 16384]` | HLIR's own instr cap (not `IR_MAX_INSTRS` unless they are the same concept) |

**Production, smaller than the governing constant, same defect class:**
- 1 × `compiler/module_loader.sio:3587` `built_functions: [IrFunction; 256]` written at `gi < final_module.fn_count` with no bounds check (not in the Madaros build closure)
- 4 × `compiler/module_frontend.sio` `fn_remap: &[i64; 2048]` — 2048 is not `IR_MAX_FUNCS`; unnamed merge-context cap

**Per-specialization, coincidentally-16384, NOT linked — 9 sites** in `check/specializer.sio` (`marks: [i64; 16384]` / `SPEC_DCE_G_MARKS`). Own size concept (`SPEC_DCE_SLOTS`) but the arrays are handwritten.

**Separate backends, own capacity tiers — do not silently treat as `IR_MAX_FUNCS`:**
- 6 × `native/wide.sio` + `wide_driver.sio` `fn_offsets: [i64; 1024]`
- 5 × `native/codegen.sio` + `render_native_compile_driver_f64.sio` `fn_offsets: [i64; 256]`
- 1 × `compiler/render_native_compile_driver_lean.sio` `fn_offsets: [i64; 512]`
- 2 × `bootstrap/bootstrap_v0.sio` `fn_offsets: [i64; 512]` (frozen bootstrap; do not raise)

**Test fixtures — small by construction, still handwritten:**
- 53 × `[IrFunction; 1024]` (`compiler/main.sio` 4, `ir/inline.sio` 30, `ir/tailcall.sio` 19)
- 1 × `[IrFunction; 64]` (`ir/closure.sio`)
- 2 × `fn_offsets: [i64; 64]` (`native/test_phase1.sio`)

**How many were found**

| Class | Sites |
|---|---:|
| Gate-bound this turn (serialize functions/strings + IrModule.functions) | 3 |
| Production coincidentally-16384, still unlinked | 16 |
| Production smaller / unnamed (module_loader 256 + frontend 2048) | 5 |
| Specializer marks 16384 | 9 |
| Separate-backend / bootstrap fn_offsets | 14 |
| Test-fixture `[IrFunction; N]` / tiny fn_offsets | 56 |
| **Total sites reported** | **103** |

Each unlinked production site is a blocker waiting to be found the way `[IrFunction; 1024]` in serialize sat for five weeks. This PR does not raise them. It names them.

## Related

- #1897 — raised `IR_MAX_FUNCS` in `ir.sio` / `lower.sio` / `codegen_x86_linux.sio`, missed `serialize.sio`
- #1649 — moved `IrFunction.instrs` to an arena region
- `042c29be53` — `dce_run_impl` refuses functions above `DCE_MAX_INSTRS`
- Watch: when `[IrFunction; 1024]` leaves the SOIR deserializer on main, #870 #881 #883 #885 reopen
