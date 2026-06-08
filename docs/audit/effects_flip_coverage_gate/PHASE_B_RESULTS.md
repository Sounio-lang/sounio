# EFF.1 Phase B — per-module coverage measurement (RESULTS)

**Date.** 2026-06-08. **Baseline.** `526b0091f` (parse_module_item fix), warn-mode (toggle = 1).
**Method.** `--check <module>` over the 107 live modules (each runs seed-imports-then-check-*target*
= `import_typecheck_main(path)`). Workflow `wf_72ca4baf-5cf`; raw per-module data in `phase_b_raw.json`.

> **Counts below are the authoritative structured totals** (16 / 66 / 25). The workflow's synthesis
> agent reported 19 / 69 / 19 — a recount error; ignore it. The raw per-module array and the script
> summary both give 16 / 66 / 25.

## Headline

| Status | Modules | W035 |
|---|---:|---:|
| **OK** (clean target check) | **16** | **20** |
| **TYPEFAIL** | 66 | 95 (partial, obscured by type errors — not a clean count) |
| **PARSEFAIL** | 25 | 0 |
| Total | 107 | — |

- **The only trustworthy effect-annotation gap among cleanly-checkable modules: 20 W035, all in
  `self-hosted/native/regalloc.sio`.** The other 15 OK modules are clean (0 W035).
- OK modules (leaf-ish, few imported-type deps): `interop/artifact`, `interop/contract`,
  `ir/algebra`, `lexer/numparse`, `native/aarch64`, `native/contract`, `native/gc`, `native/regs`,
  **`native/regalloc` (20 W035)**, `native/runtime_context`, `native/stack_maps`, `resolve/scope`,
  `wasm/buf`, `wasm/core`, `wasm/opcodes`, `wasm/types`.

## Dominant failure modes (the two blockers)

**(1) 66 TYPEFAIL — incomplete imported-TYPE seeding.** First-error code distribution:
`E015 unknown struct type` ×16, `E001 binding type mismatch` ×13, `E016 field-initialiser wrong
type` ×8, `E011` ×6, `E004` ×6, `E005` ×5, `E035` ×3, `E007`/`E013` ×1, none ×7. The `E015`/`E016`/
many-`E001` cluster is the signature of a module referencing an imported **struct/enum** that the
seeder never registered: `checker_boot4_seed_imported` seeds imported **function** signatures but
**not imported type definitions**. So a module checked as a target can't resolve `AstPath`, `Name`,
IR struct types, etc. → it TYPEFAILs before its true effect gap is visible. (e.g. `check/defs.sio`
311 items → E015; `check/knowledge_context.sio` 158 items → E015 + W035 = 2.)

**(2) 25 PARSEFAIL — the 271-wall.** `parse_failed` on `check/check.sio`, `check/mod.sio`, all of
`ir/{ir,lower,layout,loop_opt,normalize,opt_cleanup,profile,tailcall,inline,dce}.sio`,
`lexer/{cursor,span,tables}.sio`, `compiler/{module_frontend,module_native_driver}.sio`,
`gpu/hlir_to_gpu.sio`, `hlir/ir.sio`, … — gen-N can't parse these (keyword-as-identifier
collisions, `module`/`effect`/`is`/`study`; see `project_parser_selfhost_gap_2026-06-08`).

## Consequence for the dispatch — Phase A is NOT a thin wrapper

EFF.1 §2 (Fb) assumed generalising `import_typecheck_main` → `_target` was a thin rename. Phase B
**refutes** that: a sound per-module coverage gate is blocked behind **two** upstream changes, both
larger than the dispatch's contained scope:

1. **Extend `checker_boot4_seed_imported` to seed imported TYPE definitions** (structs/enums), so a
   module checked as target resolves its imported types and TYPEFAILs only on genuine errors. This
   is a soundness-sensitive checker change in its own right (clears the 66 TYPEFAILs) — it needs its
   own evidence/verification, not a rushed edit.
2. **The 271-wall fix** (keyword demotion, concurrent session) — clears the 25 PARSEFAILs and is the
   prerequisite for `main.sio`'s own coverage + `gen2 == gen3` flip verification.

Until both land, the coverage gate can only cover the 16 leaf modules. Phase C (annotate) and
Phase D (flip + gate) are therefore **out of reach this session** — they are not "do the next step"
work; they are gated on (1) and (2).

## What this does NOT measure

- `main.sio` itself (not in the 107; gated by the 271-wall).
- TYPEFAIL modules' true effect gaps (masked by the type-seeding failure — the 95 TYPEFAIL W035 are
  not a clean count).
- Warn W035 is per-use-site, not per-fn.

## Net

Phase B succeeded as a *measurement* and did its job: it converted EFF.1 from "thin wrapper + gate"
into a correctly-scoped two-dependency problem (imported-type seeding + 271-wall), and it produced
the one actionable clean datum — `native/regalloc.sio`'s 20-use-site IO/effect gap. The remaining
phases are halted at that boundary pending authorisation for the imported-type-seeding change and
the upstream 271-wall fix.
