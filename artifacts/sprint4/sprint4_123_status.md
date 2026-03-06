# Sprint 4 1-2-3 Status Pack

- status: `pass`
- reason: `all_lanes_pass`

## Lanes
- sprint1_critical_bug_fixes: `pass` reason=`all_steps_passed`
- sprint1_frontend: `pass` reason=`target_met`
- sprint2_parser_stability: `pass` reason=`all_cases_passed`
- sprint2_lexer_features: `pass` reason=`all_cases_passed`
- sprint2_wasm_dispatch: `pass` reason=`all_cases_passed`
- sprint2_ir_lower: `pass` reason=`all_cases_passed`
- sprint3_wasm: `pass` reason=`all_cases_passed`
- sprint3_native: `pass` reason=`native_output_observed`
- sprint4_elf_execution: `pass` reason=`all_cases_passed` artifact=`artifacts/sprint4/elf_execution_gate.v1.json`

## Blockers
(none)

## Key Fix (Sprint 4)

Root cause: `emit`, `fresh_reg`, `fresh_label`, `lower_fn_params` used multi-level
nested array writes (`lo.module.functions[fn_id].instrs[i] = x`) which are silent
no-ops in the SOBC/JIT value-semantics VM.

Fix: apply copy-mutate-write-back pattern (as `find_or_add_fn_id` and `intern_string`
correctly do): `var f = m.functions[fn_id]; f.xxx = y; m.functions[fn_id] = f; lo.module = m`.

Result: selfhosted compiler now emits correct IR → native ELF exits with correct code.
`fn main() -> i64 { 0 }` produces ELF that exits 0 (was: 1 due to empty function body).
