<!-- docs:meta
topic_id: repo.docs.audit.madaros-multimodule-native-seed-segfault-2026-06-22
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-multimodule-native-seed-segfault-2026-06-22
-->

# Madaros Multimodule Native Seed Segfault - 2026-06-22

## Scope

This note pins the current imported/native compiler blocker that is preventing
the solver proof-checker modules from running through their imported-library
gates. The failure is broader than `theorem::smt`, `theorem::pb`,
`theorem::lrat`, `theorem::qflra_exact`, or `math::rational`: the official
minimal multimodule witness also fails before any solver code is executed.

## Branch And Compiler

- Repo: `/workspace/sounio`
- Branch: `feat/escape-analysis-raw-refs`
- Commit: `0e3a8ebe5`
- Wrapper: `/workspace/sounio/bin/souc`
- Raw Madaros: `/workspace/sounio/bin/madaros-linux-x86_64`
- Compiler identity: `Madares v0.80.0 -- the Sounio self-hosted compiler`

## Minimal Reproducer

Official witness:

```bash
OUT=/tmp/sounio-mm-witness-20260622T092443Z
SOUNIO_MADAROS_MM_WITNESS_DIR="$OUT" \
  bash scripts/ci/madaros_multimodule_witness.sh
```

Observed:

```text
[madaros-mm-witness] FAIL: thin_single expected_exit=7 actual_exit=139
Main file: 2 items
Merged IR: 1 functions
Imported frontend summary: ok
module_native_driver: imported source uses modular IR path
imported_compile: begin
imported_compile: loaded 2 modules
imported_compile: typecheck ok
imported_compile: lower_begin
lower_array: seed_begin
Segmentation fault
```

The witness program is `tests/multimodule/thin_single_main.sio`, which imports
`thin_single_lib::{add_public}` and should return `7`.

## Additional Probe

Temporary probes under `/tmp/smt-mm-repro-qlC0MK` showed that even an unused
stdlib import hits the same crash shape:

```sounio
use theorem::smt::*
fn main() -> i32 { return 0 }
```

Result:

```text
check=0
compile=139
imported_compile: loaded 2 modules
imported_compile: typecheck ok
imported_compile: lower_begin
lower_array: seed_begin
Segmentation fault
```

The same pattern reproduced for:

- `use theorem::pb::*`
- `use theorem::lrat::*`
- `use theorem::cardinality::*`
- `use theorem::qflra_exact::*`
- `use math::rational::*`

## Current Boundary

The frontend and typechecker are not the blocker:

- `./bin/souc check stdlib/theorem/smt.sio` passes.
- `./bin/souc check tests/run-pass/smt_assumption_core_imported.sio` passes.
- `./bin/souc check tests/stdlib/theorem/test_smt_solver_basic.sio` passes.
- `./bin/souc check tests/multimodule/thin_single_main.sio` passes.

The failure happens during imported/native lowering in
`module_frontend_compile_imported_to_file`:

1. modules load;
2. multimodule typecheck passes;
3. `module_frontend_lower_programs_array_boxed` starts;
4. the process segfaults after `lower_array: seed_begin`, before `seed_done`.

That points at the seed lowering path:

```text
module_frontend_lower_programs_array_boxed
  -> seed_prog_copy = programs[0]
  -> module_frontend_lower_program_box_traced_with_externs(seed_prog_copy, ...)
```

## Impact On Solver Work

The solver/proof-checker imported gates are currently valid frontend contracts
but cannot yet be runtime evidence:

- `smt_assumption_core_imported` is a known failure at native runtime.
- `pb_row_chain_imported` is a known failure at native runtime.
- `lrat_chain_imported` is a known failure at native runtime.
- `test_qflra_exact` / rational imported tests remain blocked downstream.

Self-contained solver gates remain the executable evidence until this compiler
blocker is repaired.

## Next Action

Debug the seed lowering path with the smallest official witness:

```bash
./bin/souc check tests/multimodule/thin_single_main.sio
./bin/souc run tests/multimodule/thin_single_main.sio
```

Acceptance gate for the compiler repair:

```bash
bash scripts/ci/madaros_multimodule_witness.sh
bash scripts/run_sio_test_suite.sh smt_assumption_core --verbose
bash scripts/run_sio_test_suite.sh test_smt_solver_basic --verbose
```

Only after the official multimodule witness passes should the solver imported
known-failures be promoted to runtime gates.
