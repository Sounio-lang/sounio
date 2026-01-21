# MIR navigation

## Core types

- `compiler/src/mir/instructions.rs` (instruction set, blocks, functions, module)
- `compiler/src/mir/types.rs` (MIR types, ids)
- `compiler/src/mir/builder.rs` (construction helpers)

## Analysis

- `compiler/src/mir/analysis/` (SSA validation, dataflow, CFG helpers)

## Optimization

- `compiler/src/mir/optimization/mod.rs` (pass exports)
- `compiler/src/mir/optimization/pass_manager.rs` (pass trait + pipeline)
- Pass implementations:
  - `compiler/src/mir/optimization/common_subexpression_elimination.rs`
  - `compiler/src/mir/optimization/dead_code_elimination.rs`
  - `compiler/src/mir/optimization/constant_propagation.rs`
  - `compiler/src/mir/optimization/licm.rs`

## Lowering / bridges

- `compiler/src/mir/lower/` and `compiler/src/mir/lower.rs` (HLIR → MIR)
- `compiler/src/codegen/` (MIR → backend codegen paths)
