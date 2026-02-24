# Sounio Original Plan (Canonical)

This document captures the original execution plan from the start of this thread and maps it to the current implementation specs.

## Scope Rules (Locked)

- Stay hardware-first (`.sio`, K-AXI, FPGA path).
- Preserve strict gate evidence flow.
- No scope drift from the original sequence.

## Original Plan (Execution Order)

1. Omega L2 hardware fusion step 1:
   - Real PTX `.knowledge.*` intrinsics for `Knowledge<f64>`.
   - Runtime propagation for value/variance/provenance/confidence.
   - HLIR lowering emits epistemic operations.
2. K-AXI transition:
   - Move from passive transport to active epistemic fabric.
   - Bus-side propagation and counter updates per operation.
3. Hardware Epistemic Power accumulator:
   - Accumulate hardware counters and expose live score signal.
4. Pure Sounio enforcement:
   - Canonical source in Sounio hardware declarations/emitter path.
   - Keep generated outputs aligned with strict gate.
5. Locked sequence A -> B -> C:
   - A: Interpreter struct-array alias fix + regression.
   - B: Native `hardware_publish` + emitter.
   - C: Real PTX launch path with K-AXI + accumulator feedback.
6. Evidence discipline:
   - Strict gate validation and auditable artifacts after each stage.

## Mapping to Current Open Offload Specs

The following files in `.claude/offload-specs/` are the active implementation map for the remaining self-hosted compiler/runtime expansion:

1. `data_structures.md` -> Base self-hosted collections/runtime data model
   - `self-hosted/collections/ordered_map.sio`
   - `self-hosted/collections/arena.sio`
   - `self-hosted/intern.sio`
   - `self-hosted/collections/graph.sio`

2. `gpu_ir_expansion.md` -> Expanded GPU IR surface
   - `self-hosted/gpu/kernel_ir.sio`
   - New opcodes, memory-space enum, and IR metadata fields

3. `hlir_lowering.md` -> Frontend-to-backend bridge
   - `self-hosted/hlir/lower.sio`
   - AST/HIR -> HLIR SSA lowering and control-flow construction

4. `metal_msl_codegen.md` -> Portable backend emission
   - `self-hosted/gpu/metal.sio`
   - MSL text emitter parity with PTX-style buffer patterns

5. `ptx_regalloc_expansion.md` -> PTX backend completion path
   - `self-hosted/gpu/ptx_advanced.sio`
   - Register allocation, opcode coverage, architecture capability gates

## Current Recommended Build Order (No Drift)

1. `data_structures.md`
2. `gpu_ir_expansion.md`
3. `hlir_lowering.md`
4. `metal_msl_codegen.md`
5. `ptx_regalloc_expansion.md`

This order preserves the original plan dependency chain: foundational data model -> IR surface -> lowering -> backend emitters.
