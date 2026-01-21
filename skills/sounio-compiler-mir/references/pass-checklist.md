# MIR pass checklist (SSA + semantic safety)

## Before coding

- State the pass’s soundness assumptions:
  - Does it assume SSA? dominance?
  - Does it assume no side effects? no aliasing? no UB?
- Identify which MIR instructions are “pure” for this pass.

## While coding

- Preserve SSA (or explicitly rebuild it later).
- Be conservative around:
  - `Load`/`Store` unless you have alias info
  - `Call`/`CallIndirect` unless you have purity/effects facts
  - control-flow sensitive transforms (dominators, loops)

## After coding

- Add a minimal MIR-level unit test (builder-based if available).
- Run the pass manager SSA validator on modified IR in tests/debug builds.
- Consider interaction with existing passes in `pass_manager.rs` ordering.
