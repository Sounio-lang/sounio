---
name: sounio-compiler-mir
description: "Implement and debug Sounio MIR/SSA analyses and optimization passes (CSE/DCE/const-prop/LICM); use when editing `compiler/src/mir/` or adding MIR-level performance work."
---

# Sounio Compiler MIR

## Overview

Make safe, SSA-aware performance improvements in MIR without breaking semantics.

## Workflow

### 1) Orient in the MIR stack

- Read `docs/MIR_OPTIMIZATION_STRATEGY.md` for the intended pass roadmap.
- Use `references/mir-navigation.md` to jump to the right module quickly.

### 2) When editing an existing pass (e.g., CSE)

- Confirm effect/purity assumptions:
  - Treat `Call`/`CallIndirect` as effectful unless proven pure.
  - Treat `Load` as unsafe for CSE without alias info (current CSE is conservative).
- Preserve SSA invariants (or mark `preserves_ssa()` false and re-establish SSA).

### 3) When adding a new MIR pass

- Implement the pass under `compiler/src/mir/optimization/`.
- Export it in `compiler/src/mir/optimization/mod.rs`.
- Register it in `compiler/src/mir/optimization/pass_manager.rs` if it’s in the default pipeline.
- Add focused unit tests close to the pass (see existing pass tests).

## References

- MIR map + entry points: `references/mir-navigation.md`
- New-pass checklist: `references/pass-checklist.md`
