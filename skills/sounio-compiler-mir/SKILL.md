---
name: sounio-compiler-mir
description: "Implement and debug Sounio SSA analyses and optimization passes (CSE/DCE/const-prop/LICM); use when editing passes under `self-hosted/ir/`."
---

# Sounio Compiler MIR

## Overview

Make safe, SSA-aware performance improvements without breaking semantics. IR passes live in `self-hosted/ir/`.

## Workflow

### 1) When editing an existing pass (e.g., CSE)

- Confirm effect/purity assumptions:
  - Treat `Call`/`CallIndirect` as effectful unless proven pure.
  - Treat `Load` as unsafe for CSE without alias info (current CSE is conservative).
- Preserve SSA invariants (or mark `preserves_ssa()` false and re-establish SSA).

### 2) When adding a new pass

- Add the pass under `self-hosted/ir/`.
- Add a focused test under `tests/`.

## References

- New-pass checklist: `references/pass-checklist.md`
