---
name: Sprint 225 — Uncertainty-Aware E-Graph FP Rewriting (Phase 1 done)
description: First analytically GUM-guided float rewriting in any compiler; Phase 1 e-graph infrastructure complete, Phase 2 opt_cleanup integration pending
type: project
---

## Status

**Phase 1 (DONE 2026-03-15):** E-graph infrastructure and GUM uncertainty propagation foundation.
Commit `f92c1adf`: 241 insertions across `self-hosted/ir/egraph.sio`. New opcodes, struct extensions, 10 tests (T61–T70).

**Phase 2 (PENDING):** Integrate epistemic pass into `opt_cleanup.sio` with `ocp_egraph_epistemic_pass()` function.

**Phase 3 (PENDING):** Full round-trip IR patching; new test file `tests/stdlib/epistemic/test_eg_epistemic_rewrite.sio` with T009–T018; gate script validation.

## Scientific Claim

**World-first:** Analytical GUM-guided floating-point associativity rewriting. Herbie (Panchekha et al., PLDI 2015) uses random sampling + search. Sounio proves: the e-graph holds all equivalent groupings; the extractor selects the one with minimum accumulated uncertainty per GUM §5.1.2 (delta method: `u(z)² = u(x)² + u(y)²`). **No existing compiler does this analytically.**

Why it matters: Sounio's `Epistemic<f64>` type carries measurement uncertainty. For a measured value ± σ propagating through matrix operations, the compiler can now choose `(a+b)+c` vs `a+(b+c)` based on which grouping minimizes rounding error given the known uncertainty. This is the bridge between metrological rigor (GUM) and compiler optimization (e-graph equality saturation).

## What Phase 1 Added

**New float opcodes:**
```
EG_OP_FADD = 8, EG_OP_FSUB = 9, EG_OP_FMUL = 10, EG_OP_FDIV = 11,
EG_OP_FNEG = 12, EG_OP_MEASURE = 13, EG_OP_COMBINE = 14
```

**EgSmallContext struct extended:**
- `unc_lo[64]: [i64; 64]` — GUM standard uncertainty × 1e9 (integer-quantized)
- `unc_hi[64]: [i64; 64]` — reserved for interval arithmetic (future)

**New helper functions:**
- `eg_isqrt(x: i64) -> i64` — Integer sqrt via 7 Newton iterations
- `eg_quantize_unc(u_f64: f64) -> i64` — Convert `f64` uncertainty to `i64` scale

**New e-graph functions:**
- `eg_small_add_var_with_unc(ctx, var_id, unc) -> i64` — Add VAR node with uncertainty
- `eg_small_add_binop_with_unc(ctx, op, lhs, rhs) -> i64` — Add binop with GUM propagated uncertainty
- `eg_small_extract_epistemic(ctx, class_id) -> EgNode` — Extract node with minimum uncertainty
- `eg_small_saturate_float(ctx) -> i64` — Float associativity saturation (8 iterations max)

**Tests (10 new, T61–T70):**
```
T61: eg_quantize_unc(2.5) == 2500000000
T62: eg_isqrt(25000000000) == 158113 (within 1)
T63: eg_isqrt(0) == 0
T64: eg_isqrt(1) == 1
T65: add_var_with_unc sets ctx.unc_lo correctly
T66: extract_epistemic picks min-uncertainty node
T67: saturate_float completes without error
T68: quantize_unc(0.0) == 0
T69: quantize_unc(-1.5) == 0 (negative clamp)
T70: eg_isqrt(4000000000) == 63245 (sqrt(4e10))
```

## What Phase 2 Needs

**File:** `self-hosted/ir/opt_cleanup.sio`

**Helper functions to add (before all callers):**
```sio
fn ocp_is_measure(op: IrOpcode) -> bool { ... }
fn ocp_binop_to_float_eg_op(op: IrOpcode) -> i64 { ... }  // OpAdd→EG_OP_FADD
```

**Main pass function:**
```sio
fn ocp_egraph_epistemic_pass(func: IrFunction) -> IrFunction with Mut, Div, Panic {
    if func.compile_strategy != IR_STRATEGY_PRECISION_PRESERVING { return func }
    if func.instr_count > 16 { return func }

    // 1. Scan for IrMeasure → record register uncertainties
    // 2. Scan for IrBinOp on epistemic registers → build EgSmallContext
    // 3. eg_small_saturate_float (8 iters)
    // 4. eg_small_extract_epistemic
    // 5. Patch instruction if structure differs
    func
}
```

**Integration:** In `opt_cleanup_function`, insert after `ocp_egraph_mini_pass`:
```sio
current = ocp_egraph_epistemic_pass(current)
```

**References:**
- `IR_STRATEGY_PRECISION_PRESERVING` = 2 (line 385 of `ir.sio`)
- `IrMeasure` opcode (line 322 of `ir.sio`)
- `IrInstr.imm_f64` field (line 362 of `ir.sio`)

## What Phase 3 Needs

**New test file:** `tests/stdlib/epistemic/test_eg_epistemic_rewrite.sio` (10 tests: T009–T018)
- T009: Pass does not fire for `IR_STRATEGY_STANDARD`
- T010: Pass fires for `IR_STRATEGY_PRECISION_PRESERVING` without crashing
- T011–T018: Full round-trip IR patches

**Gate script:** `scripts/sprint225_epistemic_egraph_gate.sh`
1. `$SOUC check self-hosted/ir/egraph.sio` → PASS
2. `$SOUC check self-hosted/ir/opt_cleanup.sio` → PASS
3. `$SOUC run self-hosted/ir/egraph.sio` → `70 / 70 passed`
4. `$SOUC run tests/stdlib/epistemic/test_eg_epistemic_rewrite.sio` → `10 / 10 passed`

## Blocked By / Blocking

**Blocked by:** Nothing. E-graph Phase 1 is self-contained.

**Blocks:**
- Epistemic WMMA tensor core backend (full PTX `wmma.mma.sync` emission)
- GUM arithmetic operator dispatch in native codegen
- Dimensional-epistemic type unification

## SOTA References

- **Herbie (Panchekha et al., PLDI 2015):** Empirical floating-point rewriting via random sampling + search
- **egg (Willsey et al., 2021):** Fast equality saturation framework
- **JCGM 100:2008 (GUM):** Guide for Uncertainty Measurement, delta-method §5.1.2
- **Nelson & Oppen (1980):** Congruence closure decision procedures
- **Tate et al. (PLDI 2009):** Equality saturation optimization framework

## Commit History

- `f92c1adf` [egraph] Sprint 225 Phase 1: uncertainty-aware e-graph infrastructure
