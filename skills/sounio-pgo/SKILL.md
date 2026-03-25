---
name: sounio-pgo
description: "Work on the profile-guided optimization pipeline: counter injection, .sprof file output, strategy promotion, inlining, layout, const-fold/DCE, and register allocation; use when editing any sprint 38–52+ IR optimization files."
---

# Sounio PGO Pipeline

## Overview

The full PGO pipeline: **instrument → profile → promote → inline → layout → const_fold → DCE → regalloc → disp8 → compact frame → peephole → codegen**.

All passes live in `self-hosted/ir/` and `self-hosted/native/`, wired in `self-hosted/compiler/main.sio`'s `compile()` function. The native backend is complete as of Sprint 65 (30/30 gate).

Strategies: `STANDARD=0`, `PRECISION=1`, `AGGRESSIVE=2`, `INSTRUMENTED=3`.

## Workflow

### 1) Understand which pass to touch

| Pass | File | Runs when |
|------|------|-----------|
| Strategy codegen | `ir/lower.sio` | Always — `ir_compute_strategy_from_ast()` |
| Counter injection | `ir/opt_strategy.sio` | strategy=INSTRUMENTED — `ir_opt_apply_strategy()` |
| .sprof output | `native/codegen.sio` | At exit via `__prof_dump` function |
| Profile reader | `ir/profile.sio` | `--use-profile` flag — `sprof_parse()` |
| Strategy promotion | `ir/profile.sio` | After profile load — `sprof_apply_promotion()` |
| Inlining | `ir/inline.sio` | `opts.optimize=true` — `inl_run_pass()` |
| Layout | `ir/layout.sio` | `opts.optimize + valid profile` — `layout_sort_by_profile()` |
| Const-fold + DCE | `ir/opt_cleanup.sio` | `opts.optimize=true` — `opt_cleanup_module()` |
| Register alloc (4-preg r12-r15) | `native/regalloc.sio` | Always — `nc_run_regalloc()` (Sprint 52/62) |
| Disp8 adaptive encoding | `native/lower_ir.sio` | Always — `ir_slot_fits_disp8()` (Sprint 63) |
| Compact frame sizing | `native/lower_ir.sio` | After regalloc — `nc_min_frame_size()` (Sprint 64) |
| Peephole optimizer | `native/peephole.sio` | After regalloc — `ph_run_on_func()` (Sprint 65) |

### 2) Add a new pass

1. Create `self-hosted/ir/your_pass.sio` with `module ir::your_pass` declaration
2. Define main entry: `fn your_pass_run(module: &!IrModule) with Mut, Panic, Div`
3. Add `use ir::your_pass::*` in `self-hosted/compiler/main.sio`
4. Call `your_pass_run(&!ir_module)` after the appropriate stage in `compile()`
5. Add self-tests T(N+1) and T(N+2) in `main.sio` self-test block

**Module import rule**: use wildcard `use ir::your_pass::*` — selective imports break `==` comparisons on imported types (known Sounio issue; see Sprint 51 notes).

### 3) Run the relevant gate

```bash
SOUC=./bin/souc

bash scripts/sprint65_peephole_gate.sh     # 30/30 — peephole optimizer wiring
bash scripts/sprint64_frame_trim_gate.sh   # 25/25 — compact frame sizing
bash scripts/sprint63_disp8_gate.sh        # 30/30 — adaptive disp8/disp32 encoding
bash scripts/sprint62_multipreg_gate.sh    # 35/35 — 4-preg linear-scan regalloc
bash scripts/sprint52_regalloc_gate.sh     # 33/33 — regalloc wiring baseline
bash scripts/sprint51_opt_cleanup_gate.sh  # 28/28 — const_fold + DCE
bash scripts/sprint50_layout_pgo_gate.sh   # 26/26 — function layout
bash scripts/sprint49_inline_pgo_gate.sh   # 26/26 — strategy-aware inlining
bash scripts/sprint47_sprof_gate.sh        # 32/32 — .sprof output + reader
```

New passes must add their own gate script following the sprint52 pattern (see `scripts/sprint52_regalloc_gate.sh`).

### 4) PGO pipeline integration

**Generating a profile:**
```bash
# Compile with instrumented strategy
$SOUC run examples/instrumented_program.sio
# → writes profile.sprof to current directory

# Apply profile
$SOUC run self-hosted/compiler/main.sio -- --use-profile profile.sprof --compile examples/hot_program.sio
```

**Promotion thresholds** (in `sprof_apply_promotion()`):
- count > 1000 → `AGGRESSIVE` strategy
- count > 100 → `PRECISION` strategy
- count ≤ 100 → no promotion (strategy unchanged)

**`sprof_promotion_target(count) -> i64`** returns the target strategy constant or -1 (no promotion). Used by `--show-profile` diagnostic.

**Strategy-aware inlining** (inline.sio):
- Callee strategy `AGGRESSIVE` → +80 benefit bonus
- Callee strategy `PRECISION` → +40 benefit bonus
- Inline threshold: default 100 (benefit - cost > 0)

**Layout pass** (layout.sio):
- Functions scored by `sprof_lookup()` count
- Sorted descending (hot first, unknown/cold last)
- `layout_patch_fn_ids()` fixes all `IrCall.fn_id` references after reorder
- Logs: `[layout] N functions: H hot, C cold`

## References

- Pipeline wiring: `self-hosted/compiler/main.sio` — `compile()` function, `opts.optimize` flag
- Strategy computation: `self-hosted/ir/lower.sio` — `ir_compute_strategy_from_ast()`
- Counter injection: `self-hosted/ir/opt_strategy.sio` — `ir_opt_apply_strategy()`
- Profile: `self-hosted/ir/profile.sio` — `SprofProfile`, `sprof_parse()`, `sprof_lookup()`, `sprof_apply_promotion()`, `sprof_promotion_target()`
- Inlining: `self-hosted/ir/inline.sio` — `InlFuncInfo`, `inl_compute_benefit()`, `inl_run_pass()`
- Layout: `self-hosted/ir/layout.sio` — `LayoutScore`, `layout_sort_by_profile()`
- Const-fold + DCE: `self-hosted/ir/opt_cleanup.sio` — `ocp_const_fold()`, `ocp_dce()`, `opt_cleanup_module()`
- Regalloc: `self-hosted/native/regalloc.sio` — `ra_run_pass()`, Poletto-Sarkar linear scan
- Encoder helpers: `self-hosted/native/encode.sio` — `emit_mov_rdi_r12`, `push_r12`, `pop_r12`
- .sprof output: `self-hosted/native/codegen.sio` — `emit_prof_dump_function()`
- Gates: `scripts/sprint47_sprof_gate.sh` through `scripts/sprint52_regalloc_gate.sh`
- Sprint history: MEMORY.md §Project Status (sprints 38–52)
