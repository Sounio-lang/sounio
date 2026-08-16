<!-- docs:meta
topic_id: repo.docs.audit.lane-b3-status-20260813
authority: repo_only
audience: users
last_validated: 2026-08-14
validated_by: lane-b3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.lane-b3-status-20260813
-->

# B3 Lane status 2026-08-14 (session handoff)

## Goal
Get IrModule.functions off multi-MiB value-type stack so hello does not SEGV under 32 MiB.

## Confirmed root causes (measured)

### 1. Seed miscompiles global `[IrFunction; N]` element access
- Whole-value and field loads return pointer garbage (~1.4e14)

### 2. Seed miscompiles global `[Option<Box<T>>; N]`
- gdb: scale-8 index load then `mov (%rax),%rax` with `rax=0x40` → SEGV
- Option array treated as pointer table then double-dereferenced

### 3. Global scalar `Option<Box<IrFnTable>>` set is a silent no-op
- Prefill/install appeared to work; `ir_fn_set` did not persist names
- Measured: preseed filled all 8192 slots because every find_or_add was "new"

### 4. Nested place `(*(*m).fn_table).slots[i] = f` does not persist
- Fix: move Box to local, write through `(*table).slots[i].field = …`, put Box back
- Verified: `readback name_len=4 eq=1` after set

### 5. ItemList walk re-visits under seed-built madaros
- `current = (*list).tail` / `current = None` does not stop the loop reliably
- `break` inside match may not leave `while true`
- **Workaround that works:** `return` after first preseeded item (`item_n > 1`)
- Hello shows "Main file: 1 items" but walk re-enters 64+ times without return cap

### 6. Body lower SEGV after bodies_begin (open)
- After hard-capped preseed (fn_count=1), bodies_begin adds two more fns then SEGV
- gdb: `rep movsq` (IrFunction-sized copy) with ~700 stack frames of `0xffffffff`
- Stack overflow / infinite recursion in body path — not yet isolated

## Current design (in tree)

```
pub struct IrFnTable { slots: [IrFunction; 256] }  // temp shrink from 8192
pub struct IrModule {
    pub fn_table: Box<IrFnTable>,  // heap; IrModule stays small when moved
    pub fn_count: i64,
    ...
}
ir_fn_get(m: &IrModule, i)  -> clone slots[i] through Box
ir_fn_set(m: &! IrModule, i, f) -> Box extract, field write, put-back
```

- `IR_MAX_FUNCS = 256` and `BSS_MAX_GLOBALS = 255` temporarily (restore 8192 when green)
- Modular build: **green** (`/tmp/madaros-b3-g22` / g23)
- Hello compile: **not green** (SEGV after bodies_begin)

## Progress vs prior handoff

| Checkpoint | Prior | Now |
|---|---|---|
| Modular build | green | green |
| E035 residuals | open | closed for B3 path |
| summary_begin SEGV (Option array) | open | closed |
| ir_fn_set persists names | no | yes (readback eq=1) |
| preseed completes | hang/8192 slots | yes with hard cap |
| bodies_begin | not reached | reached |
| hello Written | no | no (SEGV) |
| stack floor ≤32 MiB | unmeasured | unmeasured |

## Next actions (ordered)

1. **Isolate body SEGV** after `bodies_begin` (700-frame overflow during IrFunction memcpy).
   - Trace `lower_program_bodies_from_summary_with_epistemic_boxed_owned` and
     `lowerer_lower_fn_item_mut` — find the recursive call.
2. **Fix ItemList walk** properly (not hard-cap): seed-safe advancement that does not
   re-enter the same node. Apply to all preseed/body item walkers.
3. Restore `IR_MAX_FUNCS = 8192` once write-through and walks are solid.
4. Strip diagnostic prints once hello compiles.
5. Run `scripts/ci/measure_madaros_stack_floor.sh` (or equivalent).
6. Un-draft #1729 when ≤32 MiB hello is green.

## Binary / branch

- Branch: `fix/lane-b3-ir-module-heap-20260813`
- Worktree: `/workspace/sounio-worktrees/lane-b-functions`
- Latest modular ELF: `/tmp/madaros-b3-g23/madaros` (build green)
- Do not wrap `build_modular_madaros.sh` in outer `souc-build-lock` (nested deadlock)
