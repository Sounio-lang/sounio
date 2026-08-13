<!-- docs:meta
topic_id: repo.docs.audit.lane-b3-status-20260813
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.lane-b3-status-20260813
-->

# B3 Lane status 2026-08-13 (session handoff)

## Goal
Get IrModule.functions off multi-MiB value-type stack so hello does not SEGV under 32 MiB.

## Confirmed root causes

1. **Seed miscompiles global `[IrFunction; N]` element access**
   - `return IR_FN_TABLE[i]` returns pointer garbage (instr_count ≈ 1.4e14)
   - Field-by-field loads from the same global array are also garbage
   - Measured with b3_probe after lower_done before resolve walk

2. **Dropped BSS helpers** during early B3 rewrites (restored):
   - BSS_MAX_GLOBALS, BSS_BASE_LINUX, IR_STRATEGY_BSS_GLOBAL, BSS_INIT_STRING_MAGIC
   - bss_name_hash, ir_load_global/store_global, ir_store_ptr, ir_call_extern, ir_load_fn_ref, ir_call_indirect

3. **Broken bulk rewrite pattern** (fixed in lower.sio):
   - `(*ir_fn_get((*lo).module).fn_table_slot, i)` → `ir_fn_get((*(*lo).module).fn_table_slot, i)`

4. **Seed arity false positive** on one-line if (fixed):
   - `if changed { ir_fn_set(a, b, c) }` → multiline block

5. **ItemClaim exhaustiveness** (fixed in imports.sio + check.sio)

## Direction that typechecked once (b3n binary ~99MB)
- Storage: `Option<Box<IrFunction>>` pools (not global IrFunction arrays)
- ir_fn_get: clone from heap box
- ir_fn_set: overwrite in place without Alloc (requires prefill)
- ir_empty_module: prefill N boxes with Alloc
- Prefill of 8192 SEGVs; 512 is the working-set compromise for small modules

## Remaining seed typecheck errors (3)
- "effect not declared at line 6645/6672/6708" in lower.sio emit path
- Not cured by adding Alloc to emit/fresh_*/array literal helpers
- Likely residual from empty_module Alloc cascade or seed import effect analysis
- Blocker for green modular build until resolved

## Runtime status
- Stack floor not improved yet (compile path not green end-to-end)
- check works under madaros; compile hangs/SEGVs depending on table design

## Next actions
1. Isolate the 3 seed effect errors (compare against last green b3n tree)
2. Lazy Box alloc in ir_fn_set with Alloc + complete effect cascade (or freelist without Alloc)
3. Measure stack floor once compile succeeds for hello
4. Un-draft #1729 when ≤32 MiB hello is green
