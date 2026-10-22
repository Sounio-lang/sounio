# Performance Investigation Plan

## Goal
Understand and fix 3× slowdown with `&Lowerer` and 4× memory with `ir_fn_get_ref`.

## Baseline Confirmed
- Commit: 8c4717178
- Time: 162s
- Memory: 1.58 GB
- Stack: Works with 8 MB for main.sio, needs override for 8191-global test

## Issue 1: &Lowerer is 3× slower (491s vs 162s)

### What Changed
Functions changed from:
```sio
fn field_idx_from_name(self, name: Name) -> i64  // copies ~4.5MB
```
To:
```sio
fn field_idx_from_name(self: &Lowerer, name: Name) -> i64  // reference
```

### Hypothesis
The slowdown is NOT from cache locality—same RSS means no extra allocations.

More likely:
1. **Struct was never actually copied**: Compiler may have been doing RVO (return value optimization)
2. **Reference prevents optimization**: Taking `&Lowerer` forces materialization
3. **Calling convention**: By-value allows register passing, reference forces memory

### Test Plan
1. Add timing around `field_idx_from_name_simple` calls
2. Compare assembly of by-value vs by-reference version
3. Check if Lowerer is actually moved vs copied in baseline

## Issue 2: ir_fn_get_ref uses 4× memory (6.46 GB vs 1.58 GB)

### What Changed
Changed from:
```sio
let func = ir_fn_get(&module, i)  // copies ~1.9 KB
```
To:
```sio
let func = ir_fn_get_ref(&module, i)  // reference
(*func).field  // explicit deref
```

### Hypothesis
The memory explosion is suspicious—references should REDUCE memory usage.

Likely causes:
1. **Cache pollution**: `.cache/madaros/` grew from 0 to 375 MB during test
2. **Reference lifetime**: Keeping &IrFunction alive prevents chunk deallocation
3. **Hidden copy in modified code**: The fix may have introduced accidental copies elsewhere

### Test Plan
1. Clear cache before each test
2. Track `.cache` size during compilation
3. Check if chunks are freed in baseline but not in fix

## Next Actions

1. ✅ Remove doc files from repo root
2. ⏳ Profile &Lowerer slowdown (current)
3. ⏳ Fix or explain &Lowerer regression
4. ⏳ Profile ir_fn_get_ref memory
5. ⏳ Fix or explain memory regression
6. ⏳ Address stack overflow properly (convert hot recursion to iterative)

## Stack Overflow - Proper Fix

The 618 functions taking `Lowerer` by value is the root cause.

**Strategy**:
1. Identify the top 5-10 recursive hotspots (likely expr lowering, pattern matching)
2. Convert to iterative with explicit stack
3. Target: <1 MB stack usage for 10k globals

**NOT**: Just pass &Lowerer everywhere (proven slow)
**INSTEAD**: Keep by-value where possible, make deep recursion iterative
