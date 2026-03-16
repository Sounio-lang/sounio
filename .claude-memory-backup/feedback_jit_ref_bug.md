---
name: JIT mutation bugs — complete diagnosis with array workaround
description: Cranelift JIT stores through dereferenced &! pointers fail for scalar/struct but WORK for array element writes via (*arr)[idx]. Complete taxonomy and workaround.
type: feedback
---

## Cranelift JIT Mutation Bugs — Complete Taxonomy (2026-03-13)

### What's BROKEN

1. **`*p = val` on `&!i64`** — write lost, callee can't even see its own write
2. **`(*s).field = val` on `&!Struct`** — write lost
3. **`a.b.c = val` nested struct field (2+ levels)** — write lost even on locals (no &!)
4. **`(*boxed).field = val` through Box deref** — write lost
5. **`emit_byte_mut(&! b, val)`** — all ref-mutation encode functions produce zero bytes

### What WORKS

1. **`(*arr)[idx] = val` on `&![T; N]`** — array index forces real pointer arithmetic ✅
2. **`s.field = val` single-level struct field on local** — direct stack slot access ✅
3. **By-value return `fn f(x: S) -> S`** — result materialized through rax ✅
4. **Reading through `&` (shared ref)** — works correctly ✅
5. **Direct top-level field writes: `nc.symbols = val`** — works ✅

### Root Cause

The JIT generates stores to the parameter's own stack slot (which holds the pointer value) instead of through the pointer to the target memory. Array element access `(*arr)[idx]` works because the codegen emits actual pointer arithmetic (`base + idx*8`) which forces a real memory dereference path.

### KEY DISCOVERY: Array Workaround

Use `[i64; N]` arrays instead of struct fields for data mutated through `&!`:
```sio
// BROKEN:
fn find_or_add(module: &!IrModule) { (*module).fn_count = (*module).fn_count + 1 }

// WORKS:
fn find_or_add(state: &![i64; 4]) -> i64 {
    let count = (*state)[0 as usize]
    (*state)[0 as usize] = count + 1
    count
}
// Verified: id0=0, id1=1, id2=2, final_count=3
```

### Safe Patterns (ranked by preference)

1. **By-value return**: `c = fn(c, args)` — always safe
2. **Array &! mutation**: `(*arr)[idx] = val` — works for shared mutable state
3. **Copy-out/modify/put-back**: `var tmp = *boxed; tmp.field = val; boxed = Box::new(tmp)`
4. **Single-level field write**: `s.field = val` on local variable only

### Verified Results (tested 2026-03-13)

| Pattern | Result |
|---------|--------|
| `*p = 99` on `&!i64` | ❌ callee sees 0, caller sees 0 |
| `(*v).n = 99` on `&!Struct` | ❌ callee sees 0, caller sees 0 |
| `(*arr)[i] = 99` on `&![i64;4]` | ✅ both see 99 |
| `find_or_add` via array-ref | ✅ correct incrementing |
| `o.inner.count += 1` (2-level local) | ❌ stays 0 |
| `o.b += 1` (1-level local) | ✅ works |
| By-value return | ✅ always works |

**How to apply:** For the IR pipeline: use the FLAT-OWNED summary path (implemented 2026-03-13).

### IR PIPELINE FIX: Flat-Owned Pre-Population (2026-03-13)

**Root problem:** `lower_program_to_ir` → `preseed_program_items_ref` → `find_or_add_fn_id` writes `(*lo.module).fn_count = idx+1` (broken). All functions get fn_id=0 because fn_count never increments.

**Fix:** Use `lower_program_to_ir_summary_flat_owned_with_epistemic_ref` (works on LOCAL IrModule, not Box) then `lowerer_new_from_program_summary_flat` to create a Lowerer with correct fn_count pre-seeded via copy-out pattern.

**`flush_current_func` already uses copy-out correctly:**
```sio
var module = *lo.module          // copy-out from Box — works
module.functions[lo.current_fn as usize] = *lo.current_func  // 1-level write — works
lo.module = Box::new(module)     // put back — works
```

**Verified (2026-03-13):** `ir_pipeline_v2.sio` → add_i64:id=0, negate:id=1, square:id=2, all instrs>0.

**New files:**
- `self-hosted/ir/lower_prepopulate.sio` — `lowerer_prepopulate_from_program_ref(prog, epistemic)` wrapper
- `self-hosted/compiler/ir_pipeline_v2.sio` — IR-only verification driver; prints fn_ids + instr_counts
