# Door 1: Break the Fixed-Array Wall

## Mission

Remove the 256-variable-per-function limit and enable large array allocations (64K+ elements) in `lean_single.sio`. This is the single biggest architectural blocker preventing Sounio from supporting real ML/DL workloads.

## Context

The self-hosted native compiler (`self-hosted/compiler/lean_single.sio`) uses parallel arrays of size 256 to track local variable metadata:

```sio
var VAR_NS:      [i64; 256]   // name start
var VAR_NE:      [i64; 256]   // name end
var VAR_SLOT:    [i64; 256]   // stack slot
var VAR_TY:      [i64; 256]   // type tag
var VAR_TY_HASH: [i64; 256]   // type hash
var VAR_ALEN:    [i64; 256]   // array length
var VAR_ESIZ:    [i64; 256]   // element size
var VAR_IS_F64:  [i64; 256]   // float flag
var VAR_MUT:     [i64; 256]   // mutable flag
var VAR_IS_CLOSURE: [i64; 256]
var VAR_LEN_SLOT: [i64; 256]  // slice length slot
var VAR_USED:    [i64; 256]
var VAR_LINEAR:  [i64; 256]
var VAR_UNIT:    [i64; 256]
var VAR_COUNT: i64 = 0
```

Similarly, globals are limited to 256:
```sio
var GL: [i64; 1024] = [0; 1024]     // 4 fields × 256 globals
var GL_TY: [i64; 256] = [0; 256]
var GL_TY_HASH: [i64; 256] = [0; 256]
var GL_COUNT: i64 = 0
```

Array elements don't consume VAR slots — they consume `NEXT_SLOT` entries on the stack frame. A `[i64; 256]` uses 257 slots (256 elements + 1 pointer). The stack frame is calculated as `(NEXT_SLOT * 8 + 15) & ~0xF` with no upper bound check.

**The 256 limit is on metadata capacity, not array size.** You can declare `[i64; 10000]` if you have VAR slots available.

## Current Architecture

### Stack allocation flow:
1. `var_add_arr(name, alen, esiz, elem_ty)` — allocates `alen` slots via `NEXT_SLOT += alen`, stores metadata in `VAR_*[VAR_COUNT]`
2. Array init via `rep stosq` (3 instructions for any size) — recently optimized
3. Frame size: `sub rsp, frame_sz` at function entry

### Heap allocation:
- `Box::new(expr)` — hardcoded `mmap(4096)` syscall, stores single value
- No dynamic sizing, no free/dealloc

### BSS (global data):
- `GL_BSS_SIZE` accumulates, placed in ELF BSS section
- No per-global size limit, but max 256 globals
- BSS is zero-initialized by the OS loader

### Other limits:
- Functions: 4,096 (`FN_COUNT`)
- Structs: 512 (`ST_COUNT`)
- Struct fields: 64 per struct
- Tokens: 32,767 (lexer)

## Required Changes

### Phase 1: Expand metadata arrays (minimum viable)

Increase all VAR_* arrays from 256 to 1024:
```sio
var VAR_NS:      [i64; 1024]
var VAR_NE:      [i64; 1024]
var VAR_SLOT:    [i64; 1024]
// ... all 14 VAR_* arrays
var VAR_COUNT: i64 = 0
```

Similarly for globals:
```sio
var GL: [i64; 4096] = [0; 4096]     // 4 fields × 1024 globals
var GL_TY: [i64; 1024] = [0; 1024]
var GL_TY_HASH: [i64; 1024] = [0; 1024]
```

**Why 1024:** Each VAR_* array at 1024 entries = 8KB. 14 arrays × 8KB = 112KB total. BSS can handle this easily. The compiler already uses ~55MB BSS.

**Search-and-replace pattern:** Every `[i64; 256]` in the VAR_* and GL_* declarations needs to become `[i64; 1024]`. The GL array needs `[i64; 4096]` (4 fields per global × 1024).

### Phase 2: Add stack frame size guard

After calculating `frame_sz`, add a guard:
```sio
let frame_sz = (NEXT_SLOT * 8 + 15) & 0xFFFFFFF0
if frame_sz > 4194304 {  // 4MB limit (half of typical 8MB stack)
    print("error: stack frame too large (")
    print_int(frame_sz)
    print(" bytes) — consider using global arrays\n")
    return 1
}
```

Insert this right after the frame size calculation (search for `let frame_sz =` in `compile_all()`).

### Phase 3: Dynamic Box::new sizing

Replace the hardcoded 4096 mmap with a size based on the expression:
```sio
// Before: mmap(NULL, 4096, ...)
// After: mmap(NULL, size, ...) where size = max(4096, element_count * 8)
```

For `Box::new(array)`, compute the array's byte size and round up to page boundary (4096). This enables `Box::new([0; 65536])` = heap-allocated 512KB array.

Implementation: after `compile_or()` produces the value, check `EXPR_TY == 8` (array type). If so, compute `arr_storage_slots(EXPR_TY_HASH) * 8` and use that as the mmap size (rounded up to 4096 boundary).

### Phase 4: BSS spill for large local arrays (optional, advanced)

When a local array exceeds a threshold (e.g., 4096 elements), automatically allocate it in BSS instead of on the stack:
1. Add a global BSS slot for the array
2. At function entry, emit `lea rax, [bss_addr]` instead of `lea rax, [rbp - offset]`
3. Mark the variable as BSS-backed in VAR metadata

This is optional but would eliminate stack overflow for very large arrays.

## Hard Constraints

- **Self-host must preserve**: gen2==gen3 bit-identical after every change
- **No regressions**: All existing run-pass tests must pass
- **Atomic commits**: One logical change per commit
- **Sounio syntax**: `var` not `let mut`, `&!` not `&mut`, no semicolons
- **BSS growth is OK**: The compiler already uses ~55MB BSS; adding 100KB is negligible
- **Do NOT change array init codegen**: `rep stosq` already handles any size efficiently

## Verification

After each phase:
1. Self-host chain: `gen1.elf` → `gen2.elf` → `gen3.elf`, verify `md5(gen2) == md5(gen3)`
2. Run key tests:
   ```bash
   ./bin/souc run tests/run-pass/algebra_g2_invariants.sio
   ./bin/souc run tests/run-pass/bdf_stiff.sio
   ./bin/souc run tests/run-pass/ontology_roles_basic.sio
   ./bin/souc run tests/run-pass/hypothesis_registered.sio
   ```
3. Test large array:
   ```sio
   fn main() -> i32 with IO, Mut, Panic {
       var weights: [f64; 4096] = [0.0; 4096]
       weights[0] = 1.0
       weights[4095] = 2.0
       let sum = weights[0] + weights[4095]
       if sum > 2.9 && sum < 3.1 { println("PASS: 4096-element array") }
       else { println("FAIL") }
       0
   }
   ```
4. Test many variables (>256):
   ```sio
   fn main() -> i32 with IO, Mut, Panic {
       var a0: i64 = 0
       var a1: i64 = 1
       // ... (generate 300+ var declarations)
       var a299: i64 = 299
       let sum = a0 + a299
       if sum == 299 { println("PASS: 300 variables") }
       else { println("FAIL") }
       0
   }
   ```

## Files to Modify

| File | Change |
|------|--------|
| `self-hosted/compiler/lean_single.sio` | Expand VAR_*/GL_* arrays, add frame guard, improve Box::new |
| `artifacts/self-hosted/souc-self-hosted-x86_64` | Rebuilt binary |

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Max variables per function | 256 | 1024 |
| Max globals | 256 | 1024 |
| Max array elements (stack) | ~1M (no guard) | ~512K (with 4MB guard) |
| Max array elements (heap) | 512 (Box = 4KB) | 524K (Box = dynamic) |
| Neural net layer size | 256 params | 65K+ params |

## What This Unblocks

- Dense layers with 1024+ neurons
- Attention heads with 512-dim keys/queries
- Weight matrices up to 256×256 (65K elements)
- Batch processing with 1000+ samples
- State space models with 1024-dim hidden state
- Real ML/DL training, not toy demos
