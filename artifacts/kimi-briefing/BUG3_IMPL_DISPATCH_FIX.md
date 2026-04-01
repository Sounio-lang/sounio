# Kimi 2.5 Briefing: Fix Bug 3 — Impl Method Name Collision in lean_single.sio

## Your Mission

Fix the method dispatch collision bug in the Sounio self-hosted compiler (`self-hosted/compiler/lean_single.sio`, 15,459 lines). After your fix, two structs can both have methods named `new` (or any shared name) and the compiler resolves the correct one based on receiver type.

---

## CRITICAL: Sounio Is NOT Rust

You are editing a compiler written IN Sounio. These rules are absolute:

| Rust (WRONG) | Sounio (CORRECT) |
|---|---|
| `let x = 5;` | `let x = 5` (no semicolons EVER) |
| `let mut x = 5;` | `var x: i64 = 5` |
| `&mut T` | `&!T` |
| `assert!(cond)` | `assert(cond)` |
| `println!("hi")` | `println("hi")` |
| `fn foo() { }` | `fn foo() with IO { }` (effects required) |
| `#[test]` | No attributes exist |
| `Vec<T>` | Fixed arrays: `[i64; 8192]` |

**Read these files first:**
- `docs/guide/MINIMUM_VIABLE_SOUNIO.md` — Full syntax reference
- `docs/guide/SOUNIO_GOTCHAS.md` — Common mistakes

---

## The Bug

### Symptom
When the self-hosted compiler processes programs with multiple structs that share method names (e.g., both `Foo` and `Bar` have a method called `new`), it calls the wrong function at runtime → crash or incorrect behavior.

### Root Cause
`fn_find()` at line ~3949 does a flat linear scan by name only:

```sio
fn fn_find(ns: i64, ne: i64) -> i64 with Panic, Mut {
    var i: i64 = 0
    while i < FN_COUNT {
        if name_eq(FN_NS[i as usize], FN_NE[i as usize], ns, ne) { return i }
        i = i + 1
    }
    return -1
}
```

When `bar.new()` is compiled, `fn_find` searches for "new" and returns the **first** function named "new" — which belongs to `Foo`, not `Bar`. No receiver type is checked.

### Why It Exists
The compiler has no `impl` block support. Methods are bare functions with an explicit `self` parameter. The function table (`FN_NS[]`, `FN_NE[]`, etc.) has no struct association metadata.

---

## The Fix (4 Steps)

### Step 1: Add `FN_RECV_HASH` Array

At line ~81 (after `FN_COUNT`), add:

```sio
var FN_RECV_HASH: [i64; 8192] = [0; 8192]
```

This stores the struct type hash of the receiver for each function. Value `0` = free function (no receiver). Non-zero = method of that struct type.

### Step 2: Populate `FN_RECV_HASH` at Registration

In **Pass 1** (line ~10060), when registering functions, detect if the first parameter is a struct type and store its hash:

Current code at lines ~10060-10077:
```sio
// Pass 1: register function names
while p < TC {
    if TK[p as usize] == 1 && TK[(p + 1) as usize] == 3 {
        // ... existing registration ...
        FN_NS[FN_COUNT as usize] = TS[p as usize]
        FN_NE[FN_COUNT as usize] = TE[p as usize]
        FN_OFF[FN_COUNT as usize] = -1
        FN_COUNT = FN_COUNT + 1
    }
    p = p + 1
}
```

**After** setting `FN_NS` and `FN_NE`, scan forward to find the first parameter's type. If it matches a known struct name (via `st_find` or hash lookup), set `FN_RECV_HASH[FN_COUNT] = hash`. This requires a forward scan from the `fn` token to find `(param_name: TypeName, ...)`.

**Simpler approach:** In Pass 2 (when function bodies are compiled and parameter types are known), backfill `FN_RECV_HASH[fi]` based on the first parameter's type hash. This is safer because type resolution is complete by Pass 2.

### Step 3: Add `fn_find_method()` for Qualified Lookup

Add a new function near line ~3956:

```sio
fn fn_find_method(ns: i64, ne: i64, recv_hash: i64) -> i64 with Panic, Mut {
    // First: try exact match (name + receiver type)
    var i: i64 = 0
    while i < FN_COUNT {
        if name_eq(FN_NS[i as usize], FN_NE[i as usize], ns, ne) {
            if FN_RECV_HASH[i as usize] == recv_hash { return i }
        }
        i = i + 1
    }
    // Fallback: any function with this name (backward compatible)
    i = 0
    while i < FN_COUNT {
        if name_eq(FN_NS[i as usize], FN_NE[i as usize], ns, ne) { return i }
        i = i + 1
    }
    return -1
}
```

### Step 4: Update Method Call Sites

**X86 method call** at line ~7396:
```sio
// BEFORE:
let mfi = fn_find(fns, fne)

// AFTER:
// Get receiver type hash from the expression being called on
let recv_h = expr_type_hash  // You need to track this during expression compilation
let mfi = fn_find_method(fns, fne, recv_h)
```

**ARM64 method call** at line ~13351:
Same change.

The receiver type hash comes from the struct type of the expression before the `.` dot. The compiler already tracks struct types during expression compilation (for field access). Find where `st_find` is called during dot-expression handling and reuse that struct index to get the hash.

---

## Key Data Structures You Need to Understand

### Function Table (lines 70-81)
```
FN_NS: [i64; 8192]     — Source start offset of function name
FN_NE: [i64; 8192]     — Source end offset of function name
FN_OFF: [i64; 8192]    — Code offset in output binary
FN_SIG: [i64; 8192]    — Signature metadata
FN_ARITY: [i64; 8192]  — Number of parameters
FN_RET_TY: [i64; 8192] — Return type tag
FN_RET_HASH: [i64; 8192] — Return type hash
FN_EFFECTS: [i64; 8192] — Effect bitmask
FN_COUNT: i64           — Number of registered functions
```

### Struct Table (lines 109-117)
```
ST: [i64; 26000]       — Struct metadata (130 slots per struct)
ST_FTY: [i64; 25600]   — Field types
ST_FHASH: [i64; 25600] — Field type hashes  
ST_COUNT: i64           — Number of registered structs
```

Each struct uses 130 slots in `ST[]`:
- `ST[si*130]` = name hash
- `ST[si*130 + 1]` = field count
- `ST[si*130 + 2 + k]` = field name hash for field k

### Name Comparison
```sio
fn name_eq(ns1: i64, ne1: i64, ns2: i64, ne2: i64) -> bool
```
Compares source substrings by offset ranges.

### Source Access
Source code is in `SRC: [i8; 8388608]`. `TS[tok]`/`TE[tok]` give start/end offsets for token `tok` in the source.

---

## How Method Calls Work Currently

When compiling `foo.bar(arg1, arg2)`:

1. `foo` is compiled as an expression → value on stack or in rax
2. Parser sees `.` (dot token), then `bar` (identifier), then `(` (open paren)
3. Compiler calls `fn_find("bar")` — **THIS IS THE BUG POINT**
4. Found function index `mfi` is used
5. `foo` is pushed as first argument (implicit self)
6. `arg1`, `arg2` compiled and pushed
7. Call is emitted as `call FN_OFF[mfi]`

The fix changes step 3 to also consider the TYPE of `foo`.

---

## Tracking Receiver Type During Expression Compilation

The compiler already knows the type of expressions during compilation. Look for:

- `VAR_TY_HASH[]` — Variable type hashes (used to resolve struct field access)
- When a `.field` access is compiled, the compiler calls `st_find()` to get the struct index
- The struct's type hash is `ST[si * 130]`

**The receiver hash for a method call is the same hash used for field access on the expression before the dot.**

Search for how the compiler handles `obj.field` (non-method dot access) — it resolves the struct type there. The method call path should use the same type resolution, then pass the hash to `fn_find_method()`.

---

## Testing the Fix

After implementing, verify with:

```bash
SOUC=./bin/souc

# 1. Check the modified lean_single.sio compiles
$SOUC check self-hosted/compiler/lean_single.sio

# 2. Bootstrap chain
artifacts/self-hosted/souc-self-hosted-x86_64 self-hosted/compiler/lean_single.sio gen1.elf
./gen1.elf self-hosted/compiler/lean_single.sio gen2.elf
./gen2.elf self-hosted/compiler/lean_single.sio gen3.elf

# 3. Fixed point check
md5sum gen2.elf gen3.elf
# gen2.elf and gen3.elf MUST have identical md5

# 4. Test with collision case
cat > /tmp/test_impl_collision.sio << 'EOF'
struct Foo { x: i64 }
struct Bar { y: i64 }

fn new_foo() -> Foo { Foo { x: 42 } }
fn new_bar() -> Bar { Bar { y: 99 } }

fn value(self: Foo) -> i64 { self.x }
fn value(self: Bar) -> i64 { self.y }

fn main() with IO {
    let f = new_foo()
    let b = new_bar()
    print_i64(f.value())  // Should print 42
    print_i64(b.value())  // Should print 99
}
EOF
./gen2.elf /tmp/test_impl_collision.sio /tmp/test_collision.elf
/tmp/test_collision.elf
# Expected output: 42 then 99
```

---

## Files to Read (in order)

1. `docs/guide/MINIMUM_VIABLE_SOUNIO.md` — Syntax reference (MUST READ FIRST)
2. `docs/guide/SOUNIO_GOTCHAS.md` — Common mistakes
3. `self-hosted/compiler/lean_single.sio` lines 70-181 — Array declarations
4. `self-hosted/compiler/lean_single.sio` lines 3949-3968 — `fn_find` / `fn_find_lit`
5. `self-hosted/compiler/lean_single.sio` lines 7300-7430 — X86 method call compilation
6. `self-hosted/compiler/lean_single.sio` lines 10060-10080 — Pass 1 function registration
7. `self-hosted/compiler/lean_single.sio` lines 13270-13400 — ARM64 method call compilation

---

## Constraints

- **Do NOT add semicolons.** Sounio has NO semicolons.
- **Do NOT use `let mut`.** Use `var`.
- **Do NOT use `&mut`.** Use `&!`.
- **Keep changes minimal.** Add the new array, the new lookup function, and update the call sites. Do not refactor surrounding code.
- **Fixed arrays only.** `[i64; 8192]` not `Vec<i64>`.
- **All new functions must declare effects.** `with Panic, Mut` for anything that reads/writes globals.
- **The self-hosting fixed point must be preserved.** gen2.elf == gen3.elf after the fix.

---

## Expected Diff Size

~50-80 lines added/changed:
- 1 new array declaration (1 line)
- 1 new function `fn_find_method` (~15 lines)
- Pass 1/2 receiver hash population (~15-20 lines)
- 2 method call site updates (x86 + arm64) (~10 lines each)
- Test adjustments if needed

This is a surgical fix. Do not restructure the compiler.
