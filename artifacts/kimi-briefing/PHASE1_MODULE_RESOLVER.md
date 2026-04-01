# Kimi 2.5 Briefing: Phase 1 -- BFS Module Resolver for lean_single.sio

## Your Mission

Add multi-file module resolution to the Sounio self-hosted native compiler (`self-hosted/compiler/lean_single.sio`, 15,459 lines). After your changes, programs using `use math::sedenion64::*` will compile natively by resolving the path to `stdlib/math/sedenion64.sio`, reading the file, and concatenating it into the SRC buffer before lexing.

The existing `resolve_imports()` function (line 15061) already does most of this. Your job is to fix the gaps so it handles the full Sounio module path syntax and transitive dependencies correctly.

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
| `x.push(v)` | Manual index: `ARR[COUNT] = v; COUNT = COUNT + 1` |
| `for x in iter` | `var i = 0; while i < n { ... i = i + 1 }` |
| `String::from("x")` | `"x"` (string literals are the only strings) |
| Closures `\|x\| x+1` | Not supported; use named function refs |
| Semicolons anywhere | NEVER. Sounio has NO semicolons |

**Read these files first:**
- `docs/guide/MINIMUM_VIABLE_SOUNIO.md` -- Full syntax reference
- `docs/guide/SOUNIO_GOTCHAS.md` -- Common mistakes

---

## What Already Exists

### Current `resolve_imports()` (line 15061-15309)

The function already:
1. Scans SRC byte-by-byte for lines starting with `use ` (bytes 117,115,101,32)
2. Extracts the module path (e.g., `math::sedenion64::*`)
3. Converts `::` to `/` to build a filesystem path (e.g., `math/sedenion64.sio`)
4. Tries multiple search paths: `BASE_DIR/path`, raw path, `self-hosted/path`, `stdlib/path`
5. Reads the file via `read_file()` syscall
6. Appends file contents to SRC buffer with marker bytes (0x01 path 0x02 content)
7. Deduplicates by hash + full path comparison
8. Since the loop condition is `while pos < SRC_LEN` and imports extend SRC_LEN, newly imported files are scanned too -- this IS BFS

### Key global buffers (lines 9-21)

```
SRC: [i8; 8388608]         -- 8MB source buffer (all files concatenated)
SRC_LEN: i64               -- current end of source
IMP_PATH: [i8; 4096]       -- scratch: resolved filesystem path
IMP_RAW_PATH: [i8; 4096]   -- scratch: raw module path before base_dir prepend
IMP_LOADED: [i64; 1024]    -- hash of each loaded import path (for dedup)
IMP_LOADED_PATHS: [i8; 262144] -- packed loaded path strings
IMP_LOADED_OFFS: [i64; 1024]   -- offset into IMP_LOADED_PATHS per import
IMP_LOADED_LENS: [i64; 1024]   -- length of each loaded path
IMP_COUNT: i64              -- number of imports loaded so far
BASE_DIR: [i8; 512]        -- directory of the entry-point file
BASE_DIR_LEN: i64
```

### How the lexer handles imports

The lexer (line 3066) treats bytes 0x01 and 0x02 as whitespace, so the `\x01path\x02` markers are silently skipped. The `use` keyword is NOT a lexer keyword -- it gets lexed as identifier (token kind 3). Since the compiler's pass 1 only looks for `fn` (kind 1), `struct` (kind 46), `enum` (kind 47), etc., the `use` lines are harmlessly ignored after import resolution.

### How `pub` works

`pub` is token kind 49 (line 3201). The current compiler lexes it but does NOT enforce visibility. All functions, structs, and globals from imported files are visible. This is fine for Phase 1.

---

## What Needs to Change

### Problem 1: Named imports not resolved to files

Lines like `use math::cayley_dickson::{oct64_mul_components, oct64_conj_components}` extract the path up to the `::` before `{`, producing `math/cayley_dickson.sio`. This already works.

But the search order is wrong for stdlib modules. The current search tries:
1. `BASE_DIR/math/cayley_dickson.sio` (wrong -- BASE_DIR is the entry file's dir)
2. `math/cayley_dickson.sio` (relative CWD -- works only if CWD is repo root)
3. `self-hosted/math/cayley_dickson.sio` (wrong prefix)
4. `stdlib/math/cayley_dickson.sio` (correct -- but only tried as last resort)

**Fix:** Move the `stdlib/` prefix fallback BEFORE the `self-hosted/` fallback. Or better: detect that `math`, `epistemic`, `core`, `collections`, etc. are stdlib top-level modules and immediately prepend `stdlib/`.

### Problem 2: SOUNIO_STDLIB_PATH not read

When compiling outside the repo, `SOUNIO_STDLIB_PATH` environment variable should be used. The compiler does not read it. Add a function to read it at startup using the `read_env` pattern (if available in the native compiler) or skip this for Phase 1.

### Problem 3: Transitive imports in stdlib files

Many stdlib files have their own `use` statements (e.g., `stdlib/math/sedenion64.sio` uses `math::cayley_dickson`). Since `resolve_imports` already does BFS (appends to SRC, loop reads SRC_LEN), these are handled automatically. BUT: the BASE_DIR changes via the `\x01path\x02` marker, so the resolver needs to correctly update BASE_DIR for each imported file's directory. Verify that `set_base_dir_from_src_path` (line 14782) correctly handles `stdlib/math/sedenion64.sio` -> BASE_DIR = `stdlib/math`.

### Problem 4: `mod.sio` convention

Some modules use `mod.sio` as the entry point (e.g., `use check::mod` maps to `self-hosted/check/mod.sio`). The path converter produces `check/mod.sio` which is correct. But for `use collections::vec` it should try both `stdlib/collections/vec.sio` and `stdlib/collections/vec/mod.sio`. Add a `mod.sio` fallback.

---

## Implementation Plan (4 Steps)

### Step 1: Add stdlib-prefix detection

At line ~15100, after the `::` to `/` conversion but before the search cascade, add detection of known stdlib top-level module names. If the first path segment matches a known stdlib directory, immediately try `stdlib/` prefix.

Add a helper function:

```sio
fn is_stdlib_prefix(s: i64, e: i64) -> i64 with Panic, Mut {
    let len = e - s
    if src_match_buf(IMP_PATH, 0, len, "math") { return 1 }
    if src_match_buf(IMP_PATH, 0, len, "epistemic") { return 1 }
    if src_match_buf(IMP_PATH, 0, len, "core") { return 1 }
    if src_match_buf(IMP_PATH, 0, len, "collections") { return 1 }
    if src_match_buf(IMP_PATH, 0, len, "algebra") { return 1 }
    if src_match_buf(IMP_PATH, 0, len, "units") { return 1 }
    if src_match_buf(IMP_PATH, 0, len, "graph") { return 1 }
    if src_match_buf(IMP_PATH, 0, len, "io") { return 1 }
    if src_match_buf(IMP_PATH, 0, len, "bayes") { return 1 }
    if src_match_buf(IMP_PATH, 0, len, "data") { return 1 }
    if src_match_buf(IMP_PATH, 0, len, "cybernetic") { return 1 }
    if src_match_buf(IMP_PATH, 0, len, "integrate") { return 1 }
    if src_match_buf(IMP_PATH, 0, len, "complex") { return 1 }
    if src_match_buf(IMP_PATH, 0, len, "functional") { return 1 }
    return 0
}
```

**However**, `src_match_buf` does not exist. You need to write a helper that compares `IMP_PATH[0..first_slash]` against a string literal. Use the same byte-comparison pattern as `src_match` (line ~603) but operating on `IMP_PATH` instead of `SRC`.

Actually, a simpler approach: find the first `/` in `IMP_PATH[0..fp_len]`, extract the segment before it, and check if `stdlib/<segment>` is a valid directory. But `stat()` syscalls are expensive. Instead, use a hardcoded list OR just always try `stdlib/` prefix as the FIRST fallback instead of last.

**Recommended change:** In the search cascade (lines 15132-15215), reorder so `stdlib/` is tried SECOND (after BASE_DIR), not fourth:

Current order: BASE_DIR -> raw -> self-hosted -> stdlib
New order: BASE_DIR -> stdlib -> raw -> self-hosted

This is a minimal change: move the `stdlib/` block (lines 15196-15214) up to right after the first `raw` attempt (line 15163).

### Step 2: Add `mod.sio` fallback

After each path resolution attempt, if the file is not found, also try appending `/mod.sio` instead of `.sio`. For example, `collections/vec.sio` fails -> try `collections/vec/mod.sio`.

Add this after line 15124 (after `.sio` is appended):

```sio
// Save the .sio path length for mod.sio fallback
let sio_fp_len = fp_len
// Build mod.sio alternative: replace trailing .sio with /mod.sio
var MOD_PATH: [i8; 4096] = [0; 4096]
var mod_fp_len: i64 = fp_len - 4  // strip .sio
// Copy path without .sio
var mpi: i64 = 0
while mpi < mod_fp_len {
    MOD_PATH[mpi as usize] = IMP_PATH[mpi as usize]
    mpi = mpi + 1
}
// Append /mod.sio
MOD_PATH[mod_fp_len as usize] = 47     // /
MOD_PATH[(mod_fp_len+1) as usize] = 109  // m
MOD_PATH[(mod_fp_len+2) as usize] = 111  // o
MOD_PATH[(mod_fp_len+3) as usize] = 100  // d
MOD_PATH[(mod_fp_len+4) as usize] = 46   // .
MOD_PATH[(mod_fp_len+5) as usize] = 115  // s
MOD_PATH[(mod_fp_len+6) as usize] = 105  // i
MOD_PATH[(mod_fp_len+7) as usize] = 111  // o
mod_fp_len = mod_fp_len + 8
MOD_PATH[mod_fp_len as usize] = 0
```

**WAIT** -- you cannot declare `var MOD_PATH` inside a function with boot4 compilation (local arrays are allowed but expensive). Instead, add a global buffer:

```sio
var MOD_PATH: [i8; 4096] = [0; 4096]
```

at the top of the file (around line 13, near `IMP_PATH`). Then in the fallback logic, when `read_file(IMP_PATH)` returns empty, try `read_file(MOD_PATH)`.

### Step 3: Name collision avoidance

**Current state:** When files are concatenated into SRC, all top-level names (functions, structs, enums) are globally visible. If two imported modules define a function with the same name, `fn_find` returns the first one. This is the SAME bug as BUG3 (impl dispatch), but for modules.

**Phase 1 strategy: do NOT prefix names.** Instead, rely on the existing dedup check -- if a function name is already registered, `tc_duplicate_fn(p)` is called (line 10069). For Phase 1, change this from an error to a silent skip (last-wins or first-wins semantics). This matches how the JIT compiler handles glob imports.

Alternatively, do nothing -- the current behavior is that duplicate function names emit a warning but compilation continues. The first definition wins (line 10068 returns >= 0 and the duplicate is not registered). This is acceptable for Phase 1.

**Phase 2 (future):** Add module-qualified name resolution (e.g., `sedenion64::sed64_add` dispatches correctly even if another module has `sed64_add`). This requires changes to `fn_find` and the call-site compilation, which is out of scope for Phase 1.

### Step 4: Verify transitive BFS works

The existing loop `while pos < SRC_LEN` in `resolve_imports` already provides BFS:
1. Main file loaded into SRC[0..N]
2. resolve_imports scans, finds `use math::sedenion64::*`
3. Loads `stdlib/math/sedenion64.sio`, appends to SRC[N..M], SRC_LEN = M
4. Continues scanning, reaches the appended content
5. Finds `use math::cayley_dickson::...` in sedenion64.sio
6. Loads `stdlib/math/cayley_dickson.sio`, appends to SRC[M..P], SRC_LEN = P
7. Continues scanning until pos >= SRC_LEN

This is already implemented. The only risk is that `set_base_dir_from_src_path` (called when a `\x01path\x02` marker is encountered) might not correctly parse `stdlib/math/sedenion64.sio` as BASE_DIR = `stdlib/math`. Verify that the marker contains the full resolved path (it does -- lines 15265-15270 write IMP_PATH into the marker).

---

## Files to Edit

| File | What to change |
|---|---|
| `self-hosted/compiler/lean_single.sio` | Reorder search cascade, add mod.sio fallback, add MOD_PATH global |

Only ONE file needs editing. Everything is in lean_single.sio.

---

## Where to Make Each Edit

| Line | What |
|---|---|
| ~13 | Add `var MOD_PATH: [i8; 4096] = [0; 4096]` global buffer |
| ~15098-15116 | After `::` to `/` conversion: build MOD_PATH variant |
| ~15132-15215 | Reorder search cascade: stdlib BEFORE self-hosted |
| ~15162-15215 | After each `read_file` attempt that fails, also try MOD_PATH |

---

## Detailed Edit: Reorder Search Cascade

Current code (lines 15132-15215) does this:

```
1. needs_base_dir? -> prepend BASE_DIR/
2. read_file(IMP_PATH)
3. if fail && needs_base_dir:
   a. try raw path (no prefix)
   b. if fail: try self-hosted/ prefix
   c. if fail: try stdlib/ prefix
```

Change to:

```
1. needs_base_dir? -> prepend BASE_DIR/
2. read_file(IMP_PATH)
3. if fail && needs_base_dir:
   a. try stdlib/ prefix          <-- moved UP
   b. if fail: try raw path
   c. if fail: try self-hosted/ prefix
```

The actual code to move: take the block at lines 15196-15214 (the `stdlib/` attempt) and swap it with the block at lines 15162-15172 (the raw path attempt). Keep the same structure, just reorder.

---

## Detailed Edit: mod.sio Fallback

After each `read_file(IMP_PATH)` call that returns empty (`copy_len <= 0`), try:

```sio
// Try mod.sio fallback
if copy_len <= 0 {
    // Build mod.sio path: strip .sio, append /mod.sio
    var mj: i64 = 0
    var mfp: i64 = fp_len - 4  // remove .sio
    while mj < mfp {
        MOD_PATH[mj as usize] = IMP_PATH[mj as usize]
        mj = mj + 1
    }
    MOD_PATH[mfp as usize] = 47
    MOD_PATH[(mfp+1) as usize] = 109
    MOD_PATH[(mfp+2) as usize] = 111
    MOD_PATH[(mfp+3) as usize] = 100
    MOD_PATH[(mfp+4) as usize] = 46
    MOD_PATH[(mfp+5) as usize] = 115
    MOD_PATH[(mfp+6) as usize] = 105
    MOD_PATH[(mfp+7) as usize] = 111
    mfp = mfp + 8
    MOD_PATH[mfp as usize] = 0
    raw = read_file(MOD_PATH)
    copy_len = str_len(raw)
    fsz = file_size(MOD_PATH)
    if copy_len > 0 {
        // Switch IMP_PATH to MOD_PATH for dedup tracking
        mj = 0
        while mj < mfp {
            IMP_PATH[mj as usize] = MOD_PATH[mj as usize]
            mj = mj + 1
        }
        fp_len = mfp
        IMP_PATH[fp_len as usize] = 0
    }
}
```

Insert this block after EACH `read_file` + `copy_len` check in the cascade. There are 4 such checks (lines 15162, 15173, 15196, 15217). Add the mod.sio fallback after each one.

**Optimization:** To avoid repeating the mod.sio construction 4 times, build MOD_PATH once (right after line 15124 where fp_len is finalized for the raw path), store `mod_fp_len`, and reuse it. Each time IMP_PATH changes (due to prefix prepend), also rebuild MOD_PATH with the same prefix.

---

## Testing Procedure

### Test 1: Direct stdlib import

Create `tests/run-pass/test_module_import.sio`:

```sio
use math::constants::*

fn main() -> i64 with IO {
    print("module import test\n")
    0
}
```

Compile and run:
```bash
./bin/souc compile tests/run-pass/test_module_import.sio -o /tmp/test_mod.elf
/tmp/test_mod.elf
```

Expected: prints "module import test" and exits 0. The import line should log `import: stdlib/math/constants.sio NNN bytes`.

### Test 2: Transitive import (BFS)

Create `tests/run-pass/test_transitive_import.sio`:

```sio
use math::sedenion64::*

fn main() -> i64 with IO {
    print("transitive import test\n")
    0
}
```

This should resolve `math::sedenion64` -> `stdlib/math/sedenion64.sio`, which itself imports `math::cayley_dickson`, triggering the BFS to also load `stdlib/math/cayley_dickson.sio`.

### Test 3: Self-hosted import

```bash
./bin/souc compile self-hosted/compiler/lean_single.sio -o /tmp/gen1.elf
```

This is the self-compilation test. lean_single.sio has no `use` statements (it is self-contained), so this verifies the changes don't break the existing code path.

### Test 4: Duplicate function names

Create a file that imports two modules with identically-named helper functions. Verify compilation succeeds (first-wins semantics) and the correct function is called.

---

## Constraints

- SRC buffer is 8MB (`8388608` bytes). Total concatenated source must fit.
- IMP_COUNT max is 1024 imports. Enough for any real program.
- All edits must be in lean_single.sio only.
- No semicolons. No Rust macros. No closures. Effects clauses on every function.
- Test with both `./bin/souc` (JIT) and the self-hosted binary if available.

---

## Key Functions Reference

| Function | Line | Purpose |
|---|---|---|
| `sb(i)` | 596 | Read byte from SRC at position i |
| `src_match(s, len, lit)` | ~603 | Compare SRC[s..s+len] to string literal |
| `resolve_imports()` | 15061 | Main import resolution loop |
| `set_base_dir_from_src_path(ps, pe)` | 14782 | Update BASE_DIR from a file path |
| `imp_path_hash(len)` | 14763 | Hash IMP_PATH for dedup |
| `print_imp_path()` | 14774 | Print current IMP_PATH for diagnostics |
| `read_file(path)` | builtin | Read file contents (syscall wrapper) |
| `file_size(path)` | builtin | Get file size (syscall wrapper) |
| `str_len(s)` | builtin | String length |
| `compile_all()` | 9823 | Main compilation (Pass 1-3) |
| `lex_all()` | 3057 | Tokenize SRC into TK/TS/TE/TV arrays |

---

## What NOT to Do

- Do NOT add `use` as a lexer keyword. It is handled pre-lex.
- Do NOT enforce `pub` visibility filtering. Phase 1 exposes all symbols.
- Do NOT implement module-qualified name resolution (e.g., `sedenion64::sed64_add`). Phase 1 uses flat namespace.
- Do NOT add new syscalls. `read_file` and `file_size` are sufficient.
- Do NOT change the marker byte protocol (0x01 path 0x02). It works.
- Do NOT add semicolons. Ever.
