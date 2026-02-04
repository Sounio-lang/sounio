# Self-Hosting Bootstrap: Delegate Tasks

**Status:** 3/34 stdlib modules compile. 30 modules blocked by known issues.
**Target:** Enable all 34 stdlib modules to compile through bytecode codegen pipeline.

## Task Distribution

### Priority 1: Bytecode Codegen - Complex Assignments (2 modules)
**Modules:** `parser::expr`, `parser::stmt`
**Issue:** Bytecode codegen doesn't support field/index assignment targets
**Error:** `Unsupported: Complex assignment target`

**Specific fix needed:**
- File: `crates/souc/src/codegen/bytecode.rs`
- Function: `compile_expr_assign()`
- Add cases for:
  - Field assignment: `expr.field = value` → bytecode for struct field write
  - Index assignment: `arr[idx] = value` → bytecode for array write
- Generate: LOAD expr, LOAD value, STORE_FIELD/STORE_INDEX
- Tests: Add 2-3 cases for each assignment type

**Effort:** 50 LOC, 2-3 hours

---

### Priority 2: Type Checker - Duplicate Externs (2 modules)
**Modules:** `parser::expr_parse`, `parser::mod`
**Issue:** Type checker rejects duplicate extern declarations
**Error:** `Duplicate definition: print` (in extern blocks)

**Specific fix needed:**
- File: `crates/souc/src/check/mod.rs`
- Function: `check_extern_decl()` or similar
- Change: Merge duplicates with same signature instead of error
- Logic:
  ```
  If extern with same name exists:
    Check signature matches
    If yes: skip (use first definition)
    If no: error (signature mismatch)
  ```

**Effort:** 30 LOC, 1-2 hours

---

### Priority 3: Parser - Expression Errors (3 modules)
**Modules:** `parser::expr_simple`, `parser::stmt_parse`, `check::env`
**Issue:** Parser expects expressions but gets something else
**Error:** `Expected an expression`

**Investigation needed:**
- For each module, find exact line causing error
- Check for:
  - Incomplete expression parsing (missing cases)
  - Operator precedence issues
  - Token lookahead problems
  - Comment/whitespace handling
- Test: Create minimal reproducers for each

**Effort:** 4-6 hours analysis + fixes

---

### Priority 4: Type System - Definitions & Mismatches (3 modules)
**Modules:** `parser::struct_def`, `linear::modality`, `check::expr`
**Issues:**
- `struct_def`: Infinite size (recursive struct)
- `modality`: Type mismatch bool vs int
- `expr`: 40+ undefined types/variables

**Fixes:**
1. **Struct recursion:** Add Box wrapper
   ```sio
   struct Recursive {
     next: Box<Recursive>,  // Not: next: Recursive
   }
   ```

2. **Type mismatches:** Add explicit conversions
   ```sio
   fn bool_to_int(b: bool) -> i32 {
     if b { 1 } else { 0 }
   }
   ```

3. **Undefined types:** Find where used and declare
   - Search: `Undefined type: HirExpr`
   - Add: `struct HirExpr { ... }`

**Effort:** 3-5 hours

---

### Priority 5: Module Linking (8+ modules)
**Modules:** `check::stmt`, `check::expr`, `epistemic/*`, `effects/*`, etc.
**Issue:** Missing type/variable definitions from unimplemented imports
**Error:** 30+ undefined types per module

**Current state:**
- Stdlib modules exist but don't import from each other
- No module dependency resolution
- Type definitions scattered across files

**High-level approach:**
1. Define module boundaries (already in place)
2. Add `use` statements for cross-module types
3. Implement basic import resolution in type checker
4. Add missing type stubs where needed

**Example fix for `check::expr`:**
```sio
// Add at top of check/expr.sio:
use types::{Type, Kind}
use ast::{Expr, BinaryOp}

// Then reference types as: Type, Kind, etc.
```

**Effort:** 8-10 hours (largest task)

---

## Testing Strategy

After each fix:
```bash
# Test single module compilation
cargo run --bin souc -- check stdlib/compiler/parser/expr.sio

# Run self-compilation test
cargo test test_compile_stdlib_module_to_bytecode -- --nocapture

# Expected output: module compiles with X bytecode instructions
# compiled parser::expr (XXX instructions)
```

---

## Success Criteria

- [ ] All 5 Priority 1 modules work (currently 3/3 working)
- [ ] Add Priority 2 modules (0/2 working)
- [ ] Add Priority 3 modules (0/3 working)
- [ ] Add Priority 4 modules (0/3 working)
- [ ] Add Priority 5 modules (0/8+ working)
- **Goal:** 15+ modules compiling by end of sprint

---

## Notes for Implementers

1. **Reuse existing patterns** - Look at working modules for patterns
2. **Minimal changes** - Don't refactor, just fix the specific error
3. **Test after each fix** - Run the test immediately
4. **Document findings** - If error pattern is unusual, note it
5. **Use cheaper models** - haiku/claude works fine for these

---

## Module Status Summary

```
✅ Working (3):
  - parser::fn_def (675 instructions)
  - parser::item (362 instructions)
  - parser::impl_def (175 instructions)

❌ Blocked (30):
  Priority 1 (Complex assignment): 2 modules
  Priority 2 (Duplicate externs): 2 modules
  Priority 3 (Parse errors): 3 modules
  Priority 4 (Type system): 3 modules
  Priority 5 (Module linking): 8+ modules
  Other (unspecified): ~12 modules
```

Last updated: 2026-02-03
