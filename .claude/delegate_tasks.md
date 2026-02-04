# Self-Hosting Bootstrap: Delegate Tasks

**Status:** 24/33 stdlib modules compile. 9 modules blocked by remaining issues.
**Previous:** 8/33 (up from 3/34 at start of sprint)
**Target:** Enable all 33 stdlib modules to compile through bytecode codegen pipeline.

## Current Progress

### Completed: Effect Signature Annotations
Added `with Mut, Panic, Div` effect annotations to 16 modules that were blocking on undeclared effects:
- check/checker, check/context, check/env, check/types
- codegen/bytecode, codegen/vm
- lexer/comparison_harness, lexer/keywords, lexer/mod, lexer/scanner
- parser/ast, parser/expr, parser/mod, parser/stmt, parser/stmt_parse
- types/type

---

## Remaining Blockers

### Category 1: Type Mismatch Errors (5 modules)
**Modules:** `check/pattern`, `effects/effect`, `parser/expr_parse`, `parser/expr_simple`, `units/dimension`

**Root cause:** These modules call methods on placeholder types that don't exist.

Example from `check/pattern.sio`:
```sio
struct TypePool {
    count: i64,  // Placeholder only
}

// Code calls non-existent method:
ctx.type_pool.get_primitive_type("i64")
```

**Fix options:**
1. Add stub methods to placeholder types
2. Refactor code to not depend on unimplemented methods
3. Create proper type definitions

**Effort:** 2-4 hours per module

---

### Category 2: Resolution Errors - Missing Types (3 modules)
**Modules:** `check/stmt`, `effects/handler`, `effects/infer`

**Issue:** References to undefined types from other modules.

`check/stmt` missing:
- TypeContext, HirStmtKind, HirStmtKindLet, HirStmtKindAssign
- HirStmtKindExprStmt, HirStmtKindReturn, HirStmtKindWhile
- HirStmtKindBlock, HirStmt, HirExpr

`effects/handler` missing:
- Duplicate definition: Result
- Undefined: resolve_handler_impl, execute_computation, null, allocate_memory

`effects/infer` missing:
- TypeContext, CompileError, HirExpr, SourceLocation
- FunctionDef, FunctionSignature

**Fix options:**
1. Add forward declarations/stubs for missing types
2. Implement module imports (use statements)
3. Create shared types module

**Effort:** 4-6 hours

---

### Category 3: Parse Error (1 module)
**Module:** `epistemic/knowledge`
**Error:** `P0001: Expected [, found {`

**Investigation needed:** Check actual syntax causing parse failure.

**Effort:** 1-2 hours

---

## Module Status Summary

```
✅ Working (24):
  - check/checker, check/context, check/env, check/expr, check/types
  - codegen/bytecode, codegen/vm
  - lexer/comparison_harness, lexer/keywords, lexer/mod, lexer/scanner, lexer/tokens
  - linear/modality
  - parser/ast, parser/expr, parser/fn_def, parser/impl_def, parser/item
  - parser/mod, parser/stmt, parser/stmt_parse, parser/struct_def
  - types/type, types/unify

❌ Blocked (9):
  Type mismatch (5): check/pattern, effects/effect, parser/expr_parse,
                     parser/expr_simple, units/dimension
  Resolution (3):    check/stmt, effects/handler, effects/infer
  Parse error (1):   epistemic/knowledge
```

---

## Testing Commands

```bash
# Test single module compilation
target/debug/souc check stdlib/compiler/<module>.sio

# Count all passing modules
for f in stdlib/compiler/*/*.sio; do
  result=$(target/debug/souc check "$f" 2>&1)
  if echo "$result" | grep -q "All checks passed"; then
    echo "✅ $(basename $f .sio)"
  fi
done | wc -l

# Run self-compilation test
cargo test test_compile_stdlib_module_to_bytecode -- --nocapture
```

---

## Next Steps

1. **Quick wins:** Fix epistemic/knowledge parse error (1 module)
2. **Medium effort:** Add type stubs to fix resolution errors (3 modules)
3. **Larger effort:** Add method stubs or refactor type mismatch modules (5 modules)

**Goal:** 27+ modules compiling (add 3 more from resolution/parse fixes)

---

Last updated: 2026-02-04
