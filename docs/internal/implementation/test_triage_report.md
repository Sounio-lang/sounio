<!-- docs:meta
topic_id: repo.docs.internal.implementation.test-triage-report
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.implementation.test-triage-report
-->

# Test Triage Report - Phase C.1

**Date:** 2026-02-26  
**Total Ignored Tests Processed:** 49  
**Tests Un-Ignosed:** 4  
**Remaining Ignored:** 45

---

## Executive Summary

| Category | Count | Description |
|----------|-------|-------------|
| **READY** | 4 | Tests can be un-ignored - they now fail as expected |
| **NEEDS_FIX** | 4 | Tests fail but need error-pattern/description updates |
| **STILL_BLOCKED** | 41 | Tests remain ignored - require compiler enhancements |

**Tests Un-Ignored This Phase:** 4

---

## Category 1: READY - Tests to Un-Ignore

These tests now correctly fail with the expected errors. The compiler has implemented the necessary features.

### 1. `tests/ui/lexer/invalid_escape.sio`
- **Status:** ✓ Error correctly detected
- **Current Output:** `Error: unknown escape sequence: \q`
- **Action:** Remove `//@ ignore` line
- **Notes:** Escape sequence validation is working

### 2. `tests/ui/type/recursive_type.sio`
- **Status:** ✓ Error correctly detected
- **Current Output:** `error: Struct 'Node' has infinite size: field 'next' creates a cycle without indirection`
- **Action:** Remove `//@ ignore` line
- **Notes:** Type cycle detection for recursive structs is working

### 3. `tests/ui/type/ref_deref_mismatch.sio`
- **Status:** ✓ Error correctly detected
- **Current Output:** `error: Type mismatch: expected 'i32', found 'Ref { mutable: false, lifetime: None, inner: I64 }'`
- **Action:** Remove `//@ ignore` line
- **Notes:** Reference type checking is working

### 4. `tests/ui/type/loop_return_mismatch.sio`
- **Status:** ✓ Error correctly detected
- **Current Output:** `error: Type mismatch: expected I32, found Unit`
- **Action:** Remove `//@ ignore` line
- **Notes:** Loop return type checking is working (though error location needs improvement)

---

## Category 2: NEEDS_FIX - Tests with Wrong Error Patterns

These tests fail but need their error-pattern or description updated to match actual compiler output.

### 1. `tests/ui/pattern/non_exhaustive.sio`
- **Status:** ⚠ Fails but with wrong error
- **Expected:** Pattern exhaustiveness error
- **Actual:** Type mismatch error (test code structure issue)
- **Action:** Keep ignored, update test code or description

### 2. `tests/run-pass/closure_basic.sio`
- **Status:** ⚠ Marked as run-pass but fails
- **Issue:** Closure syntax parsing incomplete
- **Action:** Keep ignored, update description to clarify closure parser gap

### 3. `tests/run-pass/closure_effect_infer.sio`
- **Status:** ⚠ Parse error - expected expression
- **Issue:** `with` keyword not recognized (effect handler syntax)
- **Action:** Keep ignored, effect system not implemented

### 4. `tests/run-pass/handler_discharge.sio`
- **Status:** ⚠ Parse error - expected expression
- **Issue:** `with` keyword not recognized (effect handler syntax)
- **Action:** Keep ignored, effect handlers not implemented

---

## Category 3: STILL_BLOCKED - Requires Compiler Enhancements

### A. Units of Measure (6 tests)
All blocked - units module not implemented.

| Test | Current Error | Blocker |
|------|---------------|---------|
| `compile-fail/unit_cast_incompatible.sio` | Undefined type: mg, m | units module |
| `compile-fail/unit_mismatch.sio` | Import 'units' not found | units module |
| `run-pass/unit_cast_compatible.sio` | Undefined type: mg, kg | units module |
| `run-pass/unit_cast_time.sio` | Undefined type: s, ms, min | units module |
| `ui/type/unit_mismatch.sio` | Parse error (syntax not recognized) | units syntax |

### B. Effect System (4 tests)
All blocked - effect system not implemented.

| Test | Current Error | Blocker |
|------|---------------|---------|
| `ui/effect/alloc_not_declared.sio` | Undefined variable: box | effect primitives |
| `ui/effect/io_not_declared.sio` | Compiles but shouldn't | effect checking |
| `ui/effect/panic_not_declared.sio` | Compiles but shouldn't | effect checking |
| `run-pass/handler_discharge.sio` | Parse error | effect handler syntax |

### C. Linear/Affine Types (8 tests)
All blocked - ownership system not implemented.

| Test | Current Status | Blocker |
|------|----------------|---------|
| `compile-fail/affine_double_use.sio` | Compiles (should fail) | affine type checking |
| `compile-fail/linear_capture_closure.sio` | Compiles (should fail) | linear types |
| `compile-fail/linear_early_return.sio` | Compiles (should fail) | linear types |
| `compile-fail/linear_field_unconsumed.sio` | Compiles (should fail) | linear types |
| `compile-fail/linear_loop_consume.sio` | Compiles (should fail) | linear types |
| `compile-fail/linear_reassign_lost.sio` | Compiles (should fail) | linear types |
| `ui/ownership/affine_copy.sio` | Compiles (should fail) | affine type checking |
| `ui/ownership/linear_not_consumed.sio` | Compiles (should fail) | linear types |
| `ui/ownership/mutable_borrow_immut.sio` | Compiles (should fail) | borrow checking |
| `run-pass/affine_can_drop.sio` | Compiles (correctly) | test needs validation |

### D. Name Resolution (3 tests)
Blocked - enhanced name resolution needed.

| Test | Current Status | Blocker |
|------|----------------|---------|
| `ui/resolve/duplicate_param.sio` | Compiles (should fail) | duplicate detection |
| `ui/resolve/shadow_builtin.sio` | Compiles (should fail) | shadow detection |
| `ui/resolve/use_before_def.sio` | Compiles (should fail) | use-before-def check |

### E. Type System Enhancements (17 tests)
Blocked - various type checking features needed.

| Test | Current Status | Blocker |
|------|----------------|---------|
| `ui/type/array_elem_mismatch.sio` | Compiles (should fail) | array type checking |
| `ui/type/comparison_type_mismatch.sio` | Compiles (should fail) | comparison validation |
| `ui/type/condition_not_bool.sio` | Compiles (should fail) | condition validation |
| `ui/type/division_by_zero.sio` | Compiles (should fail) | const evaluation |
| `ui/type/extra_args.sio` | Compiles (should fail) | arg count checking |
| `ui/type/field_not_found.sio` | Compiles (should fail) | field validation |
| `ui/type/generic_constraint.sio` | Resolution error | generics system |
| `ui/type/if_branch_mismatch.sio` | Compiles (should fail) | branch type checking |
| `ui/type/invalid_binary_op.sio` | Compiles (should fail) | operator validation |
| `ui/type/invalid_unary_op.sio` | Compiles (should fail) | operator validation |
| `ui/type/logical_not_bool.sio` | Compiles (should fail) | logical op validation |
| `ui/type/match_arm_mismatch.sio` | Compiles (should fail) | match type checking |
| `ui/type/method_not_found.sio` | Compiles (should fail) | method resolution |
| `ui/type/mismatch_arg.sio` | Compiles (should fail) | arg type checking |
| `ui/type/not_callable.sio` | Compiles (should fail) | call validation |
| `ui/type/range_type_mismatch.sio` | Compiles (should fail) | range type checking |
| `ui/type/refinement_violation.sio` | Parse error | refinement types |
| `ui/type/struct_extra_field.sio` | Compiles (should fail) | struct validation |
| `ui/type/wrong_arg_count.sio` | Compiles (should fail) | arg count checking |

### F. Pattern Matching (1 test)
| Test | Current Status | Blocker |
|------|----------------|---------|
| `ui/pattern/non_exhaustive.sio` | Wrong error | exhaustiveness checking |

### G. Async/Closures (2 tests)
| Test | Current Status | Blocker |
|------|----------------|---------|
| `run-pass/async_channels.sio` | Resolution error | async/await |
| `run-pass/closure_effect_infer.sio` | Parse error | closure parser |

---

## Actions Taken

### Tests Un-Ignored (4)
1. ✅ `tests/ui/lexer/invalid_escape.sio` - Removed `//@ ignore`
2. ✅ `tests/ui/type/recursive_type.sio` - Removed `//@ ignore`
3. ✅ `tests/ui/type/ref_deref_mismatch.sio` - Removed `//@ ignore`
4. ✅ `tests/ui/type/loop_return_mismatch.sio` - Removed `//@ ignore`

### Tests Still Blocked (45)
All remaining tests keep their `//@ ignore` directive. Reasons summarized below:

| Reason | Count |
|--------|-------|
| Units of measure not implemented | 5 |
| Effect system not implemented | 4 |
| Linear/affine types not implemented | 8 |
| Enhanced name resolution needed | 3 |
| Type checking enhancements needed | 17 |
| Pattern exhaustiveness needed | 1 |
| Async/await not implemented | 2 |
| Closures incomplete | 3 |
| Error pattern needs updating | 2 |

---

## Test Counts by Category

### By Directory
| Directory | Ignored | Un-Ignored | Remaining |
|-----------|---------|------------|-----------|
| `compile-fail/` | 8 | 0 | 8 |
| `run-pass/` | 8 | 0 | 8 |
| `ui/effect/` | 3 | 0 | 3 |
| `ui/lexer/` | 1 | 1 | 0 |
| `ui/ownership/` | 3 | 0 | 3 |
| `ui/pattern/` | 1 | 0 | 1 |
| `ui/resolve/` | 3 | 0 | 3 |
| `ui/type/` | 22 | 3 | 19 |
| **TOTAL** | **49** | **4** | **45** |

### By Blocker Type
| Blocker | Count |
|---------|-------|
| Type system enhancements | 17 |
| Linear/affine types | 8 |
| Units of measure | 5 |
| Effect system | 4 |
| Name resolution | 3 |
| Async/await | 2 |
| Pattern exhaustiveness | 1 |
| Test/code fixes needed | 5 |

---

## Recommendations

### Immediate
1. ✅ Complete - 4 tests un-ignored and verified

### Short Term (Next Phase)
1. Fix test code in `non_exhaustive.sio` to properly test exhaustiveness
2. Investigate `closure_basic.sio` - closures may be closer to working than expected
3. Review `run-pass/affine_can_drop.sio` - might be testable now

### Medium Term
1. Implement function/field argument validation (17 type tests)
2. Implement name resolution enhancements (3 resolve tests)
3. Complete closure parser (2 closure tests)

### Long Term
1. Effect system implementation (4 tests)
2. Units of measure (5 tests)
3. Linear/affine type system (8 tests)
4. Pattern exhaustiveness checking (1 test)

---

## Verification

All 4 un-ignored tests were verified to:
1. Compile with the current `souc` compiler
2. Produce the expected error output
3. Match their `error-pattern` directive

Test command used:
```bash
./artifacts/diagnostic/20260223T084050Z/cargo-target/debug/souc check <test.sio>
```

---

*Report generated by Phase C.1 triage workflow*

---

## Appendix: Test Files Modified

The following test files were modified (removed `//@ ignore` line):

1. `tests/ui/lexer/invalid_escape.sio`
   - Removed: `//@ ignore`
   - Status: Now active - tests escape sequence validation

2. `tests/ui/type/recursive_type.sio`
   - Removed: `//@ ignore`
   - Status: Now active - tests infinite type detection

3. `tests/ui/type/ref_deref_mismatch.sio`
   - Removed: `//@ ignore`
   - Status: Now active - tests reference type checking

4. `tests/ui/type/loop_return_mismatch.sio`
   - Removed: `//@ ignore`
   - Status: Now active - tests loop return type checking

---

*End of Report*
