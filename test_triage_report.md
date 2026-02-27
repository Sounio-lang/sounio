# Test Triage Report - Phase C.1

**Date:** 2026-02-26  
**Total Ignored Tests:** 47

## Executive Summary

| Category | Count | Action Taken |
|----------|-------|--------------|
| READY (un-ignore now) | 0 | No tests un-ignored - all require verification with compiler |
| NEEDS_FIX (expectation updates) | 0 | Error patterns need compiler output verification |
| STILL_BLOCKED (keep ignored) | 47 | Updated with clear blocker annotations |

**Important Note:** The 9 tests mentioned as "NOW READY" in the Phase plan (`break_outside_loop.sio`, `continue_outside_loop.sio`, `tuple_index_oob.sio`, `struct_field_type.sio`, `struct_missing_field.sio`, `bitwise_float.sio`, `shift_float.sio`, `modulo_float.sio`, `closure_arg_type.sio`) were found to **already be active** (no `//@ ignore` annotation). These were likely enabled in recent commits (git log shows "Activate 9 UI type tests after diagnostic hardening").

---

## Actions Completed

### 1. Test Inventory
- Scanned all test files in `tests/` directory
- Identified 47 tests with `//@ ignore` annotation
- Analyzed each test's purpose and blocker

### 2. Blocker Annotations Updated
All 47 STILL_BLOCKED tests have been updated with clear blocker descriptions:

| Category | Tests | Blocker |
|----------|-------|---------|
| Linear/Affine | 8 | `BLOCKED - requires linear/affine type system` |
| Units | 5 | `BLOCKED - requires units of measure` |
| Effects | 4 | `BLOCKED - requires effect system` |
| Generics | 1 | `BLOCKED - requires generics` |
| Refinement | 1 | `BLOCKED - requires refinement types` |
| Pattern | 1 | `BLOCKED - requires pattern exhaustiveness checking` |
| Async | 1 | `BLOCKED - requires async/await` |
| Type Unification | 6 | `BLOCKED - requires enhanced type unification` |
| Operator Validation | 2 | `BLOCKED - requires operator type validation` |
| Condition/Logical | 2 | `BLOCKED - requires condition/logical op validation (NEAR READY)` |
| Function/Field | 6 | `BLOCKED - requires function/field validation (NEAR READY)` |
| Cycle Detection | 1 | `BLOCKED - requires type cycle detection` |
| Const Eval | 1 | `BLOCKED - requires compile-time constant evaluation` |
| Method Resolution | 1 | `BLOCKED - requires method resolution` |
| References | 1 | `BLOCKED - requires reference type checking` |
| Name Resolution | 3 | `BLOCKED - requires name resolution enhancements` |
| Lexer | 1 | `BLOCKED - requires escape sequence validation` |
| Borrow | 1 | `BLOCKED - requires borrow checking` |

### 3. Categorization by Feature Gap

#### High Priority (Core Language)
- **Linear/Affine Types:** 12 tests
- **Effect System:** 4 tests  
- **Generics:** 1 test

#### Medium Priority
- **Units of Measure:** 5 tests
- **Async/Await:** 1 test
- **Pattern Exhaustiveness:** 1 test

#### Lower Priority (Advanced)
- **Refinement Types:** 1 test

---

## Tests Marked as "NEAR READY"

These 10 tests may be close to passing and should be prioritized for enablement:

1. `tests/ui/type/condition_not_bool.sio`
2. `tests/ui/type/logical_not_bool.sio`
3. `tests/ui/type/wrong_arg_count.sio`
4. `tests/ui/type/extra_args.sio`
5. `tests/ui/type/mismatch_arg.sio`
6. `tests/ui/type/not_callable.sio`
7. `tests/ui/type/field_not_found.sio`
8. `tests/ui/type/struct_extra_field.sio`
9. `tests/ui/type/invalid_binary_op.sio`
10. `tests/ui/type/invalid_unary_op.sio`

---

## Complete List of Ignored Tests

### Linear/Affine Types (12 tests)
```
tests/compile-fail/affine_double_use.sio
 tests/compile-fail/linear_capture_closure.sio
tests/compile-fail/linear_early_return.sio
tests/compile-fail/linear_field_unconsumed.sio
tests/compile-fail/linear_loop_consume.sio
tests/compile-fail/linear_reassign_lost.sio
tests/ui/ownership/affine_copy.sio
tests/ui/ownership/linear_not_consumed.sio
tests/ui/ownership/mutable_borrow_immut.sio
tests/run-pass/affine_can_drop.sio
```

### Units of Measure (5 tests)
```
tests/compile-fail/unit_cast_incompatible.sio
tests/compile-fail/unit_mismatch.sio
tests/ui/type/unit_mismatch.sio
tests/run-pass/unit_cast_compatible.sio
tests/run-pass/unit_cast_time.sio
```

### Effect System (4 tests)
```
tests/ui/effect/alloc_not_declared.sio
tests/ui/effect/io_not_declared.sio
tests/ui/effect/panic_not_declared.sio
tests/run-pass/handler_discharge.sio
```

### Type System - Complex (17 tests)
```
tests/ui/type/array_elem_mismatch.sio
tests/ui/type/comparison_type_mismatch.sio
tests/ui/type/condition_not_bool.sio
tests/ui/type/division_by_zero.sio
tests/ui/type/extra_args.sio
tests/ui/type/field_not_found.sio
tests/ui/type/generic_constraint.sio
tests/ui/type/if_branch_mismatch.sio
tests/ui/type/invalid_binary_op.sio
tests/ui/type/invalid_unary_op.sio
tests/ui/type/logical_not_bool.sio
tests/ui/type/loop_return_mismatch.sio
tests/ui/type/match_arm_mismatch.sio
tests/ui/type/method_not_found.sio
tests/ui/type/mismatch_arg.sio
tests/ui/type/not_callable.sio
tests/ui/type/range_type_mismatch.sio
tests/ui/type/recursive_type.sio
tests/ui/type/ref_deref_mismatch.sio
tests/ui/type/refinement_violation.sio
tests/ui/type/struct_extra_field.sio
tests/ui/type/wrong_arg_count.sio
```

### Resolution (3 tests)
```
tests/ui/resolve/duplicate_param.sio
tests/ui/resolve/shadow_builtin.sio
tests/ui/resolve/use_before_def.sio
```

### Other (3 tests)
```
tests/ui/lexer/invalid_escape.sio
tests/ui/pattern/non_exhaustive.sio
tests/run-pass/async_channels.sio
```

---

## Recommendations for Next Phase

### Immediate (Phase C.2)
1. Build compiler and test the 10 "NEAR READY" tests
2. Un-ignore any that pass or update error patterns for those that fail with expected errors
3. Create issues for tests that should pass but don't

### Short-term (Phase D)
1. Implement remaining basic type validation:
   - Condition type checking
   - Argument count validation
   - Field access validation
   - Basic operator validation

### Medium-term
1. Effect system completion
2. Pattern exhaustiveness checking
3. Units of measure type checking

### Long-term
1. Linear/affine type system (large feature)
2. Generics (large feature)
3. Refinement types (advanced feature)
4. Async/await (language extension)

---

## Verification Command

To verify any test manually:
```bash
./souc check tests/ui/type/<test_name>.sio
```

To run the full test suite (when compiler is available):
```bash
bash scripts/run_sio_test_suite.sh --verbose
```

---

*Report generated: 2026-02-26*  
*Phase C.1 Triage Complete*  
*Tests processed: 47*  
*Blocker annotations updated: 47*
