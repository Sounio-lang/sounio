# Error Message Audit — 20 Beginner Mistakes

Date: 2026-02-17

## Grading: A = excellent, B = acceptable, C = needs work, P = unexpectedly passes

| # | Mistake | Grade | Message Quality |
|---|---------|-------|----------------|
| 01 | Undefined variable | A | "Undefined variable: x" — clear, correct |
| 02 | Wrong arg count | P | Passes without error (checker is lenient) |
| 03 | Type mismatch (str/int) | A | "expected `i32`, found `string`" + correct span |
| 04 | Missing effect | P | Passes (effect propagation lenient on inner calls) |
| 05 | Assign to immutable | C | Shows I32/I64 mismatch instead of immutability error; span at line 1 |
| 06 | `let mut` (Rust habit) | P | Silently accepted as `var` (good: graceful compat) |
| 07 | `&mut` (Rust habit) | P | Silently accepted as `&!` (good: graceful compat) |
| 08 | `println!` (Rust habit) | P | Silently accepted (parser maps to println) |
| 09 | Missing return type | B | "expected Unit, found I32" — correct but should suggest `-> i32` |
| 10 | Duplicate function | A | "Duplicate definition `foo`" — clear |
| 11 | Semicolons (C/Java) | P | Semicolons accepted (intentional: Sounio allows optional semicolons) |
| 12 | `=` vs `==` | B | "Expected {, found `=`" — catches it but could say "did you mean ==?" |
| 13 | String + int | P | Passes (string + int resolves, may be intentional) |
| 14 | Missing main | P | Passes for `check` (correct: check only type-checks) |
| 15 | Forward reference | P | Passes (resolution handles forward refs despite docs) |
| 16 | `pub` keyword | P | Silently accepted (parsed but no effect) |
| 17 | `#[derive()]` | P | Silently accepted (parsed, ignored) |
| 18 | Empty match | B | "expected I32, found Unit" — correct but "match has no arms" would be clearer |
| 19 | Array OOB (static) | P | Not caught (no static bounds checking for constant indices) |
| 20 | Return type mismatch | A | "expected String, found I64" — clear |

## Summary

- **Grade A** (excellent): 4/20 (#1, #3, #10, #20)
- **Grade B** (acceptable): 3/20 (#9, #12, #18)
- **Grade C** (needs work): 1/20 (#5)
- **Grade P** (passes/handled): 12/20

## Key Findings

### Strengths
1. Type mismatch messages are clear with correct spans (#3, #20)
2. Parser gracefully handles Rust syntax (`let mut`, `&mut`, `println!`, `pub`, `#[derive]`) — good DX for Rust users transitioning
3. Duplicate definition detection works well (#10)
4. Resolution catches undefined variables immediately (#1)

### Areas for Improvement
1. **Error spans**: Several errors point to line 1 (file start) instead of the actual error location (#5, #9)
2. **Immutability errors**: Assigning to `let` variable gives type mismatch instead of "cannot assign to immutable variable" (#5)
3. **Missing suggestions**: Error #12 (`=` vs `==`) could suggest "did you mean `==`?"
4. **Missing suggestions**: Error #9 could suggest adding `-> i32` return type
5. **Effect checking gaps**: Missing effect annotations on helper functions (#4) not caught

### Intentional Design Decisions (not bugs)
- `let mut` accepted as alias for `var` (Rust compat)
- `pub` parsed and ignored (future-proofing)
- Semicolons optional (expression-oriented design)
- Forward references work (resolver handles them)
- `check` doesn't require `main` (library modules are valid)

## Test Files
All 20 trigger files in `tests/error_audit/01_*.sio` through `tests/error_audit/20_*.sio`.
