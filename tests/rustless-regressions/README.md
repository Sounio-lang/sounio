# Rustless Regression Test Suite

This directory contains regression tests for the rustless cutover pipeline validation.

## Purpose

These tests validate the complete rustless execution pipeline:
1. Compile .sio source with Rust compiler → IR
2. Serialize IR using self-hosted serializer
3. Normalize serialized IR
4. Execute SOIR bytecode using self-hosted VM
5. Verify output matches expected

## Test Coverage

| Test | Feature Coverage | Expected Result |
|------|-----------------|-----------------|
| `01_fibonacci.sio` | Recursion, arithmetic, function calls | fib(7) = 13 |
| `02_arithmetic.sio` | Add, sub, mul, div, mod, comparisons | Exit code 0 |
| `03_control_flow.sio` | if/else, nested conditions, boolean logic | Exit code 0 |
| `04_functions.sio` | Parameter passing, return values, composition | Exit code 0 |
| `05_loops.sio` | while loops, iteration, accumulation | Exit code 0 |
| `06_arrays.sio` | Array indexing, literals, operations | Exit code 0 |
| `07_structs.sio` | Struct literals, field access, nested structs | Exit code 0 |
| `08_strings.sio` | String literals, basic operations | Exit code 0 |
| `09_integration.sio` | Combined features (comprehensive) | Exit code 0 |

## Running Tests

### Individual Tests
```bash
cargo test --test rustless_e2e rustless_e2e_fibonacci -- --nocapture
cargo test --test rustless_e2e rustless_e2e_arithmetic -- --nocapture
```

### Full Suite
```bash
cargo test --test rustless_e2e
```

### Via CI
```bash
.github/workflows/rustless-cutover.yml
```

## Test Requirements

All tests:
- Must be self-contained (no external dependencies)
- Must have predictable output
- Must complete in < 1 second
- Must handle edge cases gracefully

## Known Limitations

- **Recursion depth**: Limited by interpreter stack (fib tested to depth 7)
- **String operations**: Basic support only (full stdlib not yet available)
- **Module imports**: Not yet supported (all code must be in single file)
- **Pattern matching**: Not yet tested (enums require more infrastructure)

## Adding New Tests

1. Create `NN_testname.sio` file with:
   - Clear test description comment
   - Self-contained implementation
   - Predictable return value (0 = success, non-zero = specific failure)

2. Add test function to `crates/souc/tests/rustless_e2e.rs`:
   ```rust
   #[test]
   fn rustless_e2e_testname() {
       // Test implementation
   }
   ```

3. Update this README with test coverage

## Validation Matrix

```
┌─────────────┬────────┬────────┬─────────┬──────────┐
│ Test        │ Parse  │ IR Gen │ Execute │ Verify   │
├─────────────┼────────┼────────┼─────────┼──────────┤
│ Fibonacci   │   ✓    │   ✓    │    ✓    │    ✓     │
│ Arithmetic  │   ✓    │   ✓    │    ✓    │    ✓     │
│ Control     │   ✓    │   ✓    │    ✓    │    ✓     │
│ Functions   │   ✓    │   ✓    │    ✓    │    ✓     │
│ Loops       │   ✓    │   ✓    │    ✓    │    ✓     │
│ Arrays      │   ✓    │   ✓    │    ✓    │    ✓     │
│ Structs     │   ✓    │   ✓    │    ✓    │    ✓     │
│ Strings     │   ✓    │   ✓    │    ✓    │    ✓     │
│ Integration │   ✓    │   ✓    │    ✓    │    ✓     │
│ VM          │   ✓    │   ✓    │    ✓    │    ✓     │
└─────────────┴────────┴────────┴─────────┴──────────┘
```

## Success Criteria

- All tests pass (10/10)
- No regressions in existing functionality
- Performance within 10% of baseline
- CI pipeline green
