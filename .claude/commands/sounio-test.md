# Run Sounio tests with filtering and category options

Run the Sounio compiler test suite with options for filtering by category, name pattern, or specific test file.

## Arguments
- `--category <category>` - Test category: integration, e2e, gpu, native, epistemic, jit, all (default: all)
- `--filter <pattern>` - Filter tests by name pattern
- `--nocapture` - Show test output (println, debug info)
- `--fail-fast` - Stop on first failure
- `[test_name]` - Specific test name to run

## Examples
- `/sounio-test` - Run all tests
- `/sounio-test --category integration` - Run integration tests only
- `/sounio-test --filter semantic_types` - Run tests matching pattern
- `/sounio-test --category gpu --nocapture` - GPU tests with output
- `/sounio-test epistemic_integration` - Run specific test

$ARGUMENTS

Execute from the `compiler/` directory:

1. Parse arguments to determine test scope

2. Category to test file mappings:
   - `integration` → `cargo test --test 'integration_*'`
   - `e2e` → `cargo test --test 'e2e_*'`
   - `gpu` → `cargo test --features gpu --test 'gpu_*'`
   - `native` → `cargo test --test 'native_*'`
   - `epistemic` → `cargo test --test 'epistemic_*'`
   - `jit` → `cargo test --features jit --test 'jit_*'`
   - `all` → `cargo test`

3. Construct the cargo test command:
   ```bash
   cd /home/demetrios/sounio-1/compiler && cargo test [--test TEST_FILE] [PATTERN] [-- --nocapture]
   ```

4. Add `-- --nocapture` if `--nocapture` flag is set

5. Report test results summary (passed/failed/ignored counts)
