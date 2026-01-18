# Run pre-commit quality checks

Run formatting, linting, and tests to ensure code quality before committing.

## Arguments
- `--quick` - Skip tests for faster check (format + lint only)
- `--fix` - Auto-fix formatting and lint issues where possible
- `--no-test` - Skip test execution

## Examples
- `/sounio-precommit` - Full quality check (format, lint, test)
- `/sounio-precommit --quick` - Fast check without tests
- `/sounio-precommit --fix` - Fix issues automatically

$ARGUMENTS

Execute from the `compiler/` directory:

1. Run checks in sequence, stopping on first failure:

2. **Step 1: Format check**
   ```bash
   cd /home/demetrios/sounio-1/compiler && cargo fmt --check
   ```
   If `--fix` is set, run `cargo fmt` instead to auto-fix

3. **Step 2: Lint check**
   ```bash
   cd /home/demetrios/sounio-1/compiler && cargo clippy -- -D warnings
   ```
   If `--fix` is set, add `--fix` flag

4. **Step 3: Test (unless --quick or --no-test)**
   ```bash
   cd /home/demetrios/sounio-1/compiler && cargo test
   ```

5. **Step 4: Build check**
   ```bash
   cd /home/demetrios/sounio-1/compiler && cargo build
   ```

6. Report overall pass/fail status with summary of any issues found

7. If all checks pass, indicate the code is ready to commit
