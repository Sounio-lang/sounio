# Lint Sounio source code

Lint Sounio source files for common issues, style violations, and potential bugs.

## Arguments
- `<path>` - File or directory to lint (default: current directory)
- `--fix` - Auto-fix issues where possible
- `--deny-warnings` - Treat warnings as errors
- `--format <text|json|sarif>` - Output format (default: text)

## Examples
- `/sounio-lint` - Lint all .sio files
- `/sounio-lint examples/` - Lint examples directory
- `/sounio-lint src/module.sio --fix` - Lint and auto-fix
- `/sounio-lint --deny-warnings` - Strict mode

$ARGUMENTS

Execute from the `compiler/` directory:

1. Determine target path (default to current directory if not specified)

2. For Sounio source files (.sio):
   ```bash
   cd /home/demetrios/sounio-1/compiler && cargo run -- lint <path> [--fix] [--deny-warnings]
   ```

3. For the Rust compiler code, run clippy:
   ```bash
   cd /home/demetrios/sounio-1/compiler && cargo clippy [--fix] [-- -D warnings]
   ```

4. Linting checks include:
   - Unused variables and imports
   - Unreachable code
   - Style violations (naming conventions)
   - Potential bugs (unhandled effects, ownership issues)
   - Deprecated patterns

5. Report results:
   - List of warnings/errors with file locations
   - Summary count of issues by category
   - Suggestions for fixes when available
