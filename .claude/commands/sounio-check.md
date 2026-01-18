# Type-check Sounio source files

Type-check Sounio source files without compiling, with optional display of AST, types, and effects.

## Arguments
- `<file>` - Sounio source file to check (required)
- `--show-ast` - Display the parsed AST
- `--show-types` - Display inferred types
- `--show-effects` - Display inferred effects
- `--show-resolved` - Display resolved symbols
- `--skip-ownership` - Skip ownership/linearity checking

## Examples
- `/sounio-check examples/hello.sio` - Basic type check
- `/sounio-check examples/hello.sio --show-types` - Show inferred types
- `/sounio-check examples/epistemic.sio --show-ast --show-effects` - Show AST and effects
- `/sounio-check src/module.sio --show-resolved` - Show symbol resolution

$ARGUMENTS

Execute from the `compiler/` directory:

1. Validate that a file path is provided

2. Construct the check command with flags:
   ```bash
   cd /home/demetrios/sounio-1/compiler && cargo run -- check <file> [--show-ast] [--show-types] [--show-effects] [--show-resolved] [--skip-ownership]
   ```

3. Map arguments to compiler flags:
   - `--show-ast` → `--show-ast`
   - `--show-types` → `--show-types`
   - `--show-effects` → shows effect annotations
   - `--show-resolved` → shows name resolution results

4. Report results:
   - If successful: "Type check passed" with any requested displays
   - If errors: Show diagnostic messages with source locations

5. For Sounio-specific error codes, suggest using `/sounio-explain <code>` for details
