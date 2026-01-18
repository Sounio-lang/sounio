# Show intermediate representations

Display intermediate representations (IR) at various compilation stages for debugging and understanding.

## Arguments
- `<file>` - Sounio source file (required)
- `--stage <stage>` - IR stage: ast, hir, hlir, mir, sir (default: ast)
- `--format <text|json>` - Output format (default: text)
- `--output <file>` - Save IR to file instead of displaying

## Examples
- `/sounio-ir examples/hello.sio` - Show AST (default)
- `/sounio-ir examples/hello.sio --stage hir` - Show typed HIR
- `/sounio-ir examples/ode.sio --stage sir` - Show Scientific IR
- `/sounio-ir examples/calc.sio --stage mir --format json` - MIR as JSON

$ARGUMENTS

Execute from the `compiler/` directory:

1. Validate that a file path is provided

2. IR stage descriptions:
   - `ast` - Abstract Syntax Tree (parsed structure)
   - `hir` - High-level IR (typed, desugared)
   - `hlir` - High-level Low IR (SSA form)
   - `mir` - Mid-level IR (optimizable, lower-level)
   - `sir` - Scientific IR (domain-specific ops: ODEs, tensors, autodiff)

3. Construct the command based on stage:
   ```bash
   cd /home/demetrios/sounio-1/compiler && cargo run -- check <file> --show-ast
   ```

   For other stages, use appropriate compiler flags or emit commands:
   - `--emit-hir` for HIR
   - `--emit-hlir` for HLIR
   - `--emit-mir` for MIR
   - `--emit-sir` for SIR (requires scientific constructs)

4. Format output:
   - `text` - Pretty-printed, human-readable
   - `json` - Machine-readable JSON format

5. If `--output` specified, write to file instead of stdout

6. Include helpful annotations explaining key IR constructs
