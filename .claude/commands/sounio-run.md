# Run Sounio programs

Run Sounio programs using the interpreter or JIT compiler.

## Arguments
- `<file>` - Sounio source file to run (required)
- `--jit` - Use JIT compilation instead of interpreter
- `--optimize` - Enable optimizations (JIT mode)
- `--args <...>` - Arguments to pass to the program
- `--time` - Show execution time

## Examples
- `/sounio-run examples/hello.sio` - Run with interpreter
- `/sounio-run examples/fibonacci.sio --jit` - Run with JIT
- `/sounio-run examples/cli.sio --args "arg1 arg2"` - Pass arguments
- `/sounio-run examples/bench.sio --jit --optimize --time` - Optimized JIT with timing

$ARGUMENTS

Execute from the `compiler/` directory:

1. Validate that a file path is provided

2. For interpreter mode (default):
   ```bash
   cd /home/demetrios/sounio-1/compiler && cargo run -- run <file> [-- <args>]
   ```

3. For JIT mode (`--jit`):
   ```bash
   cd /home/demetrios/sounio-1/compiler && cargo run --features jit -- jit <file> [--optimize] [-- <args>]
   ```

4. Handle program arguments:
   - Pass `--args` content after `--` separator to the Sounio program

5. If `--time` is set, wrap execution to measure and report timing

6. Report:
   - Program output
   - Exit status
   - Execution time (if requested)
   - Any runtime errors with stack traces
