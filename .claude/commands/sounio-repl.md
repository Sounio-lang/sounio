# Start the Sounio REPL

Start the interactive Sounio Read-Eval-Print Loop for exploratory programming.

## Arguments
- `--jit` - Use JIT compilation for faster execution
- `--load <file>` - Load a file on startup
- `--no-color` - Disable ANSI color output

## Examples
- `/sounio-repl` - Start basic REPL
- `/sounio-repl --jit` - Start with JIT compilation
- `/sounio-repl --load examples/prelude.sio` - Load file on startup

$ARGUMENTS

Execute from the `compiler/` directory:

```bash
cd /home/demetrios/sounio-1/compiler && cargo run -- repl [--jit]
```

For JIT mode:
```bash
cd /home/demetrios/sounio-1/compiler && cargo run --features jit -- repl
```

## REPL Commands

Once in the REPL, these commands are available:

**General:**
- `:help` - Show help message
- `:quit` or `:q` - Exit REPL
- `:clear` - Clear screen
- `:reset` - Reset environment

**Inspection:**
- `:type <expr>` - Show type of expression
- `:info <name>` - Show info about a binding
- `:env` - Show current environment
- `:funcs` - List defined functions

**Epistemic:**
- `:confidence <expr>` - Show confidence level
- `:provenance <expr>` - Show provenance chain
- `:uncertainty <expr>` - Show uncertainty bounds

**Loading:**
- `:load <file>` - Load and evaluate a file
- `:reload` - Reload last loaded file

## REPL Features

- **Syntax highlighting** - Color-coded input
- **Multi-line input** - Use `{` to start multi-line blocks
- **Tab completion** - Complete names and keywords
- **History** - Arrow keys navigate history
- **Epistemic display** - Knowledge values show confidence/provenance

## Example Session

```
sounio> let x = 42
x: i32 = 42

sounio> let k: Knowledge<f64> = measure(3.14, uncertainty: 0.01)
k: Knowledge<f64> = Knowledge { value: 3.14, confidence: 0.95, ... }

sounio> :confidence k
0.95

sounio> :type x + 1
i32

sounio> fn square(n: i32) -> i32 { n * n }
square: fn(i32) -> i32

sounio> square(x)
1764
```
