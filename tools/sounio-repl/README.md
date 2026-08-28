# sounio-repl

Interactive REPL (Read-Eval-Print Loop) for the Sounio language.

## Usage

From the repo root:

```bash
bin/souc repl
```

Or run the binary directly:

```bash
cd tools/sounio-repl
cargo run --release
```

## Features

- **Expression evaluation**: Type any expression and see the result.
- **Declaration accumulation**: `let`, `var`, `fn`, `struct`, and `use` declarations persist across evaluations.
- **Session persistence**: Your session is automatically saved to `~/.local/share/sounio-repl/session.sio` and restored on the next startup.
- **Multi-line input**: Automatically detects unbalanced braces/parens and waits for more input. Use trailing `\` for explicit continuation.
- **Readline support**: Arrow keys, history navigation, Emacs-style editing, and tab completion for commands.
- **Timing**: `:time` toggles per-evaluation compilation/execution timing.

## Commands

| Command | Description |
|---------|-------------|
| `:quit`, `:q` | Exit the REPL |
| `:reset` | Clear the session |
| `:show` | Show current session declarations |
| `:load <file.sio>` | Load a source file into the session |
| `:save [file]` | Save session to a file (default: `repl_session.sio`) |
| `:time` | Toggle compilation/evaluation timing |
| `:help`, `:h` | Show help |

## How it works

The REPL is a Rust driver that manages session state and orchestrates compilation using the native Sounio compiler (`souc-linux-x86_64`).

- Declarations are validated by compiling them with a dummy `main()` function.
- Expressions are wrapped in a `main()` function, compiled to a temporary ELF, executed, and stdout is captured.
- The REPL filters verbose compiler output to show only errors and warnings.
- All temp files are cleaned up immediately after each evaluation.

## Building

```bash
cd tools/sounio-repl
cargo build --release
```

The release binary will be at `target/release/sounio-repl`.
