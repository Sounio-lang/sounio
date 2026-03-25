# Sounio for Claude Code

Language support for [Sounio](https://github.com/sounio-lang/sounio) in Claude Code.
Makes any Claude model write correct, idiomatic Sounio — no Rust-isms, proper effects, correct syntax.

## What You Get

- **CLAUDE.md** — Comprehensive syntax reference auto-loaded into every conversation
- **`/sounio-check`** — Type-check `.sio` files using the souc compiler
- **`/sounio-run`** — compile and run `.sio` files with the native wrapper
- **`/sounio-lint`** — Detect and fix Rust-isms (semicolons, `&mut`, `let mut`, macros)
- **Auto-validation hook** — Every `.sio` file is linted + type-checked on save

## Installation

### Quick (no compiler needed)

Copy the CLAUDE.md into your project root. Claude will write correct syntax but cannot type-check:

```bash
curl -sL https://raw.githubusercontent.com/sounio-lang/sounio/main/ecosystem/claude-code-sounio/CLAUDE.md > CLAUDE.md
```

### Standard (recommended)

Clone and run the installer:

```bash
git clone https://github.com/sounio-lang/sounio.git /tmp/sounio
bash /tmp/sounio/ecosystem/claude-code-sounio/install.sh /path/to/your/project
```

Then set the compiler path:

```bash
export SOUC_BIN=/path/to/souc-linux-x86_64-jit
```

### Full (inside the Sounio repo)

Already included. The souc binary is at `bin/souc`.

## Usage

Start Claude Code in your project:

```bash
claude
```

### Write Sounio code

Just ask Claude to write `.sio` files. The CLAUDE.md context ensures correct syntax:

```
Write a Sounio function that computes BMI with epistemic uncertainty tracking
```

### Slash commands

```
/sounio-check examples/hello.sio     # type-check
/sounio-run examples/hello.sio       # compile and run
/sounio-lint examples/hello.sio      # detect Rust-isms
```

### Auto-validation

When Claude writes or edits any `.sio` file, the post-tool-use hook automatically:
1. Lints for Rust-isms (semicolons, `&mut`, `let mut`, macros)
2. Runs `souc check` if the compiler is available
3. Reports errors inline so Claude can fix them

## What It Prevents

Claude will never write:

| Mistake | Sounio Fix |
|---------|-----------|
| `let x = 5;` | `let x = 5` (no semicolons) |
| `let mut x` | `var x` |
| `&mut x` | `&!x` |
| `println!("hi")` | `println("hi")` |
| `assert!(cond)` | `assert(cond)` |
| `#[test]` | `//@ run-pass` |
| `-42` | `0 - 42` |
| `Vec<T>` | `[T; N]` fixed arrays |
| `\|x\| x + 1` | named fn ref |
| Missing effects | `with IO, Mut, Panic, Div` |

## Requirements

- Claude Code 1.0.33+
- (Optional) `souc` compiler binary — [download from releases](https://github.com/sounio-lang/sounio/releases)

## License

MIT
