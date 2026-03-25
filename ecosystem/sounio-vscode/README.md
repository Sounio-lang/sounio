# Sounio for Visual Studio Code

Language support for [Sounio](https://github.com/sounio-lang/sounio) — the L0 systems + epistemic computing language for `.sio` files.

## Features

- Syntax highlighting for all Sounio keywords, operators, types, and effects
- Snippets for common patterns (`fn`, `struct`, `linear`, `match`, `know`, etc.)
- Real-time diagnostics via `souc check` on save
- Bracket matching and auto-closing pairs
- Code folding

## Requirements

A `souc` binary (Sounio compiler). Get it from the [Sounio repo](https://github.com/sounio-lang/sounio) at:

```
bin/souc
```

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `sounio.soucPath` | `""` | Path to the `souc` binary. Empty = auto-detect. |
| `sounio.checkOnSave` | `true` | Run `souc check` on file save. |
| `sounio.stdlibPath` | `""` | Path to stdlib directory (`SOUNIO_STDLIB_PATH`). |

### Example settings.json

```json
{
  "sounio.soucPath": "/home/user/sounio/bin/souc",
  "sounio.stdlibPath": "/home/user/sounio/stdlib"
}
```

## Snippets

| Prefix | Description |
|--------|-------------|
| `fn` | Function with effect annotation |
| `main` | `fn main() with IO {}` |
| `struct` | Struct definition |
| `linear` | Linear struct |
| `linearfull` | Linear struct with drop impl |
| `impl` | Impl block |
| `enum` | Enum definition |
| `trait` | Trait definition |
| `effect` | Algebraic effect definition |
| `handler` | Effect handler block |
| `match` | Match expression |
| `for` | For-in loop |
| `while` | While loop |
| `if` | If-else expression |
| `know` | `Knowledge<T>` epistemic measurement |
| `tyref` | Refinement type |
| `unit` | Unit of measure declaration |
| `module` | Module declaration |
| `import` | Import statement |

## Syntax Quick Reference

```sio
// Immutable / mutable bindings
let x = 5
var y: i32 = 10

// Functions with effects
fn f(x: i32) -> i32 with IO, Alloc { x + 1 }

// Linear types (must be consumed)
linear struct Handle { fd: i32 }

// Units of measure
unit mg = 0.001 * kg
let dose: mg = 500.0

// Epistemic measurements (GUM uncertainty)
let m: Knowledge<mg> = measure(500.0, uncertainty: 2.5)

// Exclusive reference (NOT &mut)
fn mutate(x: &!i32) with Mut { ... }

// Array concatenation
let arr2 = a ++ b

// Pipe operator
let result = value |> transform |> display
```

## Installation

Until the extension is published on the VS Code Marketplace, install from source:

```bash
cd triple-sounio-ecosystem/sounio-vscode
npm install
npm run compile
vsce package
code --install-extension sounio-vscode-0.1.0.vsix
```

## License

MIT
