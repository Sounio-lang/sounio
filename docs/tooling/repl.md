# Sounio REPL

The Sounio REPL (Read-Eval-Print Loop) provides an interactive shell for experimenting with Sounio code. It features rich visualization of epistemic values, making it ideal for scientific computing and data exploration.

## Starting the REPL

```bash
souc repl
```

You will see the welcome banner:

```
Sounio REPL v0.93.0
Epistemic computing at the horizon of certainty

Type :help for help, :quit to exit
Epistemic mode: ON - values show confidence badges

sio[0]>
```

## Basic Usage

### Expressions

Evaluate expressions directly:

```
sio[0]> 1 + 2 * 3
=> 7

sio[1]> "hello" ++ " world"
=> "hello world"
```

### Variables

Bind values with `let` or `var`:

```
sio[2]> let x = 42
x = 42

sio[3]> var y = 10
y = 10

sio[4]> x + y
=> 52
```

### Functions

Define functions (multiline input supported):

```
sio[5]> fn square(x: i64) -> i64 {
...         x * x
...     }
Defined function: square

sio[6]> square(5)
=> 25
```

### Types

Define custom types:

```
sio[7]> struct Point {
...         x: f64,
...         y: f64,
...     }
Defined struct: Point
```

## Epistemic Features

The REPL provides rich visualization for epistemic values.

### Confidence Badges

Values display with confidence indicators:

| Badge | Confidence Level |
|-------|-----------------|
| Green circle | High (>= 95%) |
| Yellow circle | Good (>= 80%) |
| Orange circle | Medium (>= 60%) |
| Red circle | Low (>= 30%) |
| Black circle | Very Low (< 30%) |

### Uncertain Values

Create and visualize uncertain values:

```
sio[8]> let u = uncertain(5.0, 0.3)
u = Yellow circle 5.0000 +/- 0.3000 [70.0%]

sio[9]> let v = uncertain(3.0, 0.1)
v = Green circle 3.0000 +/- 0.1000 [90.0%]

sio[10]> u + v
=> Yellow circle 8.0000 +/- 0.3162 [63.2%]
```

Notice how uncertainty propagates through operations.

### Knowledge Values

Work with full epistemic metadata:

```
sio[11]> let measurement = Knowledge::new(
...          value: 42.0,
...          uncertainty: 0.5,
...          confidence: 0.95,
...          source: "laboratory"
...      )
measurement = Green circle 42.0 [95.0%]
```

## REPL Commands

Commands start with `:` (colon).

### General Commands

| Command | Description |
|---------|-------------|
| `:help`, `:h`, `:?` | Show help |
| `:quit`, `:q`, `:exit` | Exit REPL |
| `:clear` | Clear all definitions and bindings |
| `:env` | Show current variable bindings |
| `:funcs` | Show defined functions |
| `:load <file>` | Load a Sounio source file |
| `:save <file>` | Save session to file |

### Inspection Commands

| Command | Description |
|---------|-------------|
| `:type <expr>` | Show type of expression |
| `:ast` | Toggle AST display |
| `:hir` | Toggle HIR display |
| `:types` | Toggle type display |
| `:jit` | Toggle JIT compilation |

### Epistemic Commands

| Command | Description |
|---------|-------------|
| `:epistemic` | Toggle epistemic display mode |
| `:provenance [var]` | Show provenance chain |
| `:uncertainty [var]` | Show uncertainty details |
| `:confidence [var]` | Show confidence levels |
| `:info <var>` | Show full epistemic info |

## Epistemic Exploration

### Viewing Provenance

Track the origin of values:

```
sio[12]> :provenance measurement
Provenance for: measurement
  Line: 11
  Source: Measurement via laboratory
  Revisability: Revisable if: new measurement, calibration update
```

### Uncertainty Analysis

Examine uncertainty in detail:

```
sio[13]> :uncertainty u
Uncertainty for: u
  Mean: 5.000000
  Std Dev: 0.300000
  95% CI: [4.412000, 5.588000]
  Confidence: 70.0%
  Bar: [==================  ] Yellow circle
```

### Confidence Overview

View confidence for all bindings:

```
sio[14]> :confidence
Confidence levels:
  x: Green circle [100.0%]
  y: Green circle [100.0%]
  u: Yellow circle [70.0%]
  v: Green circle [90.0%]
  measurement: Green circle [95.0%]
```

### Full Info

Get complete epistemic metadata:

```
sio[15]> :info measurement
Full epistemic info for: measurement

  Value:        Green circle 42.0 [95.0%]
  Expression:   Knowledge::new(value: 42.0, ...)
  Line:         11
  Source:       Measurement via laboratory
  Revisability: Revisable if: new measurement, calibration update
  Confidence:   [====================] (95.0%)
```

## JIT Mode

Enable JIT compilation for faster execution:

```
sio[16]> :jit
Use JIT: true

sio[17]> // Now expressions compile with Cranelift
```

Toggle back to interpreter:

```
sio[18]> :jit
Use JIT: false
```

Note: JIT requires the compiler to be built with `--features jit`.

## Loading Files

Load definitions from a file:

```
sio[19]> :load src/statistics.sio
Loading src/statistics.sio...
Defined function: mean
Defined function: std_dev
Defined function: correlation
```

## Saving Sessions

Save your work for later:

```
sio[20]> :save session.sio
Saved session to session.sio
```

The saved file contains all function and type definitions.

## Autocompletion

The REPL supports tab completion:

- **Commands**: Type `:` then Tab
- **Variables**: Start typing, then Tab
- **Keywords**: Start typing, then Tab
- **Files**: After `:load `, Tab completes filenames

## Multiline Input

For definitions spanning multiple lines, braces trigger continuation:

```
sio[21]> fn factorial(n: i64) -> i64 {
...         if n <= 1 {
...             1
...         } else {
...             n * factorial(n - 1)
...         }
...     }
Defined function: factorial
```

The prompt changes to `...` for continuation lines.

## Error Display

Errors display with context:

```
sio[22]> let x: i32 = "hello"
Parse error: unexpected end of input
   --> repl input
   |
 1 |     let x: i32 = "hello"
   |                  ^^^^^^^
   |
help: expected expression of type i32
```

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+C | Cancel current input |
| Ctrl+D | Exit REPL (same as `:quit`) |
| Up/Down | Navigate history |
| Ctrl+R | Reverse history search |
| Tab | Autocomplete |

## History

Command history is saved to `~/.sounio_history` and persists between sessions.

## Configuration

The REPL respects settings from `ReplConfig`:

```rust
ReplConfig {
    show_ast: false,      // Toggle with :ast
    show_hir: false,      // Toggle with :hir
    show_types: true,     // Toggle with :types
    use_jit: false,       // Toggle with :jit
    show_epistemic: true, // Toggle with :epistemic
    colored: true,        // Colored output
}
```

## Tips and Tricks

### Quick Type Checking

Use `:type` to check types without binding:

```
sio[23]> :type [1, 2, 3].map(|x| x * 2)
Type: Vec<i64>
```

### Debugging

Enable AST output to debug parsing:

```
sio[24]> :ast
Show AST: true

sio[25]> 1 + 2
--- AST ---
Binary { op: Add, left: Literal(1), right: Literal(2) }
=> 3
```

### Resetting State

Clear everything and start fresh:

```
sio[26]> :clear
Cleared all definitions and bindings.
```

## See Also

- [CLI Reference](cli.md) - Batch compilation
- [Language Server](lsp.md) - IDE integration
- [Getting Started](../getting-started/index.md) - Language tutorial
