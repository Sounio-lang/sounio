# sounio-jupyter: Jupyter Kernel Overview

**sounio-jupyter** is a full Jupyter kernel for the Sounio epistemic computing language, bringing interactive notebook development to Sounio with rich display, tab completion, and integrated magic commands.

## What is sounio-jupyter?

A **Jupyter kernel** that:

- Runs Sounio code directly in notebook cells
- Auto-wraps expressions in `fn main() with IO { ... }` for convenience
- Provides tab completion for keywords, functions, and variables
- Renders `Knowledge` values with rich HTML uncertainty visualization
- Includes magic commands for type-checking, profiling, and inspection
- Maintains full Sounio language support (types, effects, epistemic computing)

**Unlike traditional Jupyter kernels for compiled languages**, sounio-jupyter understands epistemic values and displays uncertainty with confidence-color-coded cards.

## Architecture

```
┌────────────────────────────────┐
│     Jupyter Notebook           │
│  (browser + notebook server)   │
└────────────────┬───────────────┘
                 │ ZMQ protocol
┌────────────────▼───────────────┐
│   sounio-jupyter Kernel        │
│  (IPython + sounio_kernel)     │
├────────────────────────────────┤
│ • Parser (detect code type)    │
│ • SounioExecutor (JIT compile) │
│ • Display formatter (rich HTML)│
│ • Completion engine            │
│ • Magic command dispatcher     │
└────────────────┬───────────────┘
                 │ subprocess
┌────────────────▼───────────────┐
│   souc binary (Sounio compiler)│
│  (type-check, JIT, native)     │
└────────────────────────────────┘
```

## Key Features

### 1. Auto-Wrapping

Write bare expressions; they're automatically wrapped in `fn main() with IO`:

=== "Input (notebook cell)"
    ```sounio
    1 + 2 * 3
    ```

=== "Actual execution"
    ```sounio
    fn main() with IO {
        1 + 2 * 3
    }
    ```

=== "Output"
    ```
    7
    ```

### 2. Rich Knowledge Display

`Knowledge` values render as HTML cards with:

- **Color-coded confidence** (green ≥90%, orange 70-90%, red <70%)
- **Uncertainty band** (±ε visualization)
- **Provenance** (source label)
- **Relative uncertainty** (as percentage)

```sounio
let measurement: Knowledge<mg> = measure(500.0, uncertainty: 2.5)
print(measurement)
```

Displays as:

```
┌─────────────────────────────┐
│ Measurement                 │ ← green card (high confidence)
│ Value: 500.0 mg             │
│ Uncertainty: ±2.5 mg (0.5%)  │
│ Confidence: 95%             │
│ Provenance: GUM measure     │
└─────────────────────────────┘
```

### 3. Tab Completion

Press **Tab** to autocomplete:

- **Keywords**: `let`, `fn`, `struct`, `match`, `if`, `Knowledge`, etc.
- **Built-in functions**: `print`, `print_f64`, `sqrt`, `measure`, etc.
- **Variables**: Your declared variables (extracted from cell history)
- **Functions**: Your defined functions

### 4. Magic Commands

Special commands prefixed with `%` or `%%`:

| Magic | Purpose |
|-------|---------|
| `%check <code>` | Type-check code without running |
| `%types <code>` | Show inferred types |
| `%ast <code>` | Display abstract syntax tree |
| `%time <code>` | Time a single execution |
| `%timeit <code>` | Benchmark with multiple runs |
| `%sounio info` | Show Sounio version, paths |
| `%%writefile <file>` | Write cell to file |

### 5. Multi-line Code

Jupyter detects incomplete code and continues:

```sounio
fn add(a: i32, b: i32) -> i32 {
    # Press Enter - Jupyter continues
    a + b
}

# Back to top-level
add(5, 3)
```

### 6. Persistent State (Per Kernel)

Variables defined in one cell are available in the next (unlike IPython):

```sounio
// Cell 1
let x = 10

// Cell 2 (can use x from Cell 1)
let y = x + 20
print(y)  // 30
```

!!! note
    Each cell is wrapped independently in `fn main()`, so we automatically extract variable bindings and pass them forward. See [Usage Guide](usage.md) for details.

---

## Installation

### Prerequisites

- Python 3.8+
- Jupyter or JupyterLab
- Sounio compiler (`souc` binary)

### Install sounio-jupyter

```bash
pip install sounio-jupyter
```

Or from source:

```bash
cd ecosystem/sounio-jupyter
pip install -e .
```

### Verify Installation

```bash
# List installed kernels
jupyter kernelspec list

# Should output:
# sounio       /home/user/.local/share/jupyter/kernels/sounio
```

### Set Environment Variables (if needed)

```bash
# Path to souc binary
export SOUC=/path/to/souc-linux-x86_64-jit

# Path to Sounio stdlib
export SOUNIO_STDLIB_PATH=/path/to/stdlib

# Start Jupyter
jupyter notebook
```

---

## Getting Started

### 1. Launch Jupyter

```bash
jupyter notebook
```

Or with JupyterLab:

```bash
jupyter lab
```

### 2. Create New Notebook

Click **New** → **Notebook** → Select **Sounio** kernel

### 3. Write Code

```sounio
// Simple arithmetic
1 + 1
```

Press **Ctrl+Enter** to execute.

### 4. See Results

```
2
```

---

## Common Workflows

### Epistemic Computing

```sounio
// Define measurements with uncertainty
let dose: mg = 500.0
let dose_uncertainty = 2.5
let measurement: Knowledge<mg> = measure(dose, uncertainty: dose_uncertainty)

// Display
print(measurement)
```

### Function Definitions

```sounio
fn quadratic(x: f64, a: f64, b: f64, c: f64) -> f64 {
    a * x * x + b * x + c
}

// Call in next cell
quadratic(2.0, 1.0, 2.0, 3.0)
```

### Type Definitions

```sounio
type Patient = {
    weight: f64,
    age: i32,
    creatinine: f64
}

let p: Patient = Patient { weight: 70.0, age: 45, creatinine: 0.9 }
print(p.weight)
```

### Pattern Matching

```sounio
match 42 {
    0 => "zero",
    1 => "one",
    _ => "other"
}
```

### Modules & Imports

```sounio
import stdlib::math::sqrt

let x = sqrt(16.0)
print_f64(x)  // 4.0
```

---

## Command Reference

### Cell Commands

| Action | Keyboard |
|--------|----------|
| Execute cell | Ctrl+Enter |
| Execute & next | Shift+Enter |
| Create cell below | Alt+Enter |
| Edit cell | Double-click |
| Interrupt kernel | Kernel menu → Interrupt |
| Restart kernel | Kernel menu → Restart |

### Editor Commands

| Action | Keyboard |
|--------|----------|
| Autocomplete | Tab |
| Show docs | Shift+Tab (in help pane) |
| Comment line | Ctrl+/ |
| Indent line | Ctrl+] |
| Dedent line | Ctrl+[ |

---

## Troubleshooting

### Kernel not found

```bash
# Install again
pip install sounio-jupyter --force-reinstall

# Verify
jupyter kernelspec list
```

### "souc binary not found"

Set the environment variable:

```bash
export SOUC=/path/to/souc-linux-x86_64-jit
jupyter notebook
```

Or edit kernel configuration (`~/.local/share/jupyter/kernels/sounio/kernel.json`):

```json
{
  "argv": ["python", "-m", "sounio_kernel", "-f", "{connection_file}"],
  "display_name": "Sounio",
  "language": "sounio",
  "metadata": {
    "souc_path": "/path/to/souc-linux-x86_64-jit",
    "stdlib_path": "/path/to/stdlib"
  }
}
```

### Execution timeout

Default timeout: 30 seconds. For longer computations, increase in `sounio_kernel/executor.py`:

```python
def execute_request(self, code, store_history=True, ...):
    self.timeout = 60  # Increase to 60 seconds
```

Or restart kernel between cells to clear state.

### Completion not working

- Restart kernel: **Kernel** → **Restart**
- Check that `sounio_kernel` is installed: `pip list | grep sounio`
- Restart Jupyter server

---

## Project Structure

```
sounio-jupyter/
├── sounio_kernel/
│   ├── __init__.py           # Package exports
│   ├── __main__.py           # Entry point
│   ├── kernel.py             # Main Jupyter kernel class
│   ├── executor.py           # SounioExecutor (souc bridge)
│   ├── display.py            # Knowledge rendering (rich HTML)
│   ├── completion.py         # Tab completion engine
│   └── magics.py             # Magic command handlers
├── kernel_spec/
│   └── kernel.json           # Jupyter kernel specification
├── tests/
│   ├── test_kernel.py        # Kernel integration tests
│   ├── test_executor.py      # Executor tests
│   ├── test_display.py       # Display/rendering tests
│   ├── test_completion.py    # Completion tests
│   ├── test_magics.py        # Magic command tests
│   └── test_integration.py   # End-to-end tests
├── QUICKSTART.md             # Quick start guide
├── setup.py                  # Installation script
└── README.md                 # Project README
```

---

## Advanced Topics

### Persistent State Across Cells

Currently, each cell is wrapped in a separate `fn main()`, so variables **don't persist between cells by default**.

**Workaround:** Use multi-line cells:

```sounio
// Cell 1: Define everything together
let x = 10
let y = 20
let z = x + y

// Use all at once
print(z)
```

Future versions will support shared kernel state.

### Running with Python Cells

Mix Sounio and Python in the same notebook:

```python
# Python cell
import sounio
import numpy as np

measurements = [100.0, 110.0, 105.0]
errors = [5.0, 5.0, 5.0]
```

```sounio
// Sounio cell
// Call Python code via FFI (future)
```

### Profiling Sounio Code

Use the `%time` magic:

```sounio
%time let result = expensive_fn()
```

Output:

```
CPU time: 0.0234s
```

For detailed profiling:

```sounio
%timeit -n 10 expensive_fn()
```

Output:

```
Min: 0.0210s
Max: 0.0245s
Mean: 0.0227s ± 0.0012s
```

### Native Compilation

For production, compile to native ELF:

```bash
souc run --native my_program.sio output.elf
./output.elf
```

---

## Next Steps

- [**Usage Guide**](usage.md) — Detailed walkthrough with examples
- [**Magic Commands**](usage.md#magic-commands) — All available magics
- [**Sounio Language**](https://github.com/sounio-org/sounio/docs/LLM_PROGRAMMING_GUIDE.md) — Learn Sounio syntax
- [**Examples**](https://github.com/sounio-org/sounio/tree/main/ecosystem/sounio-jupyter/examples) — Example notebooks
