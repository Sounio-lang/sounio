# sounio-jupyter Usage Guide

Complete guide to using sounio-jupyter for interactive Sounio development.

## Table of Contents

1. [Setup & Configuration](#setup--configuration)
2. [Basic Usage](#basic-usage)
3. [Writing Code](#writing-code)
4. [Magic Commands](#magic-commands)
5. [Rich Display](#rich-display)
6. [Tab Completion](#tab-completion)
7. [Advanced Patterns](#advanced-patterns)
8. [Performance Tips](#performance-tips)
9. [Troubleshooting](#troubleshooting)

---

## Setup & Configuration

### Installation

```bash
pip install sounio-jupyter
jupyter kernelspec list  # Verify "sounio" kernel appears
```

### Environment Variables

Set these before launching Jupyter:

```bash
# Required: path to souc binary
export SOUC=/path/to/souc-linux-x86_64-jit

# Required: path to Sounio stdlib
export SOUNIO_STDLIB_PATH=/path/to/stdlib

# Optional: temporary file directory
export TMPDIR=/tmp

# Start Jupyter
jupyter notebook
```

### Kernel Configuration

Edit `~/.local/share/jupyter/kernels/sounio/kernel.json` to customize:

```json
{
  "argv": ["python", "-m", "sounio_kernel", "-f", "{connection_file}"],
  "display_name": "Sounio",
  "language": "sounio",
  "metadata": {
    "souc_path": "/path/to/souc-linux-x86_64-jit",
    "stdlib_path": "/path/to/stdlib",
    "timeout": 30,
    "epistemic_support": true,
    "uncertainty_visualization": true,
    "provenance_tracking": true
  }
}
```

---

## Basic Usage

### Creating a Notebook

1. Launch Jupyter: `jupyter notebook`
2. Click **New** → Select **Sounio** kernel
3. Start writing code in the first cell

### Hello World

```sounio
print(42)
```

Press **Ctrl+Enter** to execute. Output:

```
42
```

### Variables & Expressions

```sounio
let x = 10
let y = 20
let z = x + y
print(z)
```

Output:

```
30
```

**Note:** Variables defined in one cell persist to the next cell (unlike traditional Jupyter).

### Comments

```sounio
// Single-line comment
let x = 5  // end-of-line comment

/*
  Multi-line comment
  spanning multiple lines
*/
```

---

## Writing Code

### Functions

Define functions across multiple cells or in a single cell:

=== "Single Cell (Recommended)"
    ```sounio
    fn square(n: i32) -> i32 {
        n * n
    }

    fn double(n: i32) -> i32 {
        n * 2
    }

    print(square(5))
    print(double(5))
    ```

=== "Across Cells"
    **Cell 1:**
    ```sounio
    fn square(n: i32) -> i32 {
        n * n
    }
    ```

    **Cell 2:**
    ```sounio
    square(5)
    ```

### Type Definitions

```sounio
type Medication = {
    name: i32,
    dose_mg: f64,
    half_life_hrs: f64
}

let aspirin: Medication = Medication {
    name: 1,
    dose_mg: 500.0,
    half_life_hrs: 2.3
}

print(aspirin.dose_mg)
```

### Struct Definitions

```sounio
struct Patient {
    id: i32,
    weight_kg: f64,
    age: i32
}

let p: Patient = Patient { id: 1, weight_kg: 70.0, age: 45 }
print(p.weight_kg)
```

### Pattern Matching

```sounio
match 42 {
    0 => "zero",
    1 => "one",
    n => "other"
}
```

Or with guards:

```sounio
match x {
    1 => "one",
    2 | 3 => "two or three",
    _ => "other"
}
```

### Conditional Expressions

```sounio
let x = 10
if x > 5 {
    print("x is large")
} else {
    print("x is small")
}
```

Or as an expression:

```sounio
let msg = if x > 5 { "large" } else { "small" }
print(msg)
```

### Loops

```sounio
// While loop
var i = 0
while i < 5 {
    print(i)
    i = i + 1
}
```

```sounio
// For loop (over range)
for i in 0..5 {
    print(i)
}
```

### Multi-line Code

Jupyter detects unclosed braces and continues the cell:

```sounio
fn factorial(n: i32) -> i32 {
    if n <= 1 { return 1 }
    n * factorial(n - 1)  # Auto-continues
}
# End function definition
print(factorial(5))  # Normal expression
```

---

## Magic Commands

Magic commands provide introspection and control. Prefix with `%` (line magic) or `%%` (cell magic).

### Type Checking: %check

Type-check code without executing:

```sounio
%check let x: i32 = "wrong type"
```

Output:

```
✗ Type error: expected i32, found string
```

Use to debug type issues quickly.

### Type Inference: %types

Show inferred types for all bindings:

```sounio
%types let x = 42
       let y = x + 3.5
```

Output:

```
x: i32
y: f64
```

### Abstract Syntax Tree: %ast

Display the AST for a code snippet:

```sounio
%ast 1 + 2 * 3
```

Output:

```
BinOp(
  Add,
  Literal(1),
  BinOp(
    Mul,
    Literal(2),
    Literal(3)
  )
)
```

Useful for understanding parsing.

### Timing: %time

Measure execution time of a single run:

```sounio
%time let result = expensive_fn()
```

Output:

```
Wall time: 0.0234s
CPU time: 0.0228s
```

### Benchmarking: %timeit

Measure average time over multiple runs:

```sounio
%timeit -n 100 fib(30)
```

Output:

```
100 loops: Min=0.0210s, Max=0.0245s, Mean=0.0227s ± 0.0012s
```

### Sounio Info: %sounio

Display configuration:

=== "Kernel info"
    ```sounio
    %sounio info
    ```

    Output:
    ```
    Sounio v1.0.0-beta.4
    souc: /usr/local/bin/souc-linux-x86_64-jit
    stdlib: /home/user/sounio/stdlib
    timeout: 30s
    ```

=== "Stdlib path"
    ```sounio
    %sounio stdlib
    ```

    Output:
    ```
    /home/user/sounio/stdlib
    ```

=== "Compiler path"
    ```sounio
    %sounio souc
    ```

    Output:
    ```
    /usr/local/bin/souc-linux-x86_64-jit
    ```

### Write to File: %%writefile

Save cell contents to a file:

```sounio
%%writefile my_program.sio
fn main() with IO {
    let x = 42
    print(x)
}
```

Creates `my_program.sio` in the current directory.

### Python Integration: %%python

(Future) Execute Python code in the same notebook:

```python
%%python
import sounio
x = sounio.Knowledge(100.0, 5.0)
print(x)
```

---

## Rich Display

### Knowledge Values

When you print a `Knowledge<T>` value, it renders as a rich HTML card:

```sounio
let measurement: Knowledge<mg> = measure(500.0, uncertainty: 2.5)
print(measurement)
```

**Display:**

```
┌──────────────────────────────┐
│ Measurement                  │ ← Green card (confidence 95%+)
│ Value: 500.0 mg              │
│ Uncertainty: ±2.5 mg (0.5%)   │
│ Confidence: 95%              │
│ Provenance: GUM measure      │
└──────────────────────────────┘
```

### Confidence Coloring

- 🟢 **Green** (≥90% confidence) — High-confidence measurement
- 🟡 **Orange** (70-90%) — Medium confidence
- 🔴 **Red** (<70%) — High uncertainty, caution needed

### Formatting Options

No special formatting needed — just `print()` your `Knowledge` values.

```sounio
let dose1: Knowledge<mg> = measure(500.0, uncertainty: 2.5)
let dose2: Knowledge<mg> = measure(600.0, uncertainty: 5.0)
let total = dose1 + dose2

print(dose1)   // Green card
print(dose2)   // Green card
print(total)   // Orange card (uncertainty doubled)
```

---

## Tab Completion

### Autocomplete

Press **Tab** while typing to trigger autocomplete:

| Category | Examples |
|----------|----------|
| Keywords | `let`, `fn`, `struct`, `type`, `match`, `if`, `for`, `while`, `import` |
| Built-ins | `print`, `print_f64`, `sqrt`, `measure`, `abs`, `sin`, `cos` |
| Variables | User-defined variables from previous cells |
| Functions | User-defined functions |
| Types | `i32`, `f64`, `Knowledge<T>`, custom types |

**Example:**

Type `p` + **Tab** → suggests `print`, `print_f64`

Type `K` + **Tab** → suggests `Knowledge`

### Shift+Tab Documentation

Press **Shift+Tab** on a symbol to see inline documentation:

```
let |  ← Press Shift+Tab
```

Shows:

```
Declare an immutable variable: let x = value
```

---

## Advanced Patterns

### Persistent State (Across Cells)

Variables from previous cells are available:

=== "Cell 1"
    ```sounio
    let x = 10
    let y = 20
    ```

=== "Cell 2"
    ```sounio
    // x and y are available here
    let z = x + y
    print(z)  // 30
    ```

**How it works:** The kernel extracts variable bindings from Cell 1 and injects them as arguments to `fn main()` in Cell 2.

### Defining Modules

```sounio
// my_math.sio
fn square(n: i32) -> i32 { n * n }
fn cube(n: i32) -> i32 { n * n * n }
```

Then use in notebook:

```sounio
// Currently: copy-paste or use %%writefile
// Future: import my_math::{square, cube}
```

### Epistemic Calculations

```sounio
// Define measurements
let dose: Knowledge<mg> = measure(500.0, uncertainty: 2.5)
let clearance: Knowledge<mL_min> = measure(120.0, uncertainty: 6.0)

// Automatic uncertainty propagation
let auc = dose / clearance
print(auc)  // Shows relative uncertainty
```

### Optimization Patterns

```sounio
// Define function once
fn compute_pk(dose: f64, kel: f64, t: f64) -> f64 {
    dose * exp(0.0 - kel * t)
}

// Use in subsequent cells
let result1 = compute_pk(500.0, 0.12, 4.0)
let result2 = compute_pk(500.0, 0.12, 8.0)
print(result1)
print(result2)
```

---

## Performance Tips

### 1. Batch-Compile Functions

Define all functions in one cell to avoid recompilation:

=== "Bad: Recompiles each cell"
    **Cell 1:**
    ```sounio
    fn f(x: i32) -> i32 { x * 2 }
    ```

    **Cell 2:**
    ```sounio
    fn g(x: i32) -> i32 { x + 1 }
    ```

    **Cell 3:**
    ```sounio
    f(g(5))  // Recompiles f, g, main
    ```

=== "Good: Define once"
    **Cell 1:**
    ```sounio
    fn f(x: i32) -> i32 { x * 2 }
    fn g(x: i32) -> i32 { x + 1 }
    ```

    **Cell 2:**
    ```sounio
    f(g(5))  // Only compiles main
    ```

### 2. Use %time for Profiling

```sounio
%time expensive_fn()
```

Identifies bottlenecks without manual timing.

### 3. JIT Warm-Up

First execution is slower (JIT compilation). Subsequent calls faster (cached).

### 4. Native Compilation

For production code, compile to ELF:

```bash
souc run --native my_program.sio my_program.elf
./my_program.elf
```

### 5. Limit Data Size

Large arrays/matrices: Keep under 100K elements per cell to avoid memory bloat.

---

## Troubleshooting

### Kernel Crashes

**Symptom:** "Kernel died" message

**Solution:**
1. Restart kernel: **Kernel** → **Restart**
2. Check for infinite loops: `while true { ... }`
3. Check for stack overflow: very deep recursion
4. Increase timeout in kernel config if computation is slow

### Incomplete Output

**Symptom:** Some print statements missing

**Reason:** Captured stdout is flushed per statement

**Solution:** Ensure each print has a newline (automatic in `print`)

### Completion Not Working

**Symptom:** Tab doesn't trigger autocomplete

**Solution:**
1. Restart kernel
2. Verify `sounio_kernel` package: `pip list | grep sounio`
3. Reload kernel: **Kernel** → **Restart**

### Type Errors Unclear

**Symptom:** Confusing error messages

**Solution:** Use `%check` to get detailed type information

```sounio
%check let x: i32 = some_expression
```

### Module Not Found

**Symptom:** "import stdlib::X failed"

**Solution:** Set `SOUNIO_STDLIB_PATH`

```bash
export SOUNIO_STDLIB_PATH=/path/to/stdlib
jupyter notebook
```

---

## Example: Complete Workflow

### Step 1: Define Types

```sounio
type Patient = {
    id: i32,
    weight_kg: f64,
    age: i32,
    creatinine: f64
}

type DrugPK = {
    clearance: f64,
    half_life: f64,
    volume: f64
}
```

### Step 2: Define Functions

```sounio
fn calculate_kel(half_life: f64) -> f64 {
    0.693 / half_life
}

fn calculate_auc(dose: f64, clearance: f64) -> f64 {
    dose / clearance
}

fn calculate_cmax_compartmental(dose: f64, volume: f64) -> f64 {
    dose / volume
}
```

### Step 3: Run Calculation

```sounio
let patient: Patient = Patient {
    id: 1,
    weight_kg: 70.0,
    age: 45,
    creatinine: 0.9
}

let drug: DrugPK = DrugPK {
    clearance: 120.0,
    half_life: 5.0,
    volume: 50.0
}

let dose = 500.0  // mg
let kel = calculate_kel(drug.half_life)
let auc = calculate_auc(dose, drug.clearance)
let cmax = calculate_cmax_compartmental(dose, drug.volume)

print(auc)
print(cmax)
```

### Step 4: Add Uncertainty

```sounio
let dose_epistemic: Knowledge<mg> = measure(dose, uncertainty: 2.5)
let clearance_epistemic: Knowledge<mL_min> = measure(drug.clearance, uncertainty: 6.0)

let auc_epistemic = dose_epistemic / clearance_epistemic
print(auc_epistemic)  // Shows with uncertainty
```

### Step 5: Benchmark

```sounio
%timeit -n 100 calculate_auc(500.0, 120.0)
```

---

## Tips & Tricks

### Rerun All Cells

**Kernel** → **Restart & Run All**

### View Variable History

All variables are logged. Access via:

```python
# In a Python cell (future)
import sounio_kernel
sounio_kernel.kernel.get_variable_history()
```

### Copy Cell Output

Right-click output → **Copy** (or use Jupyter export)

### Markdown Cells

Press **Esc** then **M** to convert cell to Markdown:

```markdown
# My Experiment

This is a heading.

## Methods

- Used Sounio 1.0
- Dataset: 50 patients
```

---

## Next Steps

- [**Overview**](overview.md) — Architecture and features
- [**Sounio Language Guide**](https://github.com/sounio-org/sounio/docs/LLM_PROGRAMMING_GUIDE.md) — Learn Sounio syntax
- [**sounio-py Guide**](../sounio-py/quickstart.md) — Python integration
- [**Examples**](https://github.com/sounio-org/sounio/tree/main/ecosystem/sounio-jupyter/examples) — Example notebooks
