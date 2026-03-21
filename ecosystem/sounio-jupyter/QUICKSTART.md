# Sounio Jupyter Kernel - Quick Start

## What is sounio-jupyter?

A **Jupyter kernel for the Sounio epistemic computing language** that brings interactive Sounio programming to notebooks. Features:

- 📝 Write Sounio code in notebook cells (auto-wrapped as needed)
- 🔍 Tab completion for keywords, functions, and variables
- 📊 Rich HTML display for Knowledge values with uncertainty visualization
- ⚙️ Magic commands for type-checking, benchmarking, and configuration
- 🎯 Full Sounio language support (types, effects, epistemic computing)

## Installation

### Prerequisites
- Python 3.8+
- Jupyter/JupyterLab installed
- Sounio compiler (`souc` binary) available

### Install Kernel

```bash
# From the sounio-jupyter directory
cd triple-sounio-ecosystem/sounio-jupyter

# Install package in development mode
pip install -e .

# Verify installation
python -m pytest tests/ -q
```

### Register with Jupyter

```bash
# Option 1: Automatic (via entry-point)
jupyter kernelspec list  # Should show "sounio" kernel

# Option 2: Manual
jupyter kernelspec install ./kernel_spec --user
```

### Configure Sounio Paths (if needed)

```bash
# Set environment variables before running Jupyter
export SOUC=/path/to/souc-linux-x86_64-jit
export SOUNIO_STDLIB_PATH=/path/to/stdlib
jupyter notebook
```

## First Notebook

1. Launch Jupyter:
```bash
jupyter notebook
```

2. Create new notebook → Select **Sounio** kernel

3. Write Sounio code (bare expressions auto-wrapped):

```sounio
// Simple arithmetic - auto-wrapped in fn main() with IO { ... }
1 + 1
```

4. Output:
```
2
```

## Common Examples

### Variables and expressions
```sounio
let x = 10
var y = 20
x + y * 2
```

### Functions
```sounio
fn square(n: i32) -> i32 {
    n * n
}

square(5)
```

### Epistemic types (Uncertainty)
```sounio
let dose: mg = 500.0
let uncertainty = 2.5
let k: Knowledge<mg> = measure(dose, uncertainty: uncertainty)
print(k)
```
→ Shows as **HTML card** with:
- Green/orange/red border (based on confidence)
- Value + uncertainty display
- Provenance/source information

### Type definitions
```sounio
type Dose = { value: f64 | value > 0.0 }
type DrugPK = {
    clearance: Dose,
    half_life: f64
}
```

### Pattern matching
```sounio
match 42 {
    0 => "zero",
    1 => "one",
    _ => "other"
}
```

## Magic Commands

Tab completion + interactive exploration:

### Type checking
```sounio
%check let x: i32 = "wrong type"
```
→ `✗ Type check failed: ...`

### Show AST
```sounio
%ast 1 + 2 * 3
```
→ Pretty-printed abstract syntax tree

### Show inferred types
```sounio
%types let x = measure(500.0, uncertainty: 2.5)
```
→ Display inferred types for all bindings

### Benchmarking
```sounio
%time let x = fib(30)
```
→ `CPU time: 0.1234s`

```sounio
%timeit -n 10 let x = fib(30)
```
→ Min/Max/Avg over 10 runs

### Kernel info
```sounio
%sounio info
```
→ Shows Sounio version, stdlib path, souc binary location

```sounio
%sounio stdlib
```

```sounio
%sounio souc
```

### Write to file
```sounio
%%writefile my_program.sio
fn main() with IO {
    let x = 42
    print(x)
}
```

## Tab Completion

Press **Ctrl+Tab** or **Tab** to autocomplete:

- **Keywords**: `let`, `fn`, `struct`, `match`, `if`, `Knowledge`, etc.
- **Built-in functions**: `print`, `sqrt`, `measure`, etc.
- **Variables**: Your declared variables (extracted from cell history)
- **Functions**: Your defined functions

## Shift+Tab Documentation

Press **Shift+Tab** on a symbol to see documentation:

```
| Symbol | Shows |
|--------|-------|
| let | "Declare an immutable variable: let x = 5" |
| fn | "Declare a function: fn name() { ... }" |
| Knowledge | "Epistemic type for uncertain values with provenance" |
| print | "Print value to stdout" |
```

## Multi-line Code

Jupyter detects incomplete code:

```sounio
fn add(a: i32, b: i32) -> i32 {
    # Press Enter here - Jupyter continues cell
    a + b
}
```

Auto-indent when detecting unclosed braces `{`, `(`, `[`

## Knowledge Value Display

When printing Knowledge values, the kernel formats them with rich HTML:

**Code:**
```sounio
let measurement: Knowledge<mg> = measure(500.0, uncertainty: 2.5)
print(measurement)
```

**Output:**
```
Knowledge { value: 500, epsilon: 2.5, prov: "GUM measurement" }
```

**Rendered as:**
- **Green card** (99% confidence)
- Value in large font
- ±2.5 uncertainty bar
- Provenance: "GUM measurement"

Confidence levels:
- 🟢 **≥90%** (High) → Green
- 🟡 **70-90%** (Medium) → Orange
- 🔴 **<70%** (Low) → Red

## Troubleshooting

### "souc binary not found"
```bash
export SOUC=$(which souc)  # or full path
export SOUNIO_STDLIB_PATH=/path/to/stdlib
jupyter notebook
```

### "ModuleNotFoundError: ipykernel"
```bash
pip install ipykernel jupyter-client
```

### Kernel dies/crashes
- Kernels are isolated subprocesses - crashes don't affect notebook server
- Restart kernel: Kernel menu → Restart
- Check `/tmp/sounio_kernel_*` temp files are cleaned up

### Code execution timeout (>30s)
- Default: 30 second timeout
- For heavy computation: use native compilation or increase timeout
- Modify in `sounio_kernel/executor.py`: `timeout=60`

### Completion not working
- Make sure `sounio_kernel` package is installed in current Python: `pip list | grep sounio`
- Restart kernel (Kernel → Restart)
- Restart Jupyter server

## Project Structure

```
sounio-jupyter/
├── sounio_kernel/
│   ├── kernel.py         # Main Jupyter kernel
│   ├── executor.py       # Runs souc binary
│   ├── display.py        # Knowledge formatting
│   ├── completion.py     # Tab completion
│   └── magics.py         # Magic commands
├── tests/                # 61 comprehensive tests
└── kernel.json           # Jupyter kernel spec
```

## Advanced Usage

### Persistent variables across cells

Currently, each cell is wrapped in a separate `fn main()`, so variables don't persist between cells. Future versions will support shared state.

**Workaround: put code in one cell**
```sounio
let x = 10
let y = 20
let z = x + y
print(z)
```

### Using external libraries

Sounio `import` statements work when stdlib is configured:

```sounio
import stdlib::math::sqrt
import stdlib::io::print

let x = sqrt(16.0)
print(x)
```

### Native compilation

For production use, compile to native ELF:

```bash
# Via souc command-line
souc run --native input.sio output.elf
```

## Performance Tips

1. **Use `%time` for quick timing**: `%time expensive_fn()`
2. **Use `%timeit` for benchmarking**: `%timeit -n 100 fn_call()`
3. **Type-check before running**: `%check your_code` (faster feedback loop)
4. **JIT compilation**: First run is slow, subsequent runs faster (cached)

## API Reference

### Kernel Configuration (kernel.json)
```json
{
  "display_name": "Sounio",
  "language": "sounio",
  "metadata": {
    "epistemic_support": true,
    "uncertainty_visualization": true,
    "provenance_tracking": true
  }
}
```

### Environment Variables
```bash
SOUC                        # Path to souc binary
SOUNIO_STDLIB_PATH          # Path to stdlib directory
PYTHONPATH                  # Python module search path
```

## Documentation

- **Language Guide**: [docs/LLM_PROGRAMMING_GUIDE.md](../../../docs/LLM_PROGRAMMING_GUIDE.md)
- **Minimum Viable Sounio**: [docs/MINIMUM_VIABLE_SOUNIO.md](../../../docs/MINIMUM_VIABLE_SOUNIO.md)
- **Implementation Details**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

## Support & Contributing

- **Report issues**: Create GitHub issue with notebook `.ipynb` + error
- **Request features**: Open GitHub discussion
- **Contributing**: PRs welcome! Run tests before submitting: `pytest tests/ -v`

## License

Apache 2.0 (same as Sounio project)

---

**Happy epistemic computing! 🚀**
