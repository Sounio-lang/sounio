# Sounio Jupyter Kernel — Quick Reference

## Installation

```bash
cd ecosystem/sounio-jupyter
pip install -e .
jupyter kernelspec install kernelspec/ --user
```

## Verification

```bash
bash verify_day1.sh           # All checks should pass
jupyter kernelspec list | grep sounio
```

## Usage

```bash
jupyter notebook              # Start notebook
# Select "Sounio" kernel in new cell
```

## Example Code (Day 2)

```sounio
let x = 1 + 1
let y = 42
let greeting = "Hello, Sounio!"
```

## File Guide

| File | Purpose | Status |
|------|---------|--------|
| `pyproject.toml` | Package config | ✓ Complete |
| `kernel.py` | SounioKernel class | ✓ Complete |
| `executor.py` | Code executor | ✓ Complete (stub) |
| `display.py` | HTML formatters | ✓ Complete |
| `magics.py` | Magic commands | ✓ Stubs |
| `tests/` | Test suite | ✓ 9/9 PASS |

## Key Classes

### SounioKernel
Base class: `ipykernel.kernelbase.Kernel`
- `do_execute(code, silent, store_history, ...)` — Execute cell
- `do_complete(code, cursor_pos)` — Tab completion
- `do_inspect(code, cursor_pos, detail_level)` — Introspection
- `do_shutdown(restart)` — Cleanup

### CellExecutor
- `run_cell(code)` → (stdout, stderr, exitcode)
- `_wrap_code(code)` → wrapped Sounio code
- `_find_souc_binary()` → auto-detect compiler
- `_find_stdlib_path()` → auto-detect stdlib

### Display Module
```python
format_knowledge_html(value, epsilon, provenance) → HTML string
format_knowledge_text(value, epsilon, provenance) → plain text
colorize_confidence(epsilon) → ANSI color code
```

## Color Scheme

```
Confidence (ε) | Color    | Hex Code
─────────────────────────────────────
    ≥ 0.9      | Green    | #2ecc71
    ≥ 0.7      | Orange   | #f39c12
    < 0.7      | Red      | #e74c3c
```

## Environment Variables

```bash
SOUC=<path>                    # souc binary (auto-detect if not set)
SOUNIO_STDLIB_PATH=<path>      # stdlib path (auto-detect if not set)
```

## Code Wrapping

Automatic wrapping (Day 2):
```
Input:  let x = 1 + 1
Output: fn main() with IO { let x = 1 + 1 }
```

Not wrapped:
```
fn my_func() { ... }    # Already a function
type MyInt = i32        # Type definition
struct Point { x: i32 } # Struct definition
```

## Testing

```bash
cd ecosystem/sounio-jupyter

# Run all tests
python3 tests/test_executor.py
python3 tests/test_display.py

# Verify installation
bash verify_day1.sh
```

## Troubleshooting

### souc binary not found
```bash
export SOUC=/path/to/souc-linux-x86_64-jit
```

### stdlib not found
```bash
export SOUNIO_STDLIB_PATH=/path/to/stdlib
```

### Python modules fail to import
```bash
pip install -e .
python3 -m py_compile sounio_kernel/*.py
```

## Documentation

- **README.md** — User guide + features
- **IMPLEMENTATION_GUIDE.md** — Technical architecture (Days 1-3)
- **DAY1_DELIVERABLE.md** — Day 1 completion report
- **QUICK_REFERENCE.md** — This file

## Next Steps

**Day 2**: Implement executor wiring
**Day 3**: Add display formatting + magics

See `IMPLEMENTATION_GUIDE.md` for detailed roadmap.

## References

- Jupyter Kernel Spec: https://jupyter-client.readthedocs.io/en/latest/kernels.html
- IPykernel: https://github.com/ipython/ipykernel
- Sounio Docs: https://docs.sounio.dev

---

Created: 2026-03-18 | Version: 0.1.0-day1
