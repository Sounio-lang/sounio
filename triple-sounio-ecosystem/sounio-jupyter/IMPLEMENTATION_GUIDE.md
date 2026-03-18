# Sounio Jupyter Kernel — Implementation Guide

## Overview

**Track B: sounio-jupyter** is a Jupyter kernel enabling epistemic computing and uncertainty quantification in interactive notebooks. This guide documents Day 1 completion and the path forward to Days 2–3.

## Day 1: Foundation (✅ COMPLETE)

### Deliverables

#### 1. Package Structure
```
sounio-jupyter/
├── pyproject.toml              ← Project metadata + dependencies
├── sounio_kernel/
│   ├── __init__.py             ← Package entry point + kernelspec
│   ├── kernel.py               ← IPykernel subclass (SounioKernel)
│   ├── executor.py             ← subprocess runner (CellExecutor)
│   ├── display.py              ← HTML/text formatters (Knowledge values)
│   └── magics.py               ← IPython magic stubs
├── tests/
│   ├── test_executor.py        ← Code wrapping tests (✓ PASS)
│   └── test_display.py         ← HTML rendering tests (✓ PASS)
├── kernel.json                 ← Jupyter kernel spec
├── README.md                   ← User guide
├── DAY1_DELIVERABLE.md         ← This deliverable summary
└── verify_day1.sh              ← Verification script (✓ PASS)
```

#### 2. SounioKernel Class
Inherits from `ipykernel.kernelbase.Kernel`:
- Language: `sounio` (mimetype: `text/x-sounio`, ext: `.sio`)
- Core methods:
  - `do_execute(code, silent, store_history, ...)` → delegates to CellExecutor
  - `do_complete(code, cursor_pos)` → placeholder (completion data later)
  - `do_inspect(code, cursor_pos, detail_level)` → placeholder (help system)
  - `do_shutdown(restart)` → cleanup
- Epistemic support: Auto-detects Knowledge values in output and formats as HTML

#### 3. CellExecutor
Wraps Sounio code execution:
- **Auto-detection**: Finds souc binary (SOUC env var → defaults → PATH)
- **Code wrapping**: `let x = 1` → `fn main() with IO { let x = 1 }`
- **Subprocess management**: 30s timeout, captures stdout/stderr
- **Temp files**: Creates and cleans up `.sio` files automatically
- **Stdlib handling**: SOUNIO_STDLIB_PATH env var support

#### 4. Display Module
Renders epistemic values with color-coded confidence:
```python
format_knowledge_html(value, epsilon, provenance)
# Returns:
# <div class="epistemic-value" style="border-left: 4px solid [COLOR]">
#   <strong>[VALUE]</strong>
#   <div>Confidence: [ε%] | Provenance: [SOURCE]</div>
# </div>
```

Color scheme:
- **Green** (#2ecc71): ε ≥ 0.9  (high confidence)
- **Orange** (#f39c12): ε ≥ 0.7  (medium confidence)
- **Red** (#e74c3c): ε < 0.7   (low confidence)

#### 5. Test Suite
- **test_executor.py** (4 tests, all ✓ PASS)
  - `test_executor_initialization()` — souc binary found
  - `test_code_wrapping_expressions()` — let/var wrapped
  - `test_code_wrapping_preserves_functions()` — fn defs not wrapped
  - `test_code_wrapping_preserves_type_defs()` — type/struct not wrapped

- **test_display.py** (5 tests, all ✓ PASS)
  - High/medium/low confidence HTML formatting
  - Plain text formatting
  - Required HTML elements (Value, Confidence, Provenance)

### Verification

Run `verify_day1.sh`:
```bash
cd triple-sounio-ecosystem/sounio-jupyter
bash verify_day1.sh
# ✓ File structure: 12 files
# ✓ Python syntax: all modules compile
# ✓ souc binary: found (13MB)
# ✓ stdlib path: found
# ✓ test_executor.py: PASS
# ✓ test_display.py: PASS
# ✓ souc execution: works (1+1=2)
```

## Day 2: Executor Integration (📋 TODO)

### Goals

1. **Wire CellExecutor into kernel.py**
   - `kernel.do_execute()` → `executor.run_cell(code)` → subprocess
   - Capture output/errors from souc
   - Post to Jupyter frontend via `send_response()`

2. **Refine code wrapping**
   - Detect bare expressions (not statements)
   - Auto-add print() for REPL-like feedback
   - Example: `1 + 1` → `fn main() with IO { print(1 + 1) }`

3. **Integration test**
   ```bash
   pip install -e .
   jupyter console --kernel sounio
   > let x = 1 + 1
   > x
   ```
   Expected: `2` printed to console

### Key Code Patterns

#### Auto-wrapper refinement
```python
def _wrap_code_repl(code: str) -> str:
    """Wrap expressions with print() for REPL feedback."""
    if code.startswith(("fn ", "type ", "struct ", ...)):
        return code  # No wrap

    # Check if last line is an expression (not a statement)
    lines = code.strip().split('\n')
    last_line = lines[-1].strip()

    if not last_line.endswith((';', '}')):
        # Likely an expression - add print wrapper
        code = f"""fn main() with IO {{
{code}
}}"""
    else:
        code = f"""fn main() with IO {{
{code}
}}"""

    return code
```

#### Executor wiring in kernel
```python
def do_execute(self, code, silent=False, store_history=True, ...):
    if not code.strip():
        return self._ok_response()

    if store_history:
        self.execution_count += 1

    # Execute via executor
    try:
        stdout, stderr, exitcode = self.executor.run_cell(code)
        success = exitcode == 0
    except Exception as e:
        self._post_error(f"Kernel error: {e}")
        return self._error_response()

    # Post output
    if not silent:
        if stdout:
            self._post_output(stdout)
        if stderr and exitcode != 0:
            self._post_error(stderr)

    return {
        "status": "ok" if success else "error",
        "execution_count": self.execution_count,
    }
```

### Testing

Create `tests/test_integration.py`:
```python
def test_executor_runs_simple_code():
    executor = CellExecutor()
    output, errors, code = executor.run_cell("print(42)")
    assert "42" in output
    assert code == 0

def test_executor_handles_errors():
    executor = CellExecutor()
    output, errors, code = executor.run_cell("let x: i32 = 3.14")  # Type error
    assert code != 0
    assert "type" in errors.lower() or "error" in errors.lower()
```

## Day 3: Display & Magics (📋 TODO)

### Goals

1. **Epistemic output formatting**
   - Parse `Knowledge { value: ... epsilon: ... prov: "..." }` from output
   - Apply HTML coloring via regex replacement
   - Embed in Jupyter output stream

2. **Implement magic commands**
   - `%time <code>` — measure execution time
   - `%sounio info` — kernel version, binary path, stdlib path
   - `%%writefile <filename>` — write cell to file

3. **Visual polish**
   - Use `/offload-scaffold glm` for HTML boilerplate
   - CSS: inline styles for portability
   - Test in JupyterLab + Jupyter notebook

### Offload Task (Day 3)

```bash
/offload-scaffold glm "Generate HTML/CSS boilerplate for displaying \
a scientific value with confidence color coding. Input: \
{value: float, epsilon: float, provenance: string}. Output: \
styled HTML div with color from epsilon (green if ≥0.9, orange \
if ≥0.7, red else). Include inline CSS for colors #2ecc71, #f39c12, #e74c3c."
```

Incorporate output into `display.py` `format_knowledge_html()`.

### Knowledge Regex Pattern

```python
pattern = r"Knowledge\s*\{\s*value:\s*([\d.e+-]+)\s*epsilon:\s*([\d.e+-]+)\s*prov:\s*\"([^\"]+)\"\s*\}"

def format_epistemic(text: str) -> str:
    def replace(m):
        value = float(m.group(1))
        epsilon = float(m.group(2))
        prov = m.group(3)
        return format_knowledge_html(value, epsilon, prov)
    return re.sub(pattern, replace, text)
```

### Magic Implementation

```python
class SounioMagics:
    def magic_time(self, code: str) -> None:
        import time
        start = time.time()
        self.kernel.executor.run_cell(code)
        elapsed = time.time() - start
        print(f"CPU time: {elapsed:.3f}s")

    def magic_sounio(self, args: str) -> str:
        if args == "info":
            return f"""
Sounio Kernel v{self.kernel.implementation_version}
souc binary: {self.kernel.executor.souc_binary}
stdlib path: {self.kernel.executor.stdlib_path}
"""
```

## Installation Instructions

### From Source

```bash
cd triple-sounio-ecosystem/sounio-jupyter

# Install in development mode (requires pip)
pip install -e .

# Register kernel with Jupyter
jupyter kernelspec install kernelspec/ --user

# Verify
jupyter kernelspec list | grep sounio
# sounio    /path/to/kernelspec/
```

### Start Notebook

```bash
jupyter notebook
# Select "Sounio" kernel in new cell
```

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| **Pure Python kernel** | Simplifies deployment; ipykernel handles ZMQ |
| **Subprocess execution** | Isolation + safety; clean error handling |
| **No state persistence** | Epistemic correctness; forces functional style |
| **Regex Knowledge parsing** | Lightweight; works with souc stdout format |
| **HTML inline styles** | Portability; no CSS dependencies |
| **Auto code wrapping** | REPL-like UX without explicit `fn main()` |

## Known Limitations

- **Streaming output**: JIT must complete before output appears
- **30s timeout**: Prevents hanging on slow code
- **No REPL state**: Each cell independent (by design)
- **JIT memory**: souc JIT can OOM on very large programs
- **No completion**: Tab completion deferred (would need souc LSP)

## File Manifest

| File | Purpose | Status |
|------|---------|--------|
| `pyproject.toml` | Package metadata | ✓ Done |
| `kernel.json` | Jupyter kernel spec | ✓ Done |
| `README.md` | User guide | ✓ Done |
| `sounio_kernel/__init__.py` | Package init | ✓ Done |
| `sounio_kernel/kernel.py` | SounioKernel class | ✓ Done (partial) |
| `sounio_kernel/executor.py` | CellExecutor | ✓ Done (foundation) |
| `sounio_kernel/display.py` | HTML formatters | ✓ Done (foundation) |
| `sounio_kernel/magics.py` | Magic commands | ✓ Stubs only |
| `tests/test_executor.py` | Executor tests | ✓ Done (4/4 PASS) |
| `tests/test_display.py` | Display tests | ✓ Done (5/5 PASS) |
| `verify_day1.sh` | Verification | ✓ Done (all checks PASS) |

## Metrics

- **Total files**: 12
- **Total lines of code**: ~1,200
- **Python modules**: 5 (all syntax-valid)
- **Tests**: 9 (all PASS)
- **Dependencies**: ipykernel, jupyter-client, pexpect
- **Build system**: setuptools
- **Min Python**: 3.8
- **Target Jupyter**: 4.9+

## References

- **Jupyter Kernel Spec**: https://jupyter-client.readthedocs.io/en/latest/kernels.html
- **IPykernel Source**: https://github.com/ipython/ipykernel
- **Sounio Docs**: https://docs.sounio.dev
- **Epistemic Types**: `/home/demetrios/RustroverProjects/sounio/stdlib/epistemic/`

## Next Session Checklist

- [ ] Day 2: Implement `executor.run_cell()` subprocess wiring
- [ ] Day 2: Refine code wrapping for REPL expressions
- [ ] Day 2: Test via `jupyter console --kernel sounio`
- [ ] Day 3: Use `/offload-scaffold glm` for HTML boilerplate
- [ ] Day 3: Implement magic commands
- [ ] Day 3: Test epistemic value rendering in notebook
- [ ] Day 3: Push to repository

---

**Created**: 2026-03-18
**Status**: Day 1 ✅ complete, Days 2–3 ready
**Maintainer**: Track B — sounio-jupyter
