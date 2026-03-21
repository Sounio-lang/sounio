# Triple Sounio Ecosystem — Integration Notes

**For:** Track D Lead (Days 11-14) and all contributors
**Purpose:** Critical handoff documentation for cross-track dependencies
**Version:** 1.0 (Day 1 prep)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Triple Sounio Ecosystem                      │
├─────────────────────┬──────────────────┬───────────────────────┤
│    Track A          │    Track B       │     Track C           │
│   sounio-py         │ sounio-jupyter   │  drug-discovery       │
│   (PyO3 FFI)        │  (Jupyter kernel)│  (Sounio pipeline)    │
└─────────────────────┴──────────────────┴───────────────────────┘
         ▲                       ▲                    ▲
         └───────────────────────┴────────────────────┘
               Canonical Format: Knowledge { value, epsilon, prov }
```

### Critical Dependencies

| Dependency | Source → Target | Required by | Impact if Missing |
|------------|-----------------|------------|------------------|
| `_executor.py` | Track A → B | Jupyter cell execution | Test 3 fails (kernel can't run cells) |
| `_executor.py` | Track A → C | Via sounio.run_file() | Test 2 fails (pipeline can't execute) |
| `souc` binary path | Track C → A/B | Environment variable | Tests fail silently if path wrong |
| `SOUNIO_STDLIB_PATH` | Track C → A/B | Full pipeline execution | Type checking fails for `use` imports |
| Canonical format | A/B/C → D | Parsing regex in demo.py | Results unparseable; integration fails |

---

## Environment Variables

All three projects depend on these **two critical env vars**. Set them before running:

### `SOUC` — Path to Sounio Compiler Binary

**Default location:**
```
/home/demetrios/RustroverProjects/sounio/artifacts/omega/souc-bin/souc-linux-x86_64-jit
```

**How to set:**
```bash
# In your shell profile (~/.bashrc, ~/.zshrc)
export SOUC=/home/demetrios/RustroverProjects/sounio/artifacts/omega/souc-bin/souc-linux-x86_64-jit

# Or set at runtime
SOUC=/path/to/souc python demo.py
```

**Verification:**
```bash
$SOUC --version
# Should output: souc version 1.0.0-beta.4 or similar
```

### `SOUNIO_STDLIB_PATH` — Path to Standard Library

**Default location:**
```
/home/demetrios/RustroverProjects/sounio/stdlib
```

**How to set:**
```bash
# In your shell profile
export SOUNIO_STDLIB_PATH=/home/demetrios/RustroverProjects/sounio/stdlib

# Or at runtime
SOUNIO_STDLIB_PATH=/path/to/stdlib python demo.py
```

**Verification:**
```bash
ls $SOUNIO_STDLIB_PATH
# Should show: epistemic/, ode/, nn/, units/, ... etc
```

---

## Canonical Output Format

**All three projects MUST output Knowledge values in this exact format:**

```
Knowledge { value: 42.000 epsilon: 0.850 prov: "source_name" }
```

### Format Specification

| Component | Type | Example | Notes |
|-----------|------|---------|-------|
| `value` | float (decimal) | `42.000` or `1.23e-4` | Can use scientific notation |
| `epsilon` | float (decimal) | `0.850` or `1.5e-2` | Uncertainty value [0, 1] |
| `prov` | string (quoted) | `"screening"` or `"pk_fit"` | Identifies source stage |
| Whitespace | fixed | ` ` (single space) | Exact format for regex match |

### Regex for Parsing (in Python)

**Used in:** `sounio/_executor.py` and `demo.py`

```python
import re

pattern = r"Knowledge \{ value: ([\d.e+-]+) epsilon: ([\d.e+-]+) prov: \"([^\"]+)\" \}"
text = 'Knowledge { value: 42.000 epsilon: 0.850 prov: "screening" }'
match = re.search(pattern, text)

if match:
    value, epsilon, prov = match.groups()
    print(f"Parsed: {float(value)}, {float(epsilon)}, {prov}")
```

### Implementation Checklist

- [ ] **Track A (sounio-py):** `knowledge.rs` `Display` trait outputs canonical format
- [ ] **Track B (sounio-jupyter):** `display.py` formats Knowledge for HTML with this string
- [ ] **Track C (drug-discovery):** `types.sio` `fn print_knowledge()` outputs this format
- [ ] All three: **NO EXTRA WHITESPACE** before/after braces or values

---

## Track A: sounio-py Handoffs

### What Track A Delivers

| File | Purpose | Used by |
|------|---------|---------|
| `sounio/__init__.py` | Module entry point, imports native extension | B, C (Python scripts) |
| `sounio/_executor.py` | Subprocess wrapper for `souc run <tmpfile>` | B (Jupyter), C (demo.py) |
| `sounio/knowledge.py` | Knowledge class (Rust-backed or pure Python) | B, C (type annotations) |
| `sounio/numpy_bridge.py` | EpistemicArray wrapper | B (display), C (optional) |

### Key Functions Track B/C Depend On

**`_executor.py.run(code: str) -> ExecutionResult`**
```python
def run(code: str) -> ExecutionResult:
    """Execute Sounio code via `souc run <tmpfile>`

    Args:
        code: Sounio source code (auto-wrapped if no fn)

    Returns:
        ExecutionResult with .stdout, .stderr, .returncode

    Raises:
        TimeoutError if souc run exceeds 30s
    """
```

**`_executor.py.run_file(path: str) -> ExecutionResult`**
```python
def run_file(path: str) -> ExecutionResult:
    """Execute Sounio file directly

    Args:
        path: Path to .sio file

    Returns:
        ExecutionResult with .stdout, .stderr, .returncode

    Raises:
        FileNotFoundError if path doesn't exist
    """
```

**`ExecutionResult` structure:**
```python
class ExecutionResult:
    stdout: str     # Full stdout from souc run
    stderr: str     # Full stderr from souc run
    returncode: int # Exit code
```

### Dependency Chain

```
Track A Build
    ↓
maturin develop → sounio._sounio_native module
    ↓
sounio/__init__.py imports native module
    ↓
sounio/_executor.py available → Track B/C can call run()
    ↓
Track B can execute cells, Track C can run full_pipeline.sio
```

---

## Track B: sounio-jupyter Handoffs

### What Track B Delivers

| File | Purpose | Used by |
|------|---------|---------|
| `sounio_kernel/kernel.py` | SounioKernel(Kernel) class | Jupyter front-end |
| `sounio_kernel/executor.py` | CellExecutor (wraps Track A) | kernel.py |
| `sounio_kernel/display.py` | HTML rendering + coloring | kernel.py |
| `sounio_kernel/magics.py` | Magic commands (%sounio_check, etc) | kernel.py |
| `kernelspec/kernel.json` | Jupyter kernel metadata | jupyter kernelspec |

### Auto-wrapper Behavior

**Cells without `fn` are auto-wrapped:**

```python
# Input cell:
let x = 1 + 1

# Auto-wrapped before execution:
fn main() with IO, Mut, Div, Panic {
    let x = 1 + 1
}

# Output: 2
```

### HTML Display Coloring

**Knowledge values displayed with color based on epsilon (confidence):**

```python
# display.py color_for_epsilon(epsilon)
if epsilon >= 0.9:
    color = "#2ecc71"  # Green (high confidence)
elif epsilon >= 0.7:
    color = "#f39c12"  # Orange (medium)
else:
    color = "#e74c3c"  # Red (low)
```

### kernel.json Environment Setup

**Critical:** kernel.json must specify SOUC and SOUNIO_STDLIB_PATH paths:

```json
{
  "display_name": "Sounio",
  "language": "sounio",
  "argv": ["python", "-m", "sounio_kernel", "{connection_file}"],
  "env": {
    "SOUC": "/path/to/souc-linux-x86_64-jit",
    "SOUNIO_STDLIB_PATH": "/path/to/stdlib"
  }
}
```

**Resolution strategy:**
1. At install time: `install.py` detects souc binary location
2. Fallback: Read from env var at kernel startup
3. Error: If neither found, kernel refuses to start with helpful message

### Magic Commands

**Required magics for Day 13 integration:**

| Magic | Implementation | Used for |
|-------|----------------|----------|
| `%sounio_check` | Call `souc check <tmpfile>` | Type-check current cell |
| `%show_types` | Parse output of `souc check --show-types` | Display inferred types |
| `%drug_pipeline` | Run `sounio.run_file("drug-discovery/examples/full_pipeline.sio")` | Execute pipeline in-notebook |

---

## Track C: drug-discovery Handoffs

### What Track C Delivers

| File | Purpose | Used by |
|------|---------|---------|
| `sounio.toml` | Package metadata (minimal format) | souc compiler |
| `src/types.sio` | Compilable KnowledgeF64, ScreeningResult, etc | All other modules |
| `src/screening.sio` | Virtual screening stage | full_pipeline.sio |
| `src/pkpd.sio` | PK/PD fitting stage | full_pipeline.sio |
| `src/simulation.sio` | Monte Carlo simulation stage | full_pipeline.sio |
| `examples/full_pipeline.sio` | End-to-end demo | demo.py (Test 2) |

### Output Format Requirements

**All outputs MUST use canonical format:**

```sio
// In full_pipeline.sio
let screening_result = screen_molecule(...)
print_knowledge(screening_result.score)
// → Knowledge { value: X epsilon: Y prov: "screening" }
```

### PK/PD Integration Risk

**File:** `stdlib/ode/epistemic_pk_fit.sio` is used by Track C.

**Pre-Day 4 check required:**

```bash
$SOUC check stdlib/ode/epistemic_pk_fit.sio
```

If it uses `pub struct` and that causes parse errors in `pkpd.sio`:
- Option 1: Copy PKFitResult definition locally (no `pub`)
- Option 2: Ask if `pub` can be removed from stdlib

### Full Pipeline Output Example

**Expected output from:** `SOUC=... SOUNIO_STDLIB_PATH=... $SOUC run examples/full_pipeline.sio`

```
=== Stage 1: Virtual Screening ===
Knowledge { value: 0.750 epsilon: 0.95 prov: "screening" }
Knowledge { value: 0.820 epsilon: 0.92 prov: "screening" }
...

=== Stage 2: PK/PD Fitting ===
Knowledge { value: 0.15 epsilon: 0.88 prov: "pk_fit" }
Knowledge { value: 0.05 epsilon: 0.85 prov: "pk_fit" }
...

=== Stage 3: Simulation ===
Knowledge { value: 0.72 epsilon: 0.80 prov: "simulation" }
Knowledge { value: 0.28 epsilon: 0.75 prov: "simulation" }

Pipeline complete
```

---

## Demo.py Integration Flow

**Day 12 execution:**

```
┌─────────────────────────────────────────────┐
│  python triple-sounio-ecosystem/demo.py     │
├─────────────────────────────────────────────┤
│                                             │
│  Test 1: Import sounio (Track A)            │
│  ✅ Create Knowledge(42.0, 0.1, "test")    │
│                                             │
│  Test 2: Run pipeline (Track C + A)         │
│  ✅ sounio.run_file("drug-discovery/...")  │
│     Parses Knowledge from stdout            │
│                                             │
│  Test 3: Check kernel (Track B)             │
│  ✅ jupyter kernelspec list | grep sounio  │
│                                             │
└─────────────────────────────────────────────┘
```

**Test 2 Parsing Detail:**

```python
# In demo.py:
result = sounio.run_file("drug-discovery/examples/full_pipeline.sio")
output = result.stdout + result.stderr

# Find all Knowledge values
pattern = r"Knowledge \{ value: ([\d.e+-]+) epsilon: ([\d.e+-]+) prov: \"([^\"]+)\" \}"
matches = re.findall(pattern, output)

for value, epsilon, prov in matches:
    print(f"Found: {value} ± {epsilon} ({prov})")
```

---

## Critical Failure Modes

### Failure: "SOUC binary not found"

**Symptom:** Test 2 fails with file not found error

**Diagnosis:**
```bash
echo $SOUC
ls $SOUC
```

**Fix:**
```bash
export SOUC=/home/demetrios/RustroverProjects/sounio/artifacts/omega/souc-bin/souc-linux-x86_64-jit
python triple-sounio-ecosystem/demo.py
```

### Failure: "cannot find stdlib"

**Symptom:** `souc run` complains about missing `use` imports (epistemic, ode, etc)

**Diagnosis:**
```bash
echo $SOUNIO_STDLIB_PATH
ls $SOUNIO_STDLIB_PATH/epistemic/
```

**Fix:**
```bash
export SOUNIO_STDLIB_PATH=/home/demetrios/RustroverProjects/sounio/stdlib
python triple-sounio-ecosystem/demo.py
```

### Failure: "sounio module not found"

**Symptom:** Test 1 fails with ImportError

**Diagnosis:**
```bash
cd triple-sounio-ecosystem/sounio-py
maturin develop --help  # Check if installed
```

**Fix:**
```bash
cd triple-sounio-ecosystem/sounio-py
maturin develop
python -c "import sounio; print(sounio.__file__)"
```

### Failure: "jupyter kernel not found"

**Symptom:** Test 3 fails, kernelspec doesn't list sounio

**Diagnosis:**
```bash
cd triple-sounio-ecosystem/sounio-jupyter
python install.py --help
```

**Fix:**
```bash
cd triple-sounio-ecosystem/sounio-jupyter
pip install -e .
python install.py
jupyter kernelspec list  # Check installation
```

### Failure: "regex doesn't match output"

**Symptom:** Knowledge values in output but demo.py can't parse them

**Diagnosis:**
```bash
# Run pipeline directly
$SOUC run triple-sounio-ecosystem/drug-discovery/examples/full_pipeline.sio
# Look at actual output format
```

**Fix:**
```python
# Update pattern if actual format differs, e.g.:
# Actual: "Knowledge: value=42.0, epsilon=0.1, prov=test"
# Update pattern accordingly
pattern = r"Knowledge: value=([\d.e+-]+), epsilon=([\d.e+-]+), prov=(\w+)"
```

---

## Handoff Schedule

| Day | Track | Deliverable | Validation |
|-----|-------|-------------|-----------|
| 3 | A/B/C | MVP 1: Compilable/installable base | Builds without errors |
| 7 | A | _executor.py ready | B can import and use it |
| 7 | B/C | Core implementations done | Run basic examples |
| 10 | A/B/C | Full pipelines working | demo.py ready for Day 12 |
| 11 | D | Integration validation starts | Checklist review |
| 12 | D | demo.py passes all tests | Track D lead runs tests |
| 13 | D | Offload docs + fixes | READMEs written |
| 14 | D | Final gate verification | All systems go |

---

## Success Metrics

✅ **Integration successful when:**

1. **demo.py Test 1 PASS:** `sounio.Knowledge(42.0, 0.1, "test")` works
2. **demo.py Test 2 PASS:** `sounio.run_file("drug-discovery/examples/full_pipeline.sio")` outputs with "Pipeline complete"
3. **demo.py Test 3 PASS:** `jupyter kernelspec list` shows "sounio"
4. **No hardcoded paths** in any project code (all use env vars)
5. **All README files exist** with working installation instructions
6. **Canonical format** respected throughout (regex parses all outputs)
7. **Zero hangs** (all subprocess calls have timeouts)
8. **Clear error messages** if dependencies missing (graceful degradation)

---

## Questions & Escalations

| Question | Answer | Escalation |
|----------|--------|-----------|
| Can `pub` be removed from `epistemic_pk_fit.sio`? | Check with original author or copy locally | Day 4 pre-check |
| Should `SOUC` path be auto-detected? | Yes, at install time via `which souc` or env var | Track A `install.py` |
| What if souc takes >30s to run pipeline? | Increase timeout; profile souc startup | Track C complexity |
| Can we use pure Python for Knowledge instead of PyO3? | Yes, but slower; Rust is preferred for performance | Trade-off decision |

