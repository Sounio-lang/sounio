# Triple Sounio Ecosystem — Integration Checklist

**Timeline:** Days 11-14 (Track D execution after Tracks A/B/C complete)

---

## Day 11: Cross-Track Integration Validation

### Track A (sounio-py) Completion Checklist
- [ ] `Cargo.toml` specifies pyo3 0.21 + abi3-py38
- [ ] `pyproject.toml` has `[tool.maturin]` with correct paths
- [ ] `maturin develop` succeeds without errors
- [ ] `sounio/__init__.py` imports `_sounio_native` module
- [ ] `Knowledge` class instantiation works: `sounio.Knowledge(42.0, 0.1, "test")`
- [ ] All attributes accessible: `.value`, `.epsilon`, `.provenance`
- [ ] `_executor.py` exists and has `run(code)` and `run_file(path)` functions
- [ ] `run_file()` returns object with `.stdout`, `.stderr`, `.returncode` attributes
- [ ] **Critical:** Canonical output format parsing regex matches `"Knowledge { value: X epsilon: Y prov: \"Z\" }"`

### Track B (sounio-jupyter) Completion Checklist
- [ ] `pyproject.toml` lists ipykernel >= 6.0 as dependency
- [ ] `sounio_kernel/kernel.py` defines `SounioKernel(Kernel)` class
- [ ] Auto-wrapper works: code without `fn` gets wrapped with `fn main() with IO, Mut, Div, Panic { ... }`
- [ ] Cell execution calls `_executor.py.run()` (depends on Track A)
- [ ] `sounio_kernel/display.py` has `EpistemicDisplay` class
- [ ] HTML coloring works: epsilon >= 0.9 → green, >= 0.7 → orange, else → red
- [ ] Magic commands registered: `%sounio_check`, `%show_types`, `%drug_pipeline`
- [ ] `install.py` successfully runs `jupyter kernelspec install`
- [ ] `jupyter kernelspec list` shows "sounio" kernel (verifies Day 12 Test 3)

### Track C (drug-discovery) Completion Checklist
- [ ] `sounio.toml` has minimal format (no Cargo-style features)
- [ ] `src/types.sio` compiles with `souc check`
  - [ ] `struct KnowledgeF64 { value: f64, epsilon: f64, prov: string }`
  - [ ] `struct ScreeningResult { molecule_id: string, score: KnowledgeF64, confidence: f64 }`
  - [ ] `struct PKResult { ka, u_ka, ke, u_ke, half_life, u_half_life: f64, converged: i32 }`
  - [ ] `struct SimulationResult { efficacy_mean, efficacy_std, toxicity_mean, toxicity_std: f64, n_patients: i64 }`
- [ ] `src/screening.sio` compiles and exports `screen_molecule()`
- [ ] `src/pkpd.sio` compiles (imports or redefines PKFitResult, no `pub`)
- [ ] `src/simulation.sio` compiles with LCG RNG and 32-patient array
- [ ] All tests pass:
  - [ ] `souc check tests/test_screening.sio` — pass
  - [ ] `souc check tests/test_pkpd.sio` — pass
  - [ ] `souc check tests/test_simulation.sio` — pass
- [ ] `examples/full_pipeline.sio` runs end-to-end:
  ```bash
  SOUC=../../bin/souc \
  SOUNIO_STDLIB_PATH=../../stdlib \
  $SOUC run examples/full_pipeline.sio
  ```
- [ ] Output contains all three stages: "Stage 1...", "Stage 2...", "Stage 3...", "Pipeline complete"

---

## Day 12: Integration Testing (demo.py)

### Canonical Output Format Validation
- [ ] All three projects output `Knowledge { value: X epsilon: Y prov: "Z" }` format
- [ ] `_executor.py` regex successfully parses all Knowledge outputs
- [ ] Pattern: `r"Knowledge \{ value: ([\d.e+-]+) epsilon: ([\d.e+-]+) prov: \"([^\"]+)\" \}"`

### Demo.py Execution
- [ ] **Test 1 (sounio-py Knowledge):** `python demo.py`
  ```
  Test 1: import sounio; x = Knowledge(42.0, 0.1, "test") ✅
  ```
- [ ] **Test 2 (drug-discovery pipeline):** Via `sounio.run_file()`
  ```
  Test 2: full_pipeline.sio runs, stdout contains "Pipeline complete" ✅
  ```
- [ ] **Test 3 (Jupyter kernel):** `jupyter kernelspec list | grep sounio`
  ```
  Test 3: kernel discovered ✅
  ```
- [ ] All tests PASS or properly SKIP (if dependencies not ready)

### Cross-Track Dependency Validation
- [ ] **Track A → Track B dependency:** `sounio_kernel/executor.py` can import and use `sounio._executor`
- [ ] **Track A → Track C dependency:** `sounio.run_file("drug-discovery/examples/full_pipeline.sio")` works
- [ ] **Track B → Track A dependency:** Auto-wrapper uses `_executor.py.run()` for cell execution

---

## Days 13-14: Documentation + Final Verification

### Offload Documentation Tasks (use `/offload-expand grok`)
- [ ] **Offload 1:** sounio-py/README.md (300 words)
  - Installation: pip + maturin develop
  - Quick start: Knowledge object with uncertainty
  - Features: epsilon/provenance, numpy/pandas integration
- [ ] **Offload 2:** sounio-jupyter/README.md (300 words)
  - Interactive REPL in Jupyter
  - Installation + example cells
  - Magic commands, colored Knowledge display
- [ ] **Offload 3:** drug-discovery/README.md (300 words)
  - Epistemic drug pipeline (screening, PK/PD, simulation)
  - How to run full_pipeline.sio
  - Python API via sounio-py

### Ecosystem-Level Documentation
- [ ] **triple-sounio-ecosystem/README.md** exists (overview, setup, all 3 projects)
- [ ] **INTEGRATION_NOTES.md** documents critical handoffs (executable paths, env vars, format specs)
- [ ] Each project has its own README with installation + quick start

### Final Gate Verification
- [ ] All 3 projects have passing tests (souc check, pytest, jupyter test)
- [ ] `demo.py` passes all 3 tests
- [ ] Environment variables properly documented:
  - `SOUC` (souc binary path)
  - `SOUNIO_STDLIB_PATH` (stdlib directory)
- [ ] Jupyter kernel.json has correct SOUC + SOUNIO_STDLIB_PATH env vars
- [ ] No hardcoded paths (all use env vars or discovery at install time)

---

## Critical Failure Modes & Recovery

| Failure | Symptom | Recovery |
|---------|---------|----------|
| Track A slow | Knowledge class takes >10s to import | Profile Cranelift JIT; consider pure Python fallback for quick test |
| Track B kernel not discoverable | Test 3 fails | Check `jupyter --version`; re-run `install.py` with `--user` or `--system` |
| Track C souc not found | Test 2 fails with "souc binary not found" | Set `SOUC` env var; verify `bin/souc` exists |
| Canonical format mismatch | Regex doesn't parse output | Compare actual output with pattern; update pattern if format differs |
| Deadlock in souc process | demo.py hangs | Ensure stderr is drained in background thread (Rust process.rs) |
| Demo.py import errors | Test 1 ImportError | Run `maturin develop` in sounio-py/ and verify `$PYTHONPATH` |

---

## Success Criteria

✅ **INTEGRATION COMPLETE** when:
1. All three projects compile/install independently
2. `demo.py` passes all 3 tests (or skips appropriately)
3. Each project has a working README
4. No hardcoded paths; all env vars documented
5. End-to-end pipeline (virtual screening → PK/PD fit → simulation) produces meaningful output
6. Jupyter kernel works interactively with colored Knowledge display
7. Python API (sounio-py) can execute Sounio code and parse results

---

## Track D Lead Responsibilities

**Days 1-3 (NOW):**
- Create demo.py skeleton ✅
- Create integration checklist ✅
- Prepare offload prompts (see below)
- Monitor Tracks A/B/C for blockers

**Days 11-12:**
- Run demo.py against final outputs
- Validate all checklist items
- Fix integration issues (env vars, paths, format mismatches)

**Days 13-14:**
- Execute offload documentation tasks
- Write ecosystem-level README
- Final gate verification

---

## Offload Prompt Templates (prepare, execute Day 13)

### Offload 1: sounio-py README
```
Provider: grok
Task: expand
File: triple-sounio-ecosystem/sounio-py/README.md

Prompt:
"Write a 300-word README for the sounio-py Python package.

Topics:
1. What it is: PyO3 binding to Sounio epistemic computing language
2. Installation: pip install (once on PyPI) + maturin develop for dev
3. Quick start: Create Knowledge object with value, epsilon, provenance
4. Key features: Knowledge[T] with uncertainty tracking, numpy array integration, pandas DataFrame support
5. Documentation links: sounio main project docs

Tone: Technical but accessible to Python scientists. Emphasize epistemic computing as novel feature."
```

### Offload 2: sounio-jupyter README
```
Provider: grok
Task: expand
File: triple-sounio-ecosystem/sounio-jupyter/README.md

Prompt:
"Write a 300-word README for sounio-jupyter Jupyter kernel.

Topics:
1. What it is: Interactive Sounio REPL in Jupyter notebooks
2. Installation: pip install + jupyter kernelspec install
3. Example cells: Simple arithmetic (auto-wrapped), Knowledge creation, epistemic calculations
4. Magic commands: %sounio_check (type-check), %show_types (type inference), %drug_pipeline (run drug discovery)
5. Display: Colored Knowledge values (green/orange/red by confidence)
6. Documentation links

Tone: User-friendly, emphasize interactivity and visual display of uncertainty."
```

### Offload 3: drug-discovery README
```
Provider: grok
Task: expand
File: triple-sounio-ecosystem/drug-discovery/README.md

Prompt:
"Write a 300-word README for drug-discovery Sounio package.

Topics:
1. What it is: Epistemic drug discovery pipeline with PK/PD modeling
2. Three stages: Virtual screening (Lipinski rules), pharmacokinetic fitting (ODE solver with uncertainty), Monte Carlo simulation
3. How to run: SOUC=path/to/souc run examples/full_pipeline.sio
4. Epistemic computing: Uncertainty propagates through all stages (GUM)
5. Python API: Import via sounio-py, run_file() integration
6. Documentation links: Sounio main project, epistemic computing papers

Tone: Scientific but accessible, emphasize world-first epistemic uncertainty in drug pipeline."
```

