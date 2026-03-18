# Triple Sounio Ecosystem — Offload Prompts (Days 13-14)

**Track D Lead:** Use these prompts with `/offload-expand grok` on Day 13 after Tracks A/B/C complete.

---

## Offload 1: sounio-py README

**Execute on Day 13:**
```bash
/offload-expand grok sounio-py/README.md
```

**Prompt to use:**

```
Write a compelling 300-word README for the sounio-py Python package.

Target audience: Python scientists and data analysts familiar with NumPy/pandas.

Required sections (not necessarily in order):
1. What it is: PyO3 binding to Sounio, a novel epistemic computing language
   - Emphasize that Sounio is NOT a Rust/Julia dialect
   - Highlight epistemic computing (uncertainty tracking) as the key innovation

2. Installation: Two methods
   - End-user: pip install sounio (future PyPI release)
   - Developer: maturin develop in the repo

3. Quick start example (code block):
   ```python
   import sounio

   # Create a Knowledge value with uncertainty
   measured_dose = sounio.Knowledge(
       value=500.0,        # measured dose in mg
       epsilon=0.05,       # 5% uncertainty
       provenance="lab_balance_XYZ"
   )

   print(measured_dose)
   # Output: Knowledge(500.0 ± 0.05, prov='lab_balance_XYZ')
   ```

4. Key features:
   - Knowledge[T] type for uncertain values (generative uncertainty model)
   - Epsilon (ε) represents confidence [0, 1] (not std deviation)
   - Provenance string tracks measurement source
   - NumPy array integration (EpistemicArray)
   - Pandas DataFrame support (DataFrameWithUncertainty)
   - Execute Sounio code directly: sounio.run(code), sounio.run_file(path)

5. Documentation links:
   - Main Sounio project: link to repo
   - Epistemic computing intro: link to docs/
   - Examples: link to examples/

6. Why epistemic computing matters:
   - Traditional software: single point values (false precision)
   - Scientific software: needs uncertainty quantification (GUM standard)
   - Sounio: first language to embed epistemic types + uncertainty propagation
   - Use case: drug discovery, climate modeling, biomedical simulation

Tone:
- Technical but accessible (assume knowledge of Python, not Sounio)
- Emphasize world-first innovation (epistemic uncertainty in a language)
- Friendly and inviting (lowercase, relaxed)
- Avoid jargon unless defined

Length: Exactly ~300 words
```

---

## Offload 2: sounio-jupyter README

**Execute on Day 13:**
```bash
/offload-expand grok sounio-jupyter/README.md
```

**Prompt to use:**

```
Write a compelling 300-word README for sounio-jupyter, a Jupyter kernel for Sounio.

Target audience: Data scientists and researchers familiar with Jupyter notebooks.

Required sections (not necessarily in order):
1. What it is: Interactive REPL for Sounio epistemic computing in Jupyter notebooks
   - Show that it feels like familiar Jupyter, but with epistemic types

2. Installation: Step-by-step
   - pip install sounio-jupyter (future PyPI)
   - python -m sounio_jupyter.install (kernel installer)
   - Verification: jupyter kernelspec list should show "sounio"

3. Example notebook cells (markdown + code):
   - Simple arithmetic: let x = 1 + 1  (auto-wrapped as fn main)
   - Knowledge creation: let dose = {value: 500, epsilon: 0.05, prov: "measurement"}
   - Epistemic calculation: automatic uncertainty propagation
   - Magic commands (next section)

4. Magic commands (% syntax):
   - %sounio_check <code> — type-check code without running
   - %show_types — display type inference for current cell
   - %drug_pipeline — execute full drug discovery pipeline in-notebook, display results
   Example: %drug_pipeline

5. Display & visualization:
   - Knowledge values displayed with color coding by confidence:
     * Green (#2ecc71): ε ≥ 0.9 (high confidence)
     * Orange (#f39c12): 0.7 ≤ ε < 0.9 (medium)
     * Red (#e74c3c): ε < 0.7 (low confidence)
   - Auto-rendering in cell output (no extra code needed)

6. Documentation links:
   - Main Sounio project: link to repo
   - Kernel architecture: link to kernel.py
   - Examples: link to notebooks/

7. Why interactive epistemic computing:
   - Exploratory data analysis with built-in uncertainty tracking
   - Immediate feedback on measurement precision
   - Scientists can see confidence visually (color) without reading numbers
   - Entire analysis notebook preserves uncertainty provenance

Tone:
- User-friendly and encouraging (this is for interactive use)
- Visual and interactive emphasis (Jupyter is UI-first)
- Emphasize ease of use (auto-wrapper, magic commands, colored output)
- Friendly, warm, approachable

Length: Exactly ~300 words
```

---

## Offload 3: drug-discovery README

**Execute on Day 13:**
```bash
/offload-expand grok drug-discovery/README.md
```

**Prompt to use:**

```
Write a compelling 300-word README for the drug-discovery Sounio package.

Target audience: Computational chemists, pharmacometricians, and drug discovery scientists.

Required sections (not necessarily in order):
1. What it is: Epistemic drug discovery pipeline in Sounio
   - World-first: uncertainty propagation through all stages (GUM standard)
   - Three-stage pipeline: screening → PK/PD → simulation
   - Native Sounio implementation (NOT wrapped Python)

2. Three pipeline stages:
   a) Virtual screening (Stage 1)
      - Lipinski's Rule of Five filters
      - Outputs: molecular score with epistemic uncertainty
      - Source: chemical property measurements

   b) Pharmacokinetic/Pharmacodynamic fitting (Stage 2)
      - ODE solver for PK models (Ka, Ke, half-life)
      - Epistemic NLLS fitting (propagates measurement uncertainty)
      - Outputs: fitted parameters ± confidence

   c) Clinical trial simulation (Stage 3)
      - Monte Carlo simulation (32 patients, LCG RNG)
      - Efficacy & toxicity with epistemic covariance
      - Outputs: efficacy/toxicity distributions with confidence intervals

3. How to run:
   ```bash
   export SOUC=/path/to/souc-linux-x86_64-jit
   export SOUNIO_STDLIB_PATH=/path/to/stdlib
   $SOUC run examples/full_pipeline.sio

   # Output:
   # === Stage 1: Virtual Screening ===
   # Knowledge { value: 0.750 epsilon: 0.95 prov: "screening" }
   # ...
   # === Stage 3: Simulation ===
   # Knowledge { value: 0.72 epsilon: 0.80 prov: "simulation" }
   # Pipeline complete
   ```

4. Epistemic uncertainty propagation:
   - Each stage outputs Knowledge { value, epsilon, provenance }
   - GUM (ISO Guide to Measurement) standard
   - Uncertainty comes from measurement + model error, not assumption
   - Enables risk-aware drug development (vs. false confidence)

5. Python API:
   - Use via sounio-py: import sounio; result = sounio.run_file("full_pipeline.sio")
   - Jupyter integration: %drug_pipeline magic in sounio-jupyter

6. Documentation links:
   - Main Sounio project: link to repo
   - GUM standard: link to NIST guide
   - Example notebooks: link to triple-sounio-ecosystem/notebooks/
   - Test suite: link to tests/

7. Why epistemic computing for drug discovery:
   - Clinical outcomes depend on precise measurements + model accuracy
   - Traditional pipelines use point estimates (false confidence)
   - Drug failures from underestimated uncertainty → patient harm
   - Sounio makes uncertainty first-class: traceable, quantified, auditable

Tone:
- Scientific and rigorous (audience is experts)
- Emphasize rigor & safety (drug discovery context)
- Highlight novelty (first epistemic drug pipeline)
- Straightforward, professional, not marketing-heavy

Length: Exactly ~300 words
```

---

## How to Execute (Day 13 Workflow)

1. **Prepare files:**
   ```bash
   cd /home/demetrios/RustroverProjects/sounio/triple-sounio-ecosystem
   touch sounio-py/README.md sounio-jupyter/README.md drug-discovery/README.md
   ```

2. **Run offload for each README:**
   ```bash
   /offload-expand grok sounio-py/README.md
   # Paste the "Offload 1" prompt above when prompted

   /offload-expand grok sounio-jupyter/README.md
   # Paste the "Offload 2" prompt above

   /offload-expand grok drug-discovery/README.md
   # Paste the "Offload 3" prompt above
   ```

3. **Review each generated README:**
   - Check length (~300 words target)
   - Verify all required sections present
   - Ensure code examples are valid
   - Check tone matches (friendly, technical, rigorous respectively)

4. **Edit for accuracy:**
   If actual implementation differs from prompt, update the generated text:
   - Actual file paths (if different from prompt examples)
   - Actual API surface (if _executor.py has different interface)
   - Actual output format (if canonical Knowledge format differs)

5. **Finalize:**
   - Commit each README to git
   - Link to each other (cross-references)
   - Add to project index

---

## Ecosystem-Level README (Days 13-14)

After individual READMEs are done, create `/triple-sounio-ecosystem/README.md`:

**Structure:**
```
# Triple Sounio Ecosystem

## Overview
[1-2 sentences explaining the three projects work together]

## Projects
- sounio-py: [1-line description + link]
- sounio-jupyter: [1-line description + link]
- drug-discovery: [1-line description + link]

## Quick Start
[Copy-paste setup from INTEGRATION_NOTES.md: env vars + demo.py]

## Architecture
[Diagram showing A→B, A→C, canonical format]

## Canonical Format
[Knowledge { value, epsilon, prov } specification]

## Integration Testing
[How to run demo.py]

## Contributing
[Link to main Sounio CONTRIBUTING.md]

## License
[Same as main Sounio]
```

---

## Quality Checklist (after offloads complete)

- [ ] Each README has 280-320 words (~300 target)
- [ ] All code examples are syntactically correct
- [ ] No grammatical errors
- [ ] Links to external projects work (or are reasonable paths)
- [ ] Installation instructions are step-by-step
- [ ] Each README is self-contained (don't assume reader knows others)
- [ ] Tone is appropriate for audience
- [ ] Epistemic computing is explained for non-experts
- [ ] No hardcoded paths (use env vars or placeholders)

---

## Notes for Track D Lead

- **Timing:** Execute these offloads on Day 13 only (after all tracks complete)
- **Provider:** Use Grok (grok is best for technical writing)
- **Fallback:** If offload provider unavailable, write manually using these prompts
- **Cost:** ~3 offload tokens per README (~9 total)
- **Time:** ~2-3 minutes per offload (typical)
- **Review:** Read each result carefully; Grok occasionally hallucinates product features

