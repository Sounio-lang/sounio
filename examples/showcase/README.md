<!-- docs:meta
topic_id: repo.examples.showcase.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.examples.showcase.readme
-->

# Sounio Showcase Examples

Ten self-contained programs demonstrating Sounio's unique capabilities for both scientific computing and systems programming. Every example type-checks and runs end-to-end.

```bash
# Run any example
souc run examples/showcase/<name>.sio

# Type-check only
souc check examples/showcase/<name>.sio
```

---

## Scientific Computing

### measurement_lab.sio — GUM Uncertainty Propagation
ISO GUM-compliant measurement uncertainty through arithmetic. Three physics examples (Ohm's law, power dissipation, thermal expansion) with expanded uncertainty at 95% confidence and sensitivity analysis showing which input dominates total uncertainty.

**Features:** structs, first-order uncertainty propagation, coverage factors, contribution indices

### drug_dose_optimizer.sio — Pharmacokinetic Modeling
Two-compartment pharmacokinetic model simulating oral drug absorption, distribution, and elimination. Propagates parameter uncertainty through Euler-integrated ODEs to produce concentration bands, then checks whether the drug stays within the therapeutic window.

**Features:** ODE integration (Euler), epistemic uncertainty bands, structs, effects

### genome_motif_scanner.sio — DNA Motif Scanning
Scans a DNA sequence for transcription factor binding sites using a position weight matrix (PWM). Each motif hit is scored in bits of information content and assigned epistemic confidence based on how far the score exceeds threshold relative to the PWM's maximum possible score.

**Features:** PWM scoring, information content (bits), epistemic confidence, bioinformatics

### ode_predator_prey.sio — Dynamical Systems
Lotka-Volterra predator-prey model integrated with RK4. Runs the system with nominal and perturbed parameters to produce uncertainty bands on population trajectories, then verifies conservation of the Volterra integral invariant.

**Features:** RK4 integration, parameter perturbation, conservation law verification

### knowledge_graph_trainer.sio — Sedenion KG Embeddings
Trains 16-dimensional sedenion (hypercomplex) embeddings on a small knowledge graph using the DistMult scoring function and margin-based negative sampling. Evaluates with Mean Rank and Hits@K metrics.

**Features:** 16D sedenion algebra, SGD training loop, link prediction evaluation

### spectral_analyzer.sio — Signal Processing
Generates a multi-frequency test signal, computes its DFT, estimates the noise floor, and identifies spectral peaks with epistemic confidence derived from signal-to-noise ratio.

**Features:** DFT, peak detection, SNR-based epistemic confidence, trigonometry

---

## Systems Programming

### linear_file_server.sio — Resource Safety with Linear Types
Demonstrates linear type semantics for file handles and connection pools. The compiler guarantees that every opened resource is closed exactly once — no leaks, no double-free. Includes scope guards for exception-safe cleanup.

**Features:** linear structs, resource lifecycle, connection pooling, scope guards, effects

### effect_test_harness.sio — Testable I/O via Algebraic Effects
Shows how algebraic effects decouple business logic from I/O. A data processing pipeline logs messages through a captured effect handler rather than real I/O, enabling deterministic testing of the full pipeline without side effects.

**Features:** algebraic effects, captured log, test assertions, mock I/O

### concurrent_pipeline.sio — Multi-Stage Data Processing
A four-stage data pipeline (validate, normalize, enrich, score) with quality tracking at each stage. Demonstrates the producer-consumer pattern with metrics aggregation. Each stage filters or transforms data items and tracks processed/filtered counts.

**Features:** pipeline pattern, structs, arrays, quality metrics, effects

### type_safe_units.sio — Dimensional Analysis
Runtime-checked dimensional analysis preventing unit mismatches (e.g., adding meters to kilograms). Includes a projectile motion simulation where every physical quantity carries its SI dimension, and a kinetic energy calculation that verifies dimensional correctness.

**Features:** dimension structs, unit checking, physics simulation, derived units

---

## Compiler Bug Notes

**Implicit `var` return with `i32`:** When a function's return type is `i32` and the last expression is a `var` variable (implicit return), the type checker may report `expected I32, found I64`. Workaround: use `return x` instead of a trailing `x`. This does not affect `f64` returns or explicit `return` statements.
