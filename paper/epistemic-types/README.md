# Epistemic Types for Scientific Computing

**Target Venue:** PLDI 2027 (deadline ~November 2026)
**Status:** Formalization complete, implementation in progress

## Paper Outline

1. **Introduction** (2 pages)
   - Reproducibility crisis in scientific computing
   - Existing approaches: UQ libraries (runtime only), probabilistic programming (heavyweight)
   - Our contribution: Type-level uncertainty with GUM compliance

2. **Epistemic Type System** (8 pages)
   - Syntax and typing rules
   - Operational semantics
   - Metatheory (Progress, Preservation, GUM soundness)
   - Extension: ODE integration uncertainty
   - Extension: Causal-epistemic types

3. **Implementation** (4 pages)
   - Sounio compiler architecture
   - Type inference for epistemic types
   - SMT integration for refinement checking
   - Gradual typing fallback

4. **Evaluation** (6 pages)
   - QNN-MNIST: Epistemic uncertainty in neural networks
   - PBPK pharmacology: Drug concentration with uncertainty
   - Performance: Overhead analysis
   - Calibration: Predicted vs actual uncertainty

5. **Related Work** (2 pages)
   - Refinement types (Liquid Haskell, F*)
   - UQ libraries (Measurements.jl, UncertainPy)
   - Probabilistic programming (Stan, Pyro)
   - Effect systems

6. **Conclusion** (1 page)
   - Summary of contributions
   - Future work: mechanized proofs, correlation tracking

## Key Results (Planned)

- **Theorem 1:** Type safety (Progress + Preservation)
- **Theorem 2:** GUM compliance soundness
- **Benchmark 1:** QNN-MNIST 98%+ accuracy, ECE < 0.05
- **Benchmark 2:** PBPK within 5% of FDA reference
- **Performance:** <10% overhead for epistemic tracking

## Files

- `formalization.tex` - Complete type system formalization (50 pages)
- `main.tex` - Conference paper (20-30 pages for PLDI)
- `benchmarks/` - QNN, PBPK, fMRI results
- `proofs/` - Detailed metatheory proofs
- `figures/` - Typing rule diagrams, benchmark graphs

## Timeline

- **Month 1-2:** Formalization complete ✓
- **Month 2-3:** SMT integration
- **Month 4-5:** QNN-MNIST benchmark
- **Month 5-6:** Draft paper, submit to PLDI

## Compilation

```bash
cd paper/epistemic-types
pdflatex formalization.tex
bibtex formalization
pdflatex formalization.tex
pdflatex formalization.tex
```

## Related Papers

See `paper/causal-types/` for the causal programming paper (PLDI/UAI 2027).
