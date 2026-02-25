# Sounio Papers

This directory contains both the legacy TeX preprint source and the JOSS submission source.

## Abstract

We present Sounio, a systems programming language with native support for
*epistemic types*—type-level representations of uncertain values that
automatically propagate measurement uncertainty through computations.

## Building

```bash
make        # Build PDF
make arxiv  # Create arXiv submission tarball
make clean  # Remove build artifacts
./reproduce.sh  # Build + benchmark logs + reference checks
```

## Files

- `paper.md` — JOSS paper source
- `paper.bib` — JOSS bibliography
- `sounio-epistemic-types.tex` — TeX preprint source
- `sounio-ieee-cise.tex` — IEEE CiSE manuscript source
- `references.bib` — TeX preprint bibliography
- `Makefile` — TeX build script
- `reproduce.sh` — end-to-end reproducibility script

## Target Venues

### Legacy Preprint
- **Primary**: arXiv cs.PL (Programming Languages)
- **Secondary**: OOPSLA, PLDI, or Software X journal

### Research Track (18-Month Roadmap)

#### Paper 1: Epistemic Types for Scientific Computing
- **Directory:** `epistemic-types/`
- **Target:** PLDI 2027 or ICFP 2027
- **Status:** Formalization complete (Month 1-2 ✓)
- **Contribution:** First type system with GUM-compliant uncertainty propagation

#### Paper 2: Causal Programming with do-Calculus Types
- **Directory:** `causal-types/`
- **Target:** PLDI 2027 or UAI 2027
- **Status:** Planned for Month 7-8
- **Contribution:** Compile-time causal identifiability verification

#### Paper 3: Quaternionic Neural Networks with Epistemic Uncertainty
- **Directory:** `qnn-epistemic/`
- **Target:** NeurIPS 2027 or ICML 2028
- **Status:** Planned for Month 15-16
- **Contribution:** Type-safe epistemic neural networks

## Citation

```bibtex
@article{chiuratto2026sounio,
  author = {Chiuratto Agourakis, Demetrios},
  title = {Sounio: Epistemic Types for Scientific Computing with Native Uncertainty Quantification},
  journal = {arXiv preprint},
  year = {2026},
  doi = {10.5281/zenodo.18404188}
}
```

## License

CC BY 4.0
