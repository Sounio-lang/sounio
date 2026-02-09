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
```

## Files

- `paper.md` — JOSS paper source
- `paper.bib` — JOSS bibliography
- `sounio-epistemic-types.tex` — TeX preprint source
- `references.bib` — TeX preprint bibliography
- `Makefile` — TeX build script

## Target Venues

- **Primary**: arXiv cs.PL (Programming Languages)
- **Secondary**: OOPSLA, PLDI, or Software X journal

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
