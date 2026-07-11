Shadow Lanes arXiv bundle
=========================

Main TeX file: main.tex
Bibliography:  refs.bib

Suggested arXiv categories:
  - cs.DC (primary)
  - cs.PL
  - cs.MS

Build locally:
  pdflatex main
  bibtex main
  pdflatex main
  pdflatex main

Or with pandoc from Markdown (optional):
  pandoc shadow_lanes_preprint.md -o main_from_md.tex --bibliography=refs.bib

Reproducibility anchor:
  https://github.com/Sounio-lang/sounio
  benchmarks/results/NVIDIA_L4_BENCHMARKS.md
