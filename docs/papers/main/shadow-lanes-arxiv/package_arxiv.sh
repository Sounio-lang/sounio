#!/usr/bin/env bash
# Build a flat arXiv submission directory for Epistemic Shadow Lanes preprint.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
STAGING="${1:-/tmp/shadow-lanes-arxiv-submit}"
TARBALL="${2:-$HERE/shadow-lanes-arxiv.tar.gz}"

cd "$HERE"

python3 md_to_tex.py "$ROOT/preprint.md"

rm -rf "$STAGING"
mkdir -p "$STAGING"

cp main.tex refs.bib "$STAGING/"
cp "$ROOT/preprint.md" "$STAGING/shadow_lanes_preprint.md"
cp metadata.yaml "$STAGING/"

cat > "$STAGING/00README.txt" <<'EOF'
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
EOF

tar -czf "$TARBALL" -C "$STAGING" .
echo "Created $TARBALL ($(du -h "$TARBALL" | awk '{print $1}'))"
echo "Contents:"
tar -tzf "$TARBALL"