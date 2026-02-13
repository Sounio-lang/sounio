# Figures (Compiler Preprint)

This folder contains small, text-based figure inputs (for example CSV) that are
used to generate plots in `docs/compiler/TECHNICAL_REPORT.tex`.

Generated artifacts:
- `octonion_matmul_points.csv`: roofline plot points derived from Criterion output.

To regenerate:
```bash
python3 scripts/roofline_octonion_matmul.py \
  --criterion-dir target/criterion/octonion_matmul \
  --out-csv docs/compiler/figures/octonion_matmul_points.csv
```

