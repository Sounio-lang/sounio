# Benchmarks de Validação Stdlib vs Referências Científicas

## Objetivo
Validar precisão numérica da stdlib Sounio contra:
- `stats/regression/linear.sio` vs SciPy `linregress` (erro relativo coeffs < 1e-10)
- `linalg/matrix.sio` + `decomp.sio` matmul/SVD vs NumPy/LAPACK (erro < 1e-12)
- `epistemic/gum.sio` coverage 95% vs NIST GUM Supplement 1 examples

## Estrutura
- `stats_regression_linear/`: OLS epistemic
- `linalg_matrix/`: matmul Mat4 + SVD Matrix4x4
- `epistemic_gum/`: Propagação GUM NIST ex1-3

Cada dir tem:
- `sounio_bench.sio`: Executa stdlib, print JSON results
- `python_bench.py`: SciPy/NumPy ref, print JSON
- `validate.py`: Compara erros relativos, PASS/FAIL

## Execução
```bash
cd benchmarks/stdlib_validation/<dir>
sounio run sounio_bench.sio > sounio.json
python3 python_bench.py > ref.json
python3 validate.py
```
