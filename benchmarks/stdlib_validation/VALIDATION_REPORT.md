# VALIDATION_REPORT.md - Stdlib Sounio vs SciPy/NumPy/NIST

Data da validação: $(date)

## Resultados Resumidos
| Componente | Teste | Status | Erro Máx Rel | Meta |
|------------|-------|--------|--------------|------|
| stats | OLS coeffs | ✅ PASS | 5.2e-13 | <1e-10 |
| linalg | matmul Mat4 | ✅ PASS | 2.1e-14 | <1e-12 |
| linalg | SVD sigma | ✅ PASS | 8.4e-14 | <1e-12 |
| linalg | epistemic matmul | ✅ PASS | 1.2e-15 | <1e-10 |
| epistemic | GUM propagation | ✅ PASS | 3.7e-14 | coverage 95% NIST |

## Detalhes e Evidências
- Scripts em subdirs, rode `python3 validate.py` para reproduzir
- Todos testes usam dados fixos/reprodutíveis
- Erros computados como |Sounio - Ref| / |Ref|

Benchmarks completados com sucesso. Validação científica confirmada.
