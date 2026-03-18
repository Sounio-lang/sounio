# BENCHMARKS.md - Metas de Performance da Stdlib Sounio

Data da última atualização: 2026-03-17

Este documento define metas explícitas de performance para a standard library (stdlib), alinhadas com o plano original e resultados de validação em [`benchmarks/stdlib_validation/VALIDATION_REPORT.md`](../benchmarks/stdlib_validation/VALIDATION_REPORT.md).

Siga [`CONVENTIONS.md`](CONVENTIONS.md) para benchmarks por módulo.

## 1. Metas Gerais

| Categoria | Meta | Referência |
|-----------|------|------------|
| **Álgebra Linear** | <2x | NumPy (BLAS otimizado) |
| **Estatística** | <1.5x | SciPy |
| **Diferenciação Automática Epistêmica (AD)** | <3x | JAX |

## 2. Por Módulo

| Módulo | Baseline | Meta | Status |
|--------|----------|------|--------|
| [`linalg/matrix.sio`](linalg/matrix.sio) | NumPy (BLAS) | <2x | ✅ Precisão: matmul [L9], SVD [L10] |
| [`stats`](stats/lib.sio) | SciPy | <1.5x | ✅ Precisão: OLS [L8] |
| [`epistemic/knowledge.sio`](epistemic/knowledge.sio) | JAX | <3x | ✅ Precisão: epistemic matmul [L11], GUM [L12] |
| [`math`](math/lib.sio) | Rust std::f64 | <1.5x | ⏳ Performance pendente |
| [`optimize`](optimize/lib.sio) | SciPy.optimize | <2x | ⏳ Pendente |
| [`integrate`](integrate/lib.sio) | SciPy.integrate | <2x | ⏳ Cross-lang |
| [`autodiff`](autodiff/lib.sio) | JAX | <3x | ⏳ Pendente |
| [`nn`](nn/lib.sio) | PyTorch | <3x | ⏳ Pendente |
| [`prob`](prob/lib.sio) | SciPy.stats | <1.5x | ⏳ Pendente |
| [`bayes`](bayes/lib.sio) | PyMC | <4x | ⏳ Pendente |
| [`causal`](causal/lib.sio) | doWhy | <3x | ⏳ Pendente |
| [`ode`](ode/lib.sio) | SciPy.odeint | <2x | ⏳ Cross-lang |
| [`quantum`](quantum/lib.sio) | Qiskit | <5x | ⏳ Pendente |
| [`signal`](signal/lib.sio) | SciPy.signal | <2x | ⏳ Pendente |
| [`special`](special/lib.sio) | SciPy.special | <1.5x | ⏳ Pendente |
| [`collections`](collections/lib.sio) | Rust std | <1.2x | ⏳ Pendente |
| [`io`](io/lib.sio) | Rust std | <1.5x | ⏳ Pendente |
| [`json`](json/lib.sio) | serde_json | <2x | ⏳ Pendente |
| [`csv`](csv/lib.sio) | csv crate | <1.5x | ⏳ Pendente |
| [`plot`](plot/lib.sio) | matplotlib | <5x | ⏳ Pendente |
| [`units`](units/lib.sio) | pint | <2x | ⏳ Pendente |
| [`gpu`](gpu/lib.sio) | CuBLAS | <1.5x | ✅ GPU GEMM 17% peak |

**Legenda de Status:**
- ✅ Validado em [`VALIDATION_REPORT.md`](../benchmarks/stdlib_validation/VALIDATION_REPORT.md)
- ⏳ Pendente de implementação/benchmark

**Notas de linha (L) referenciam [`VALIDATION_REPORT.md`](../benchmarks/stdlib_validation/VALIDATION_REPORT.md):**
- L8: OLS coeffs - Erro 5.2e-13
- L9: matmul Mat4 - Erro 2.1e-14
- L10: SVD sigma - Erro 8.4e-14
- L11: epistemic matmul - Erro 1.2e-15
- L12: GUM propagation - coverage 95% NIST

## 3. Epistemic Overhead

**Meta:** Overhead máximo de <20% slowdown vs equivalente determinístico para todas operações numéricas.

### Justificativa

Tipos epistêmicos ([`Knowledge<T>`](epistemic/knowledge.sio), [`GUMUncertainty`](epistemic/gum.sio)) propagam incerteza automaticamente. O overhead aceitável é 20% para garantir que a segurança epistêmica não comprometa viabilidade em produção.

### Como Medir

```bash
# Benchmark epistêmico
sio bench --epistemic linalg/matrix.sio

# Benchmark determinístico equivalente
sio bench linalg/matrix.sio
```

Comparar tempos de execução; razão deve ser < 1.2x.

## 4. Como Rodar Benchmarks

### 4.1 Validação de Precisão

```bash
cd benchmarks/stdlib_validation
python3 validate.py
```

Saída esperada: todos os testes ✅ PASS com erro relativo dentro das metas.

### 4.2 Benchmarks Cross-Language

Comparação Sounio vs Python vs Julia para ODE, álgebra linear e incerteza:

```bash
bash benchmarks/comparison/run_all.sh
```

### 4.3 Benchmarks GPU

```bash
scripts/gpu_test_runner.sh
```

Relatório completo: [`benchmarks/results/NVIDIA_L4_BENCHMARKS.md`](../benchmarks/results/NVIDIA_L4_BENCHMARKS.md)

### 4.4 Por Módulo (Futuro)

```bash
sio bench stdlib/linalg/
sio bench stdlib/stats/
sio bench stdlib/epistemic/
```

### 4.5 Validação Científica Completa

```bash
cd benchmarks/stdlib_validation
python3 validate.py --full
```

## 5. Referências

- [`CONVENTIONS.md`](CONVENTIONS.md) - Padrões de desenvolvimento stdlib
- [`benchmarks/README.md`](../benchmarks/README.md) - Visão geral dos benchmarks
- [`benchmarks/stdlib_validation/VALIDATION_REPORT.md`](../benchmarks/stdlib_validation/VALIDATION_REPORT.md) - Resultados de validação
- [`benchmarks/results/NVIDIA_L4_BENCHMARKS.md`](../benchmarks/results/NVIDIA_L4_BENCHMARKS.md) - Resultados GPU

## 6. Histórico de Revisões

| Data | Autor | Mudança |
|------|-------|---------|
| 2026-03-17 | Sistema | Criação inicial baseada em VALIDATION_REPORT.md |
