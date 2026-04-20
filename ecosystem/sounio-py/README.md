# Sounio-py: Python Bindings for Epistemic Computing

**Versão proposta:** 0.1.0
**Objetivo:** Permitir que cientistas usem o poder epistêmico do Sounio diretamente de notebooks Jupyter e scripts Python.

## Instalação (futuro)

```bash
pip install sounio
# ou
souc install sounio-py
```

## API Principal

### 1. `Knowledge<T>` — O coração epistêmico

```python
from sounio import Knowledge, measure, confidence_gate
from sounio.stats import normal, beta
import numpy as np

# Criação de valores epistêmicos
dose = measure(500.0, uncertainty=25.0, unit="mg", source="HPLC_2025")
volume = measure(10.0, uncertainty=0.2, unit="mL", source="pipette_A")

# Operações com propagação automática de incerteza (GUM)
concentration = dose / volume
print(concentration)
# Knowledge(value=50.0, uncertainty=2.51, confidence=0.93, provenance=...)

# Comparação com gates de confiança
if confidence_gate(concentration > 45.0, min_confidence=0.90):
    print("Faixa terapêutica atingida com alta confiança")
else:
    print("Precisa de mais dados ou melhor calibração")

# Distribuições probabilísticas epistêmicas
prior = normal(mu=0.0, sigma=1.0, epistemic=True)
posterior = beta(alpha=5, beta=2)
```

### 2. Modelos Científicos (PBPK Example)

```python
from sounio.pbpk import DarwinPBPK14, simulate_epistemic
from sounio import Knowledge

params = DarwinPBPK14(
    clearance=Knowledge(10.5, uncertainty=1.2, unit="L/h"),
    volume_central=Knowledge(15.0, uncertainty=2.0, unit="L"),
    bbb_permeability=Knowledge(0.8, uncertainty=0.15)
)

# Simulação com propagação epistêmica completa
result = simulate_epistemic(
    model=params,
    dose=Knowledge(500.0, uncertainty=10.0, unit="mg"),
    duration=48.0,
    n_samples=1000
)

print(f"Brain AUC: {result.brain_auc}")
print(f"Confidence: {result.brain_auc.confidence:.1%}")
print(f"Provenance: {result.provenance_summary()}")
```

### 3. Integração com Ecossistema Python

```python
import numpy as np
import pandas as pd
from sounio import KnowledgeArray, epistemic_corr
from sounio.ml import EpistemicMLP

# Arrays epistêmicos (mantém uncertainty por elemento)
data = KnowledgeArray.from_numpy(
    values=np.array([1.2, 2.5, 3.1]),
    uncertainties=np.array([0.1, 0.3, 0.15])
)

# Correlação epistêmica
corr = epistemic_corr(data, target)

# Rede neural com propagação de incerteza
model = EpistemicMLP(hidden=[64, 32], epistemic=True)
prediction = model.predict_with_uncertainty(test_data)
```

### 4. Compilação JIT + Execução

```python
from sounio import compile, run

# Compila código Sounio diretamente do Python
model = compile("""
use epistemic::Knowledge;
use pbpk::DarwinPBPK14;

fn run_model(dose: Knowledge<f64>) -> Knowledge<f64> {
    let model = DarwinPBPK14::default();
    model.simulate(dose)
}
""")

result = model.run(dose=Knowledge(500.0, uncertainty=25.0))
```

## Vantagens vs Alternativas

- **vs `uncertainties` (Python)**: Propagação nativa no compilador, muito mais rápida e com formal verification
- **vs JAX + PyMC**: Combina autodiff, MCMC **e** epistemic reasoning com provenance
- **vs Measurements.jl (Julia)**: Melhor integração com prova formal e regulatory compliance

## Roadmap Curto

1. **Fase 1 (2 meses)**: Binding básico de `Knowledge`, operações aritméticas, PBPK wrapper
2. **Fase 2 (4 meses)**: `sounio.compile()` JIT, suporte a arrays, integração com numpy/pandas
3. **Fase 3 (6 meses)**: Epistemic neural networks + regulatory reporting tools

---

**Próximo arquivo:** `ecosystem/sounio-py/src/sounio/__init__.py` (prototipagem)

Esta API foi projetada para ser **natural para cientistas** enquanto preserva todo o poder epistêmico do Sounio.
