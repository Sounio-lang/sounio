# Arquitetura Integrada: Triple Sounio Ecosystem

## 🏗️ Visão Arquitetural

```
┌─────────────────────────────────────────────────────────┐
│                    APLICAÇÃO CLIENTE                    │
│  (Python Scientist, Jupyter, Web Dashboard)             │
└─────────────────┬─────────────────┬─────────────────────┘
                  │                 │
          ┌───────▼───────┐ ┌───────▼───────┐
          │   sounio-py   │ │ sounio-jupyter│
          │  (Bindings)   │ │   (Kernel)    │
          └───────┬───────┘ └───────┬───────┘
                  │                 │
          ┌───────▼─────────────────▼───────┐
          │      CAMADA DE INTEGRAÇÃO       │
          │  • FFI Bridge                  │
          │  • Type Conversion             │
          │  • Async Communication         │
          └─────────────────────────────────┘
                  │
          ┌───────▼─────────────────────────┐
          │     CORE SOUNIO ECOSYSTEM       │
          │  • Drug Discovery Pipeline      │
          │  • Epistemic Types (shared)     │
          │  • Scientific Algorithms        │
          └─────────────────────────────────┘
                  │
          ┌───────▼─────────────────────────┐
          │      RUNTIME SOUNIO             │
          │  • souc Compiler               │
          │  • Knowledge[T] Runtime        │
          │  • GPU/CPU Backends            │
          └─────────────────────────────────┘
```

## 🔧 Componentes Principais

### 1. Camada Compartilhada (`shared/`)

#### `epistemic_types.sio`
```sounio
// Tipos epistêmicos compartilhados por todos os projetos
struct Knowledge[T] {
    value: T,
    ε: f64,           // Confidence (0.0-1.0)
    prov: string,     // Provenance
    metadata: Metadata,
}

struct Metadata {
    timestamp: i64,
    source: string,
    validation: ValidationInfo,
    units: string,
}

// Conversores para Python/Jupyter
fn to_python_dict(k: Knowledge[T]) -> PythonDict
fn from_python_dict(dict: PythonDict) -> Knowledge[T]
fn to_jupyter_display(k: Knowledge[T]) -> JupyterDisplay
```

#### `ffi_bridge.sio`
```sounio
// Ponte FFI para comunicação com Python
struct FFIBridge {
    // Chamada Python → Sounio
    call_python_function: fn(string, [PythonValue]) -> PythonValue,
    
    // Chamada Sounio → Python
    expose_sounio_function: fn(string, fn([Value]) -> Value),
    
    // Gerenciamento de memória compartilhada
    shared_memory: SharedMemoryBuffer,
    
    // Serialização/Deserialização
    serialize: fn(Value) -> bytes,
    deserialize: fn(bytes) -> Value,
}
```

### 2. sounio-py: Python Bindings

#### Estrutura:
```python
# sounio-py/src/sounio/__init__.py
"""Sounio Python Bindings"""

from .core import (
    Knowledge,
    epistemic_array,
    run_sounio,
    compile_sounio,
    EpistemicNumpyArray,
)

from .jupyter_support import (
    register_jupyter_magics,
    create_epistemic_widget,
    display_with_uncertainty,
)

# Integração automática com NumPy
import numpy as np
from .numpy_integration import (
    epistemic_from_numpy,
    epistemic_to_numpy,
    register_epistemic_ufuncs,
)
```

#### API Python:

> **Partial surface.** `sounio.Knowledge` IS bound in the checked artifact (`ecosystem/sounio-py/python/sounio/__init__.py:43-50`, via the native `_sounio_native` extension, falling back to the pure-Python `_knowledge` module), and the runner is exposed as `run_sio` (with the `run_code` convenience wrapper) at line 54. NOT bound: `epistemic_array` and the exact `sounio.run` name (only `run_sio`/`run_code` exist). Note the field-name drift between this sketch and the package: the constructors take `uncertainty`/`unit`/`confidence`, **not** `epsilon`/`units` — the native `Knowledge(value, uncertainty=0.0, confidence=1.0, unit="", prov="")` accepts `prov` (and exposes `epsilon` only as a post-construction property alias for `uncertainty`), while the pure-Python fallback `Knowledge(value, uncertainty=0.0, confidence=1.0, unit="", source="direct", provenance=None)` uses `source`/`provenance` instead of `prov`. The runner also performs **no** Python→Sounio value marshalling: `run_sio` (= module `run`) has signature `(source, *args, optimize=False, capture=True)` and rejects a `data=` keyword, and `run_code(code, timeout=30, **kwargs)` merely shells out to `souc run` on a temp file with no value marshalling. The inline Sounio code below uses `Knowledge[f64]`, `epistemic_mean`, and `epistemic_std`, which are **not** the Sounio language API; the checked epistemic surface is `Epistemic` with `ep_measured`, `ep_val`, `ep_std`, and `ep_div` from `stdlib/epistemic/knowledge.sio`. The sketch is the intended binding shape only.

```python
import sounio
import numpy as np

# Criar valores epistêmicos (native Knowledge ctor: value, uncertainty,
# confidence, unit, prov; the pure-Python fallback uses source/provenance)
temp = sounio.Knowledge(36.5, uncertainty=0.1,
                        unit="°C", prov="thermometer")

# NOTE: sounio.epistemic_array is NOT bound; there is no array helper yet.
data = [sounio.Knowledge(v, uncertainty=0.05, unit="°C")
        for v in np.random.normal(0, 1, 100)]

# Executar código Sounio (run_sio takes (source, *args); no data= injection)
result = sounio.run_sio("""
    fn analyze(data: Knowledge[f64]) -> Knowledge[f64] {
        let mean = epistemic_mean(data)
        let std = epistemic_std(data)
        mean / std  // Coefficient of variation
    }
""")

print(f"Result: {result.value} ± {result.uncertainty}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Provenance: {result.prov}")
```

### 3. sounio-jupyter: Kernel Jupyter

#### Arquitetura do Kernel:
```python
# sounio-jupyter/src/sounio_kernel/kernel.py

class SounioKernel(Kernel):
    """Kernel Jupyter para Sounio"""
    
    def __init__(self):
        super().__init__()
        self.sounio_compiler = SounioCompiler()
        self.epistemic_display = EpistemicDisplay()
        self.widget_manager = WidgetManager()
    
    def do_execute(self, code, silent, store_history=True,
                   user_expressions=None, allow_stdin=False):
        # Compilar e executar código Sounio
        result = self.sounio_compiler.execute(code)
        
        # Formatar saída com incerteza
        display_data = self.epistemic_display.format(result)
        
        # Enviar para frontend
        if not silent:
            self.send_response(self.iopub_socket,
                'display_data',
                {'data': display_data, 'metadata': {}})
        
        return {
            'status': 'ok',
            'execution_count': self.execution_count,
            'payload': [],
            'user_expressions': {},
        }
```

#### Magic Commands:
```python
# sounio-jupyter/src/sounio_kernel/magics.py

@register_magic
def epistemic_visualize(line):
    """%epistemic_visualize - Visualize uncertainty"""
    # Implementação da visualização
    
@register_magic  
def load_clinical_data(line):
    """%load_clinical_data file.csv - Load clinical data"""
    # Carregar dados para pipeline
```

### 4. Drug Discovery Platform

#### Pipeline Arquitetural:
```sounio
// drug-discovery/src/lib.sio

struct DrugDiscoveryPipeline {
    // Etapas do pipeline
    virtual_screening: VirtualScreening,
    pkpd_modeling: PKPDModeling,
    clinical_simulation: ClinicalSimulation,
    decision_analysis: DecisionAnalysis,
    
    // Gerenciamento de dados
    data_manager: DataManager,
    uncertainty_propagator: UncertaintyPropagator,
    provenance_tracker: ProvenanceTracker,
}

// Cada etapa retorna Knowledge[T]
impl DrugDiscoveryPipeline {
    fn run_pipeline(
        molecules: [Molecule],
        patient_data: Knowledge[PatientData]
    ) -> Knowledge[PipelineResult] {
        
        // Etapa 1: Triagem virtual
        let candidates = self.virtual_screening.screen(molecules)
        
        // Etapa 2: Modelagem PK/PD
        let pkpd_profiles = self.pkpd_modeling.model(candidates)
        
        // Etapa 3: Simulação clínica
        let trial_results = self.clinical_simulation.simulate(
            pkpd_profiles, patient_data)
        
        // Etapa 4: Análise de decisão
        let decision = self.decision_analysis.analyze(trial_results)
        
        // Rastrear proveniência através de todas as etapas
        self.provenance_tracker.track(decision, [
            candidates.prov,
            pkpd_profiles.prov,
            trial_results.prov
        ])
        
        decision
    }
}
```

## 🔄 Fluxo de Dados

### Python → Sounio:
```
Python Data → sounio-py → FFI Bridge → Sounio Runtime → Pipeline
    │           │              │              │            │
    NumPy       Bindings    Serialization   Execution   Results
    Arrays                 ↔   JSON/msgpack  Native Code  with ε
```

### Jupyter → Pipeline:
```
Jupyter Cell → Kernel → Sounio Compiler → Pipeline → Visualization
     │           │           │               │           │
   Code Input  Execute    Compile & Run   Process    Display with
     Sounio    Request     Epistemic Code  with ε    Uncertainty
```

### Pipeline → Dashboard:
```
Pipeline → Results → Dashboard Server → Web UI → Interactive Viz
    │         │           │               │           │
  Sounio   Knowledge[T]  HTTP/WebSocket  React/Vue   D3.js with
  Runtime   with ε & prov               Components   Error Bars
```

## 🗃️ Gerenciamento de Dados

### Estrutura de Dados Compartilhada:
```sounio
// shared/data_models.sio

// Dados clínicos com incerteza
struct ClinicalData {
    patients: [Patient],
    metadata: StudyMetadata,
    uncertainty: StudyUncertainty,
}

struct Patient {
    id: string,
    measurements: [ClinicalMeasurement],
    demographics: Demographics,
}

struct ClinicalMeasurement {
    parameter: string,  // "blood_pressure", "heart_rate"
    value: Knowledge[f64],
    timestamp: i64,
    instrument: string,
}
```

### Serialização para Python/Jupyter:
```sounio
fn clinical_data_to_dataframe(
    data: ClinicalData
) -> PythonDataFrame {
    // Converter para pandas DataFrame
    // Incluir colunas de incerteza e proveniência
}

fn dataframe_to_clinical_data(
    df: PythonDataFrame,
    uncertainty_columns: [string]
) -> ClinicalData {
    // Converter DataFrame de volta
    // Extrair incerteza das colunas especificadas
}
```

## 🎨 Visualização

### Componentes de Visualização:
```javascript
// Widgets Jupyter para incerteza
class EpistemicVisualizer {
    // Gráfico com barras de erro
    static errorBarChart(data) { /* ... */ }
    
    // Visualização de proveniência
    static provenanceGraph(provenance) { /* ... */ }
    
    // Dashboard interativo
    static interactiveDashboard(pipelineResults) { /* ... */ }
}
```

### Dashboard Web:
```sounio
// drug-discovery/src/dashboard/web_ui.sio

fn create_dashboard(
    pipeline_results: Knowledge[PipelineResult]
) -> WebDashboard {
    WebDashboard {
        title: "Drug Discovery Dashboard",
        panels: [
            EfficacyPanel(results.efficacy),
            SafetyPanel(results.safety),
            UncertaintyPanel(results.uncertainty),
            ProvenancePanel(results.provenance),
            DecisionPanel(results.decision),
        ],
        interactivity: InteractiveFeatures {
            sensitivity_analysis: true,
            what_if_scenarios: true,
            confidence_adjustment: true,
        }
    }
}
```

## ⚡ Otimizações de Performance

### 1. Memória Compartilhada:
```sounio
// shared_memory.sio
struct SharedMemoryBuffer {
    // Buffer compartilhado Python-Sounio
    data: [u8],
    
    // Views tipadas
    as_f64_array: fn() -> [f64],
    as_knowledge_array: fn() -> [Knowledge[f64]],
    
    // Gerenciamento de lifetime
    lock: Mutex,
    reference_count: i64,
}
```

### 2. Compilação JIT:
```sounio
// JIT compiler para hot paths do pipeline
struct PipelineJITCompiler {
    fn compile_pipeline_stage(
        stage: PipelineStage
    ) -> CompiledStage {
        // Compilar para código nativo
        // Especializar para tipos de dados
        // Cache de compilações frequentes
    }
}
```

### 3. Processamento em Lote:
```sounio
// Processamento batch otimizado
fn batch_process_molecules(
    molecules: [Molecule],
    batch_size: i64
) -> [Knowledge[ScreeningResult]] {
    // Processar em batches para melhor cache locality
    // Paralelizar com tasks assíncronas
    // Agregar resultados com propagação de incerteza
}
```

## 🔒 Segurança e Validação

### Validação de Dados:
```sounio
struct DataValidator {
    fn validate_clinical_data(
        data: ClinicalData
    ) -> ValidationResult {
        // Verificar consistência
        // Validar faixas fisiológicas
        // Checar proveniência
        // Calcular confiança agregada
    }
}
```

### Auditoria de Proveniência:
```sounio
struct ProvenanceAuditor {
    fn audit_pipeline_run(
        result: Knowledge[PipelineResult]
    ) -> AuditReport {
        // Rastrear todas as transformações
        // Verificar conservação de incerteza
        // Validar cadeia de custódia dos dados
        // Gerar relatório de reprodutibilidade
    }
}
```

## 🚀 Estratégia de Implantação

### Fase 1: Desenvolvimento
- Desenvolver os 3 projetos em paralelo
- Integração diária
- Testes de integração contínuos

### Fase 2: Integração
- Conectar Python ↔ Jupyter ↔ Pipeline
- Criar demos integradas
- Otimizar performance

### Fase 3: Produção
- Empacotar para PyPI (sounio-py)
- Publicar kernel no Jupyter
- Deploy do pipeline como serviço
- Paper reproduzível

## 📊 Monitoramento

### Métricas do Sistema:
```sounio
struct SystemMetrics {
    // Performance
    execution_times: [StageTiming],
    memory_usage: MemoryStats,
    
    // Qualidade
    uncertainty_levels: [f64],
    confidence_scores: [f64],
    provenance_completeness: f64,
    
    // Uso
    api_calls: [APICall],
    user_interactions: [UserInteraction],
}
```

### Logging com Incerteza:
```sounio
fn log_with_uncertainty(
    message: string,
    confidence: f64,
    provenance: string
) {
    // Log estruturado incluindo metadados epistêmicos
    // Para análise posterior e debugging
}
```

---

Esta arquitetura permite que os 3 projetos se desenvolvam independentemente
mas se integrem perfeitamente quando combinados. A chave é a camada compartilhada
que define os contratos e tipos comuns.
