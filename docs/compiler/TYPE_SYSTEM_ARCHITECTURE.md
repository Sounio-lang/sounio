# Sistema de Tipos e Type Checker - Arquitetura do Compilador Sounio

## Visão Geral

O sistema de tipos do Sounio é um dos mais avançados entre linguagens de programação, combinando:

1. **Tipo Epistêmico `Knowledge<T>`** - Valores com incerteza e proveniência
2. **Análise Dimensional** - Unidades físicas verificadas em tempo de compilação
3. **Inferência Bidirecional** - Type synthesis + checking
4. **Tipos Lineares/Affine** - Ownership semântica
5. **Efeitos Algébricos** - Tracking de side effects
6. **Tipos Semânticos** - Compatibilidade ontológica

```
Source (.sio)
    ↓
┌─────────────────┐
│     Parser      │  → AST (untyped)
└─────────────────┘
    ↓
┌─────────────────┐
│  Name Resolver  │  → Resolved names
└─────────────────┘
    ↓
┌─────────────────┐
│  Type Checker   │  → HIR (typed)
│  - Inference    │  → Type errors
│  - Unification  │  → Effect checking
│  - Units        │  → Epistemic constraints
└─────────────────┘
    ↓
   HIR (Typed)
```

## 1. Sistema de Tipos Epistêmicos

### Localização
- **Módulo principal**: [`crates/souc/src/epistemic/`](../../crates/souc/src/epistemic/)
- **Core**: [`crates/souc/src/epistemic/knowledge.rs`](../../crates/souc/src/epistemic/knowledge.rs) (471 linhas)
- **Operações**: [`crates/souc/src/epistemic/operations.rs`](../../crates/souc/src/epistemic/operations.rs) (540 linhas)

### Estrutura do Tipo Knowledge

```rust
// crates/souc/src/epistemic/knowledge.rs:35
pub struct Knowledge {
    /// Tipo de conteúdo (e.g., f64, custom struct)
    pub content: Box<Type>,

    /// τ: Índice de contexto temporal
    /// Tracks quando e em que contexto este conhecimento é válido
    pub temporal: ContextTime,

    /// ε: Status epistêmico (confiança, revisabilidade, fonte)
    pub epistemic: EpistemicStatus,

    /// δ: Binding de domínio ontológico
    /// Links this value to an ontology term for semantic validation
    pub domain: OntologyBinding,

    /// Φ: Traço de functor (proveniência de transformação)
    /// História completa de como este valor foi derivado
    pub provenance: Provenance,

    /// Source location
    pub span: Span,
}
```

### Arquitetura Ontológica (4 Camadas)

```rust
// crates/souc/src/epistemic/knowledge.rs:82
pub enum OntologyRef {
    /// L1: Primitivas (BFO, RO, COB) - ~850 termos, compilados
    Primitive(PrimitiveOntology),

    /// L2: Fundação (PATO, UO, IAO, Schema.org, FHIR) - ~8.000 termos
    Foundation(FoundationOntology),

    /// L3: Domínio (ChEBI, GO, DOID) - ~500.000 termos, lazy loaded
    Domain(DomainOntology),

    /// L4: Federated (BioPortal) - ~15.000.000 termos, resolução runtime
    Federated(FederatedRef),
}
```

### Operações Epistêmicas

```rust
// crates/souc/src/epistemic/operations.rs:19
pub enum KnowledgeOp {
    Assert(AssertOp),      // Criar novo conhecimento
    Query(QueryOp),        // Buscar conhecimento
    Revise(ReviseOp),     // Atualizar com nova evidência
    Translate(TranslateOp), // Converter entre ontologias
    Merge(MergeOp),       // Combinar múltiplas fontes
    Inspect(InspectOp),   // Extrair metadata
}
```

### Exemplo de Uso

```sio
// Medição com incerteza e proveniência
let mass: Knowledge[
    content = f64,
    τ = (2024, LabA, Experiment1),
    ε = (confidence: 0.95, source: Measurement),
    δ = PATO:mass,
    Φ = [sensor1 → calibration → conversion]
] = measure_mass(sample);

// Operações epistêmicas
assert dose : Knowledge[UO:milligram] = 500.0 with {
    confidence: 0.99,
    source: Measurement { instrument: "scale_001" }
};

let results = query Knowledge[δ: ChEBI, ε.confidence > 0.9]
    where relation(_, "inhibits", target);
```

## 2. Sistema de Unidades

### Localização
- **Módulo**: [`crates/souc/src/units/`](../../crates/souc/src/units/)
- **Dimensões**: [`crates/souc/src/units/dimension.rs`](../../crates/souc/src/units/dimension.rs) (553 linhas)
- **Quantidades**: [`crates/souc/src/units/quantity.rs`](../../crates/souc/src/units/quantity.rs) (540 linhas)

### Dimensões SI (7 Grandezas Base)

```rust
// crates/souc/src/units/dimension.rs:24
pub struct Dimension {
    pub mass: i8,        // [M] - quilograma
    pub length: i8,      // [L] - metro
    pub time: i8,        // [T] - segundo
    pub current: i8,     // [I] - ampere
    pub temperature: i8,  // [Θ] - kelvin
    pub amount: i8,       // [N] - mol
    pub luminosity: i8,   // [J] - candela
}

// Dimensões comuns pré-definidas
impl Dimension {
    pub const VELOCITY: Self = Self::new(0, 1, -1, 0, 0, 0, 0);  // [L T⁻¹]
    pub const FORCE: Self = Self::new(1, 1, -2, 0, 0, 0, 0);     // [M L T⁻²]
    pub const ENERGY: Self = Self::new(1, 2, -2, 0, 0, 0, 0);   // [M L² T⁻²]
    pub const CONCENTRATION: Self = Self::new(1, -3, 0, 0, 0, 0, 0); // [M L⁻³]
}
```

### Tipo Quantity

```rust
// crates/souc/src/units/quantity.rs:34
pub struct Quantity<N, U: Unit> {
    value: N,
    _unit: PhantomData<U>,
}

// Operações com verificação de unidades
impl<N: Add<Output = N>, U: Unit> Add for Quantity<N, U> {
    type Output = Quantity<N, U>;

    fn add(self, rhs: Self) -> Self::Output {
        // Erro de compilação: unidades diferentes!
        Quantity::new(self.value + rhs.value)
    }
}
```

### Exemplo de Uso

```sio
let mass: Quantity<f64, Kilogram> = Quantity::new(70.0);
let height: Quantity<f64, Meter> = Quantity::new(1.75);

// Erro de compilação:
// let wrong = mass + height;  // ERROR! Dimensões incompatíveis

// Operações que preservam unidades:
let velocity = height / time;  // Quantity<f64, MeterPerSecond>

// Units PK/PD
let dose: Quantity<f64, Milligram> = Quantity::new(500.0);
let volume: Quantity<f64, Liter> = Quantity::new(10.0);
let concentration = dose / volume;  // Quantity<f64, MilligramPerLiter>
```

## 3. Type Checker

### Localização
- **Principal**: [`crates/souc/src/check/mod.rs`](../../crates/souc/src/check/mod.rs) (8,385 linhas)

### Estrutura do TypeChecker

```rust
// crates/souc/src/check/mod.rs:142
pub struct TypeChecker {
    /// Ambiente de tipos (variável → tipo)
    env: TypeEnv,

    /// Definições de tipos
    type_defs: HashMap<String, TypeDef>,

    /// Contexto de inferência de efeitos
    effects: EffectInference,

    /// Verificador de unidades
    units: UnitChecker,

    /// Contador de variáveis de tipo fresh
    next_type_var: u32,

    /// Contador de variáveis de efeito fresh
    next_effect_var: u32,

    /// Restrições de tipo para unificação
    constraints: Vec<TypeConstraint>,

    /// Binding de variáveis de efeito
    effect_params: HashMap<String, types::EffectVar>,

    /// Erros acumulados
    errors: Vec<TypeError>,

    /// Alinhamentos ontológicos
    alignments: HashMap<(String, String), f64>,

    /// Thresholds de compatibilidade
    fn_thresholds: HashMap<String, f64>,
    default_threshold: f64,

    /// Verificador semântico ontológico
    ontology_resolver: Option<OntologyResolver>,

    /// Verificador de fidelidade do mundo
    fidelity_checker: Option<WorldFidelityChecker>,

    /// Verificador conformal para UQ
    conformal_checker: Option<ConformalTypeChecker>,
}
```

### Função Principal de Type Checking

```rust
// crates/souc/src/check/mod.rs:114
pub fn check(resolved_ast: &resolve::ResolvedAst) -> Result<Hir> {
    let mut checker = TypeChecker::new_with_resolved_ast(resolved_ast);
    checker.check_program(&resolved_ast.ast)
}

// crates/souc/src/check/mod.rs:120
pub fn check_ast(ast: &Ast) -> Result<Hir> {
    let resolved_ast = resolve::resolve(ast.clone())?;
    check(&resolved_ast)
}
```

### Ambiente de Tipos

```rust
// crates/souc/src/check/mod.rs:216
#[derive(Default)]
pub struct TypeEnv {
    scopes: Vec<Scope>,
    /// Bindings qualificados por módulo: (module_path, name) -> binding
    module_bindings: HashMap<(Vec<String>, String), TypeBinding>,
}

#[derive(Default)]
struct Scope {
    bindings: HashMap<String, TypeBinding>,
}

struct TypeBinding {
    ty: Type,
    mutable: bool,
    used: bool,
    source_module: Option<ModuleId>,
}
```

## 4. Definições de Tipos

### Tipo Core

```rust
// crates/souc/src/types/core.rs:5
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeVar(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EffectVar(pub u32);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DimSize {
    Const(usize),                    // Constante em tempo de compilação
    Symbolic(String),                // Variável simbólica (N, M)
    Var(DimVar),                    // Variável de dimensão
    Dynamic,                        // Determinado em runtime
    BinOp {                         // Operação binária
        op: DimOp,
        left: Box<DimSize>,
        right: Box<DimSize>,
    },
}
```

### Tensor Shapes

```rust
// crates/souc/src/types/core.rs:181
pub enum TensorShape {
    Static(Vec<usize>),              // Shape conhecido em compile-time
    Dynamic(usize),                  // Rank conhecido, dimensões dinâmicas
    Symbolic(Vec<String>),           // Dimensões nomeadas (generics)
    Parametric(Vec<DimSize>),        // Usando DimSize (forma mais geral)
}
```

## 5. Tipo Checker de Efeitos

### Localização
- **Principal**: [`crates/souc/src/effects/inference.rs`](../../crates/souc/src/effects/inference.rs)

### Estrutura de Efeitos

```rust
// crates/souc/src/types/effects.rs
pub struct EffectInference {
    /// Variáveis de efeito ativas
    effect_vars: HashMap<EffectVar, Effect>,
    /// Constraints de efeitos
    constraints: Vec<EffectConstraint>,
    /// handlers registrados
    handlers: HashMap<String, HandlerInfo>,
}
```

### Efeitos Suportados

| Efeito | Descrição | Handler |
|--------|-----------|---------|
| `IO` | Input/Output | `IOHandler` |
| `Mut` | Estado mutável | `MutHandler` |
| `Alloc` | Alocação de memória | `AllocHandler` |
| `Panic` | Pode panicar | `PanicHandler` |
| `Async` | Operações assíncronas | `AsyncHandler` |
| `GPU` | Execução GPU | `GpuHandler` |
| `Prob` | Operações probabilísticas | `ProbHandler` |
| `Div` | Divergência | `DivHandler` |

## 6. Restrições e Unificação

### Restrições de Tipo

```rust
// crates/souc/src/check/mod.rs:262
struct TypeConstraint {
    expected: Type,   // Tipo esperado (contexto)
    actual: Type,    // Tipo atual (expressão)
    span: Span,      // Para erro reporting
}
```

### Algoritmo de Unificação

O type checker usa unificação para resolver variáveis de tipo:

```rust
// Processo:
/// 1. Generate constraints from expressions
/// 2. Unify type variables
/// 3. Check for cycles (occurs check)
/// 4. Substitute unified types
```

### Exemplo de Inference

```sio
// Inferência de tipo
let x = 42;           // x: i32 (inferido do literal)
let y = x + 1.0;      // Erro: i32 + f64 mismatch

// Inference bidirecional
fn identity<T>(x: T) -> T {  // T é parâmetro de tipo
    x
}

let num = identity(42);      // identity::<i32>(42) - inferido
```

## 7. Verificação Semântica Ontológica

### Probabilistic Thresholds

```rust
// crates/souc/src/check/mod.rs:73
pub struct ProbabilisticThreshold {
    /// Prior sobre distância aceitável (Beta distribution)
    pub prior: BetaConfidence,
    /// Probabilidade requerida (e.g., 0.95)
    pub required_probability: f64,
}

impl ProbabilisticThreshold {
    /// Verifica se distância é aceitável com confiança bayesiana
    pub fn is_acceptable(&self, distance: f64, confidence: f64) -> bool {
        let posterior = BetaConfidence::new(
            self.prior.alpha + (1.0 - distance) * confidence * 10.0,
            self.prior.beta + distance * confidence * 10.0,
        );
        posterior.probability_above(1.0 - self.prior.mean()) >= self.required_probability
    }
}
```

### Alinhamentos Ontológicos

```rust
// crates/souc/src/check/mod.rs:163
/// Alinhamentos: (type1, type2) -> distance
alignments: HashMap<(String, String), f64>,

/// Thresholds por função do atributo #[compat]
fn_thresholds: HashMap<String, f64>,
default_threshold: f64,  // Default: 0.15
```

## 8. Suporte a Refinement Types

```rust
// crates/souc/src/check/mod.rs:57
struct RefinementInfo {
    var: String,                    // Nome da variável
    predicate: Box<Expr>,           // Predicate da AST
}

// Uso:
fn factorial(n: i32) -> i32 {
    require n >= 0;    // Refinement constraint
    if n <= 1 { 1 } else { n * factorial(n - 1) }
}
```

## 9. Módulos Avançados

### Conformal Prediction

```rust
pub use conformal::{
    CalibrationExample, ConformalConfig, ConformalResult,
    ConformalTypeChecker, MondrianConformalChecker,
};
```

### PAC Learning Types

```rust
pub use pac::{
    compute_generalization_gap, compute_sample_complexity,
    DeltaBound, EpsilonBound, ErrorBound, GeneralizationBound,
    HypothesisClass, PACBayesBound, RademacherBound, SampleBound,
    SampleComplexity, VCDimension,
};
```

### Tropical Geometry

```rust
pub use tropical::{
    parallel_compose, sequential_compose,
    ResourceType, TropicalMatrix, TropicalNumber,
};
```

## Fluxo de Type Checking

```
AST (untyped)
    ↓
resolve_names()
    ↓
ResolvedAst
    ↓
TypeChecker::check_program()
    ├─ check_items()
    │   ├─ check_function()
    │   │   ├─ check_params()
    │   │   ├─ infer_expr()  ← Bidirectional inference
    │   │   │   ├─ synthesize()  ← Infer from expected type
    │   │   │   └─ check()     ← Verify matches expected
    │   │   ├─ unify_constraints()
    │   │   └─ check_effects()
    │   ├─ check_struct()
    │   ├─ check_enum()
    │   └─ ...
    ├─ check_units()
    ├─ check_epistemic_constraints()
    └─ check_ontology_compatibility()
    ↓
HIR (typed)  OR  TypeErrors
```

## Métricas

| Componente | Linhas | Descrição |
|------------|--------|-----------|
| `types/mod.rs` | 87 | Exports e módulos principais |
| `types/core.rs` | 1,352 | Definições core de tipos |
| `check/mod.rs` | 8,385 | Type checker principal |
| `epistemic/mod.rs` | 167 | Sistema epistêmico |
| `epistemic/knowledge.rs` | 471 | Tipo Knowledge |
| `units/dimension.rs` | 553 | Dimensões SI |
| `units/quantity.rs` | 540 | Tipo Quantity |
| `effects/inference.rs` | ~500 | Inference de efeitos |

## Próximos Passos

1. **HIR → HLIR Lowering** - Conversão para IR com SSA
2. **Sistema de Efeitos** - Handlers, continuations, async
3. **Geração de Código** - Cranelift, LLVM, Native, GPU
4. **Otimizações** - Passes MIR, polyhedral

## Referências

- [`crates/souc/src/types/mod.rs`](../../crates/souc/src/types/mod.rs)
- [`crates/souc/src/check/mod.rs`](../../crates/souc/src/check/mod.rs)
- [`crates/souc/src/epistemic/mod.rs`](../../crates/souc/src/epistemic/mod.rs)
- [`crates/souc/src/units/mod.rs`](../../crates/souc/src/units/mod.rs)
