# Sistema de Efeitos - Arquitetura do Compilador Sounio

## Visão Geral

O sistema de efeitos do Sounio implementa **efeitos algébricos com handlers**, permitindo tracking explícito de side effects e composição flexível de comportamentos.

```
Function with Effects
        ↓
┌─────────────────┐
│  Effect Check   │  → Verifica efeitos declarados vs usados
└─────────────────┘
        ↓
┌─────────────────┐
│  Effect Infer   │  → Infere efeitos de operações
└─────────────────┘
        ↓
┌─────────────────┐
│   Handler       │  → Executa operações de efeito
│   Dispatch      │
└─────────────────┘
        ↓
   Runtime Execution
```

## 1. Estrutura do Sistema de Efeitos

### Localização

- **Principal**: [`crates/souc/src/effects/mod.rs`](../../crates/souc/src/effects/mod.rs)
- **Inferência**: [`crates/souc/src/effects/inference.rs`](../../crates/souc/src/effects/inference.rs) (1,335 linhas)
- **Handlers**: [`crates/souc/src/effects/handlers/`](../../crates/souc/src/effects/handlers/)
- **Continuations**: [`crates/souc/src/effects/continuation.rs`](../../crates/souc/src/effects/continuation.rs) (1,229 linhas)

### Módulos do Sistema

```rust
// crates/souc/src/effects/mod.rs:34
pub mod continuation;        // CPS transformation infrastructure
pub mod continuation_context; // Continuation context management
pub mod epistemic_effects;   // Epistemic tracking for effects
pub mod handler_capability;   // Handler trait definition
pub mod handlers;             // Concrete handler implementations
pub mod inference;           // Effect inference pass
pub mod jit_resume;          // JIT-based continuation resumption
pub mod linearity;            // Linearity checking for effects
pub mod resilience;           // Retry, circuit breaker patterns
pub mod resilient_dispatch;   // Resilient effect dispatch
pub mod simd_dispatch;       // SIMD parallel dispatch
```

## 2. Efeitos Suportados

### Tabela de Efeitos

| Efeito | Descrição | Handler | Operações |
|--------|-----------|---------|-----------|
| `IO` | Input/Output | [`IOHandler`](handlers/io_handler.rs) | `print`, `read_file`, `write_file` |
| `Mut` | Estado mutável | [`MutHandler`](handlers/mut_handler.rs) | `get`, `set`, `modify` |
| `Alloc` | Alocação | [`AllocHandler`](handlers/alloc_handler.rs) | `alloc`, `dealloc`, `realloc` |
| `Panic` | Falhas recuperáveis | [`PanicHandler`](handlers/panic_handler.rs) | `panic`, `assert`, `unwrap` |
| `Async` | Operações assíncronas | [`AsyncHandler`](handlers/async_handler.rs) | `spawn`, `await`, `join`, `select` |
| `GPU` | Execução GPU | [`GpuHandler`](handlers/gpu_handler.rs) | `launch`, `sync`, `alloc_device` |
| `Prob` | Probabilístico | [`ProbHandler`](handlers/prob_handler.rs) | `sample`, `observe`, `condition` |
| `Div` | Divisão | [`DivHandler`](handlers/div_handler.rs) | `div`, `checked_div`, `safe_div` |
| `Exn` | Exceções tipadas | [`ExnHandler`](handlers/exn_handler.rs) | `throw`, `try_catch`, `rethrow` |
| `Network` | Rede | [`NetworkHandler`](handlers/network_handler.rs) | `fetch`, `post`, `websocket` |
| `Sensor` | Sensores | [`SensorHandler`](handlers/sensor_handler.rs) | `read`, `calibrate`, `batch_read` |
| `Causal` | Inferência causal | [`CausalHandler`](handlers/causal_handler.rs) | `do`, `query`, `counterfactual` |
| `Epistemic` | Tracking epistêmico | [`EpistemicHandler`](handlers/epistemic_handler.rs) | `degrade`, `assert_confidence` |

## 3. Effect Inference

### Context do EffectChecker

```rust
// crates/souc/src/effects/inference.rs:102
pub struct EffectChecker<'a> {
    symbols: &'a SymbolTable,
    /// Optional type information from the type checker
    type_info: Option<&'a TypeInfo>,
    /// Inferred effects per function DefId
    fn_effects: HashMap<DefId, EffectSet>,
    /// Method effects: (type_name, method_name) -> EffectSet
    method_effects: HashMap<(String, String), EffectSet>,
    /// Higher-order function effects
    hof_effects: HashMap<String, EffectSet>,
    /// Current function's declared effects
    declared: EffectSet,
    /// Current function's inferred effects
    inferred: EffectSet,
    /// Errors
    errors: Vec<EffectError>,
    /// Effect source tracking
    effect_sources: HashMap<String, Span>,
}
```

### Tipos de Erros de Efeito

```rust
// crates/souc/src/effects/inference.rs:144
pub enum EffectErrorKind {
    /// Effect used but not declared
    UndeclaredEffect {
        effect: String,
        source: EffectSource,
    },
    /// Effect not handled
    UnhandledEffect { effect: String },
    /// Effectful operation in pure context
    EffectInPureContext { effect: String },
    /// Higher-order function with effectful closure
    EffectfulClosureArg {
        effect: String,
        hof_name: String,
    },
    /// Refutable pattern may panic
    RefutablePatternPanic { pattern_desc: String },
}

pub enum EffectSource {
    DirectOperation(String),
    MethodCall { receiver_type: String, method: String },
    FunctionCall(String),
    ClosureCall,
    HigherOrderFunction { hof: String, closure_effect: String },
    PatternMatch,
    Unknown,
}
```

### Exemplo de Uso

```sio
// Declaração de efeitos
fn read_file(path: string) -> string with IO {
    // IO é explicitamente declarado
    perform IO.read_file(path)
}

fn process_data(data: &! Data) -> i32 with Mut {
    // Mut é necessário para modificar data
    data.value = data.value + 1
}

// Erro de compilação:
// fn pure_add(a: i32, b: i32) -> i32 {
//     let result = a + b
//     print(result)  // ERROR: IO not declared
// }
```

## 4. Effect Handlers

### Trait HandlerCapability

```rust
// crates/souc/src/effects/handler_capability.rs
pub trait HandlerCapability {
    /// Nome do efeito (e.g., "IO", "Mut")
    fn effect_name(&self) -> &str;

    /// Operações suportadas
    fn operations(&self) -> &[OperationSpec];

    /// Executa uma operação
    fn handle(
        &self,
        operation: &str,
        args: &[Value],
        cont: Continuation,
        state: &mut HandlerState,
    ) -> HandlerResult;
}
```

### IO Handler

```rust
// crates/souc/src/effects/handlers/io_handler.rs:97
pub struct IOHandler {
    _custom_output: Option<std::sync::Arc<std::sync::Mutex<Vec<u8>>>>,
}

// Impacto epistêmico
const READ_CONFIDENCE_FACTOR: f64 = 0.95;   // 5% degradação
const WRITE_CONFIDENCE_FACTOR: f64 = 1.0;     // Sem degradação

// Operações suportadas
fn get_io_operations() -> &'static [OperationSpec] {
    vec![
        OperationSpec::new("print", "unit").with_confidence_factor(1.0),
        OperationSpec::new("println", "unit").with_confidence_factor(1.0),
        OperationSpec::new("read_line", "string").with_confidence_factor(0.95),
        OperationSpec::new("read_file", "string").with_confidence_factor(0.95),
        OperationSpec::new("write_file", "unit").with_confidence_factor(1.0),
    ]
}
```

### Async Handler

```rust
// crates/souc/src/effects/handlers/async_handler.rs:102
pub struct AsyncHandler {
    _private: (),
}

// Operações de async
fn get_async_operations() -> &'static [OperationSpec] {
    vec![
        OperationSpec::new("spawn", "FutureId").with_continuation(),
        OperationSpec::new("await", "Value").with_continuation(),
        OperationSpec::new("yield", "Unit"),
        OperationSpec::new("sleep", "Unit").with_params(vec!["I64"]),
        OperationSpec::new("join", "Array").with_continuation(),
        OperationSpec::new("select", "Tuple").with_continuation(),
        OperationSpec::new("cancel", "Bool"),
        OperationSpec::new("is_ready", "Bool"),
    ]
}
```

### Handler Composition

```rust
// crates/souc/src/effects/handlers/registry.rs
pub struct HandlerRegistry {
    handlers: HashMap<String, Box<dyn HandlerCapability>>,
}

impl HandlerRegistry {
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        registry.add(Box::new(IOHandler::new()));
        registry.add(Box::new(MutHandler::new()));
        registry.add(Box::new(AllocHandler::new()));
        registry.add(Box::new(AsyncHandler::new()));
        // ... etc
        registry
    }
}
```

## 5. Continuations (CPS)

### Continuation Infrastructure

Baseado em **Plotkin & Pretnar (2009)** e **Leijen (2017)**:

```rust
// crates/souc/src/effects/continuation.rs:34
pub type OneShotResumeFn = Box<dyn FnOnce(Value) -> Result<Value, ContinuationError> + 'static>;

pub type MultiShotResumeFn =
    Arc<dyn Fn(Value) -> Result<Value, ContinuationError> + Send + Sync + 'static>;
```

### ResumePoint Enum

```rust
// crates/souc/src/effects/continuation.rs:92
pub enum ResumePoint {
    /// Continuação de uso único (FnOnce)
    InterpreterClosure {
        resume_fn: OneShotResumeFn,
        description: Option<String>,
    },

    /// Continuação multi-shot (Fn, pode ser chamada múltiplas vezes)
    InterpreterMultiShot {
        resume_fn: MultiShotResumeFn,
        description: Option<String>,
    },

    /// Resumo em código JIT
    Jit {
        return_address: usize,
        saved_registers: Vec<u64>,
        stack_snapshot: Vec<u8>,
    },

    /// Stub placeholder
    Stub,
}
```

### Fluxo de Continuation

```
perform Effect.op(args)
        ↓
┌─────────────────────┐
│   Capture Continuation │  → Salva estado atual
└─────────────────────┘
        ↓
┌─────────────────────┐
│   Call Handler      │  → Chama handler.handle()
└─────────────────────┘
        ↓
   Handler Result:
   - Resume(value)      → Retorna valor para continuação
   - Abort(error)       → Cancela execução
   - Suspend(state)     → Salva para execução posterior
```

## 6. SIMD Parallel Dispatch

```rust
// crates/souc/src/effects/simd_dispatch.rs
pub fn simd_perform(
    messages: &[Value],
    handler: &dyn HandlerCapability,
) -> ParallelEffectResult;

// Ao invés de:
perform(msg1, print);
perform(msg2, print);
perform(msg3, print);
// ... até 8

// Usa:
simd_perform([msg1..msg8], print);  // 8 IOs em paralelo
```

## 7. Resilience Patterns

### Retry with Circuit Breaker

```rust
// crates/souc/src/effects/resilience.rs
pub fn with_retry<T>(
    operation: impl Fn() -> Result<T, E>,
    config: RetryConfig,
) -> Result<T, E>;

pub struct RetryConfig {
    max_retries: u32,
    initial_delay: Duration,
    max_delay: Duration,
    backoff_multiplier: f64,
}

pub struct CircuitBreaker {
    state: CircuitState,
    failure_count: u32,
    success_count: u32,
}

pub enum CircuitState {
    Closed,      // Normal operation
    Open,       // Failing, reject immediately
    HalfOpen,   // Testing if service recovered
}
```

## 8. Epistemic Effect Tracking

```rust
// crates/souc/src/effects/epistemic_effects.rs
pub struct EpistemicTracker {
    /// Confidence modifiers por efeito
    confidence_modifiers: HashMap<String, ConfidenceModifier>,
    /// Registry de impactos epistêmicos
    impact_registry: EpistemicImpactRegistry,
}

pub enum ConfidenceModifier {
    /// Degradação fixa (e.g., 5% para leitura de arquivos)
    Fixed(f64),
    /// Degradação baseada em operação
    PerOperation(HashMap<String, f64>),
    /// Sem degradação
    None,
}
```

### Exemplo de Tracking Epistêmico

```sio
fn read_sensor() -> Knowledge<f64> with Sensor, Epistemic {
    // Leitura de sensor: confiança depende da calibração
    let raw = perform Sensor.read("temperature");

    // Garante confiança mínima
    assert raw.confidence >= 0.85;

    raw
}

fn network_request() -> Knowledge<string> with Network, Epistemic {
    // Rede tem alta incerteza
    let response = perform Network.fetch("https://api.example.com");

    // Degradação explícita de confiança
    response.with_confidence(response.confidence * 0.9)
}
```

## 9. Syntax de Efeitos

### Handler Definition

```sio
// Definir um efeito customizado
effect Logger {
    fn log(level: string, message: string) -> ()
}

// Implementar handler
handler MyLogger with Logger {
    fn log(level, message) => {
        print("[", level, "] ", message)
        resume(())
    }
}

// Usar
fn process() -> i32 with Logger {
    perform Logger.log("info", "starting")
    42
}

let result = handle process() with MyLogger {
    Logger.log(level, message) => {
        print("[", level, "] ", message)
        resume(())
    }
}
```

### Effect Composition

```sio
fn complex_operation() -> Result<string, Error> with IO, Mut, Epistemic {
    // Combina múltiplos efeitos
    let data = perform IO.read_file("input.txt");
    perform Mut.update(&mut state, data);
    perform Epistemic.assert_confidence(0.95);
    Ok(data)
}
```

## 10. Fluxo de Type Checking de Efeitos

```
AST com anotações de efeito
        ↓
EffectChecker::check_program()
        ↓
┌─────────────────────────────┐
│ Para cada função:            │
│ 1. Check declared effects    │
│ 2. Infer used effects        │
│ 3. Unify with declaration    │
│ 4. Report mismatches         │
└─────────────────────────────┘
        ↓
HIR com effects verificados  OR  TypeErrors
```

## Métricas

| Componente | Linhas | Descrição |
|------------|--------|-----------|
| `effects/mod.rs` | 100 | Módulo principal e exports |
| `effects/inference.rs` | 1,335 | Effect inference pass |
| `effects/continuation.rs` | 1,229 | CPS infrastructure |
| `effects/handlers/*.rs` | ~4,000 | Handler implementations |
| `effects/handler_capability.rs` | ~300 | Handler trait |

## Próximos Passos

1. **Async/Await Transformation** → [`crates/souc/src/hir/async_transform.rs`](../../crates/souc/src/hir/async_transform.rs)
2. **JIT Resume** → [`crates/souc/src/effects/jit_resume.rs`](../../crates/souc/src/effects/jit_resume.rs)
3. **GPU Dispatch** → [`crates/souc/src/effects/handlers/gpu_handler.rs`](../../crates/souc/src/effects/handlers/gpu_handler.rs)

## Referências

- [`crates/souc/src/effects/mod.rs`](../../crates/souc/src/effects/mod.rs)
- [`crates/souc/src/effects/inference.rs`](../../crates/souc/src/effects/inference.rs)
- [`crates/souc/src/effects/continuation.rs`](../../crates/souc/src/effects/continuation.rs)
- [`crates/souc/src/effects/handlers/`](../../crates/souc/src/effects/handlers/)
- Plotkin & Pretnar (2009): "Handlers of Algebraic Effects"
- Leijen (2017): "Type Directed Compilation of Row-typed Algebraic Effects"
