---
title: "Epistemic Types"
description: "Knowledge<T> implementation: confidence tracking, provenance, ontology binding, and GUM-compliant uncertainty propagation."
---

## Epistemic Types: Knowledge\<T\>

Sounio's epistemic type system provides first-class support for **uncertainty quantification**, **provenance tracking**, and **ontology binding**. Every measurement or derived value can carry metadata about its confidence, origin, and scientific meaning.

### The Knowledge Type

**File**: `compiler/src/epistemic/knowledge.rs:35-57`

```rust
pub struct Knowledge {
    content: Box<Type>,        // tau: wrapped value type
    temporal: ContextTime,     // tau: temporal context
    epistemic: EpistemicStatus, // epsilon: confidence + revisability
    domain: OntologyBinding,   // delta: ontology term binding
    provenance: Provenance,    // Phi: transformation history
    span: Span,
}
```

The Knowledge type is a **4-tuple** `Knowledge<T, epsilon, delta, Phi>` where:
- **T** is the wrapped value type (e.g., `f64`, `mg/L`)
- **epsilon** is the epistemic status (confidence interval + revisability)
- **delta** is the ontology binding (scientific term reference)
- **Phi** is the provenance functor trace (transformation history)

### Epistemic Status

**File**: `compiler/src/epistemic/confidence.rs:22-34`

```rust
pub struct EpistemicStatus {
    confidence: Confidence,      // epsilon in [0.0, 1.0]
    revisability: Revisability,  // Axiom or Revisable
    source: Source,              // Origin classification
    evidence: Vec<Evidence>,     // Supporting evidence chain
}
```

**Confidence** (line 136):
- `value: f64` --- Point estimate in [0.0, 1.0]
- `lower: Option<f64>` --- Lower bound of confidence interval
- `upper: Option<f64>` --- Upper bound of confidence interval
- Methods: `propagate()`, `with_interval()`

**Revisability** distinguishes:
- `NonRevisable` --- Axioms and definitions (confidence permanently 1.0)
- `Revisable { conditions }` --- Empirical values that may change

**Source** classifies the origin:
- `Axiom` --- Mathematical or definitional truth
- `Measurement { instrument, protocol, timestamp }` --- Physical measurement
- `Derivation(String)` --- Computed from other values
- `Transformation { original, via }` --- Processed data
- `External { uri }` --- External data source

### Factory Methods

**File**: `compiler/src/epistemic/confidence.rs:40-87`

| Constructor | Confidence | Revisability | Use Case |
|-------------|-----------|--------------|----------|
| `axiomatic()` | 1.0 | NonRevisable | Constants, definitions |
| `empirical(conf, source)` | Variable | Revisable | Measurements |
| `derived(dependencies)` | Product of deps | Revisable | Computed values |

### Provenance Tracking

**File**: `compiler/src/epistemic/provenance.rs:26-35`

```rust
pub struct Provenance {
    trace: FunctorTrace,         // Ordered transformations
    origin: Origin,              // Source point
    integrity_hash: Option<String>, // Tamper detection
}

pub struct FunctorTrace {
    steps: VecDeque<Transformation>,
    max_length: usize,           // Compression limit
}
```

Each **Transformation** records:
- `name: String` --- Operation name
- `kind: TransformationKind` --- Category of operation
- `confidence_factor: f64` --- Multiplicative confidence decay

Key methods:
- `extend()` --- Append a transformation step
- `total_confidence_factor()` --- Product through the entire chain
- `depth()` --- Number of transformations from origin

### Ontology Binding (4-Layer Hierarchy)

**File**: `compiler/src/epistemic/knowledge.rs:59-175`

Each Knowledge value can be bound to a scientific ontology term:

```rust
pub struct OntologyBinding {
    ontology: OntologyRef,       // Which ontology (BFO, ChEBI, etc.)
    term: TermId,                // Specific term within ontology
    constraint: Option<Constraint>, // Optional refinement
}
```

The ontology hierarchy has **4 layers**:

| Layer | Terms | Source | Example |
|-------|-------|--------|---------|
| **L1: Primitive** | ~850 | Compiled into binary | BFO (36 classes), RO (600 relations) |
| **L2: Foundation** | ~8K | Shipped with stdlib | PATO, UO, IAO, Schema.org, FHIR |
| **L3: Domain** | ~500K | SQLite database | ChEBI, GO, DOID, HP |
| **L4: Federated** | ~15M | Network APIs | BioPortal, OLS4 |

### Confidence Propagation Rules

Following **GUM (Guide to the Expression of Uncertainty in Measurement)**, ISO/IEC Guide 98-3:2008:

**Addition/Subtraction**:
```
u_c = sqrt(u_a^2 + u_b^2)
```

**Multiplication/Division**:
```
u_rel_c = sqrt(u_rel_a^2 + u_rel_b^2)
```

**General (Taylor expansion)**:
```
u_c^2 = sum_i (df/dx_i)^2 * u_i^2 + 2 * sum_{i<j} (df/dx_i)(df/dx_j) * u_i * u_j * r_ij
```

### GPU Shadow Registers

On GPU (`codegen/gpu/ptx_epistemic_bridge.rs`), epistemic values are tracked via shadow registers:

```
Knowledge[f64, epsilon] --> {
    r_value: f64,       // Main register: actual value
    r_epsilon: f32,     // Shadow register: uncertainty bound
    r_valid: pred,      // Predicate register: validity flag
    r_prov: u64,        // Provenance register: bit-packed lineage
}
```

Operations propagate uncertainty automatically:
- `emit_epistemic_add()` --- Quadrature addition of uncertainties
- `emit_epistemic_mul()` --- Relative uncertainty combination
- `emit_epistemic_div()` --- Division uncertainty propagation

### HLIR Encoding

At the HLIR level (`hlir/ir.rs:129`):

```rust
HlirType::Knowledge {
    inner: Box<HlirType>,     // Wrapped type
    mode: KnowledgeMode,       // Full, ConfidenceOnly, ProvenanceOnly
    epsilon_bound: Option<f64>, // Static confidence bound
    provenance_id: Option<u64>, // Provenance chain reference
}
```

### Advanced Features

**File**: `compiler/src/epistemic/mod.rs:40-127`

| Feature | Description |
|---------|-------------|
| **Bayesian Fusion** | Dempster-Shafer evidence combination |
| **Beta-Knowledge** | Full distribution epistemic (not just point estimates) |
| **Firewall** | Confidence boundaries between subsystems |
| **Merkle Provenance** | Immutable audit trails with hash chains |
| **Time-Travel Debugging** | Epistemic state snapshots for replay |
| **KEC Auto-Selection** | Optimal uncertainty model selection |
| **Promotion Lattice** | Uncertainty level hierarchy for type coercions |

### Runtime Representation

**File**: `compiler/src/epistemic/composition/knowledge.rs:91-100`

```rust
pub struct EpistemicValue<T> {
    value: T,
    confidence: ConfidenceValue,
    ontology: HashSet<OntologyRef>,
    provenance: ProvenanceNode,
}
```
