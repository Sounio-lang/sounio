<!-- docs:meta
topic_id: repo.docs.compiler.epistemic-backend-guide
authority: repo_only
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.epistemic-backend-guide
-->

# Epistemic Backend Guide

> **⚠️ File paths updated 2026-07-11 (doc-reality audit).** This page was written against the retired Rust compiler tree (`crates/`, `compiler/src/*.rs`, `codegen/llvm/`); those files no longer exist — the compiler is self-hosted Sounio (Madaros v0.80.0). The design and concepts below remain accurate, but the epistemic backend/runtime now lives in `self-hosted/` as `.sio` (knowledge-runtime guards in `self-hosted/compiler/knowledge_runtime_guard*.sio`, native emission in `self-hosted/native/`) — not any `epistemic_runtime.rs` / `c_layout.rs` / LLVM backend variant. Do not look for the `.rs` paths below.


## 1. Introduction

Epistemic computing in Sounio represents a paradigm for handling uncertainty and knowledge provenance directly within the computational framework, enabling precise tracking of confidence levels, intervals, and origins in data processing. At its core, the `Knowledge<T>` type serves as a first-class primitive, embedding epistemic metadata alongside the primary value to ensure that computations reflect not just results but also their reliability and context. This approach is particularly vital in scientific computing, where epistemic metadata—such as confidence scores, uncertainty intervals, and provenance trails—facilitates reproducibility, regulatory compliance, and error mitigation in fields like pharmaceuticals, engineering, and data analysis. By making uncertainty explicit, Sounio prevents silent propagation of errors and supports informed decision-making under partial knowledge.

To accommodate diverse use cases, Sounio's epistemic backend operates in three runtime modes: Full, Compact, and Erased. These modes balance fidelity, performance, and resource constraints, allowing developers to select configurations that align with requirements ranging from high-assurance auditing to optimized production deployment.

## 2. Architecture Overview

The architecture of epistemic computing in Sounio centers on the `Knowledge` type, which is structured around four key components:

```
Knowledge[τ, ε, δ, Φ]
│        │  │  │  └── Φ: Functor trace (transformation provenance)
│        │  │  └───── δ: Domain ontology (which ontology validates this)
│        │  └──────── ε: Epistemic status (confidence, revisability, source)
│        └─────────── τ: Context-time (temporal indexing for type evolution)
└──────────────────── Knowledge: First-class epistemic primitive
```

This modular design ensures that epistemic information is tightly coupled with the value, enabling seamless propagation during operations while maintaining type safety.

### Runtime Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    EPISTEMIC BACKEND                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Knowledge<T> Source Code                                    │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────┐                                         │
│  │  Type Checker   │  Check epistemic constraints            │
│  └────────┬────────┘                                         │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                         │
│  │  Mode Selection │  Full / Compact / Erased               │
│  └────────┬────────┘                                         │
│           │                                                  │
│           ├─────────────┬─────────────┬──────────────┐       │
│           ▼             ▼             ▼              ▼       │
│    ┌─────────┐   ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│    │  Full   │   │ Compact  │  │  Erased  │  │   LLVM   │   │
│    │ Runtime │   │ Runtime  │  │  (zero)  │  │  Backend │   │
│    │ 41 bytes│   │ 14 bytes │  │  8 bytes │  │          │   │
│    └────┬────┘   └─────┬────┘  └─────┬────┘  └─────┬────┘   │
│         │              │             │             │        │
│         ▼              ▼             ▼             ▼        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  C Runtime Functions (epistemic_runtime.rs)          │   │
│  │  - sounio_epistemic_add_full/compact                 │   │
│  │  - sounio_epistemic_mul_full/compact                 │   │
│  │  - sounio_epistemic_meet/join                        │   │
│  │  System V ABI / x86-64 calling convention            │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

A critical aspect of the architecture is its C-compatible layout, which facilitates foreign function interfaces (FFI) by adhering to standard C struct conventions, including proper alignment and padding.

## 3. Three Runtime Modes

Sounio's epistemic backend supports three distinct runtime modes, each tailored to different priorities in terms of metadata retention, memory usage, and performance. These modes—Full, Compact, and Erased—allow for flexible deployment, with mode selection influencing how epistemic metadata is stored and accessed.

### 3.1 Full Mode (KnowledgeFull)

In Full mode, the `Knowledge<f64>` type employs a comprehensive memory layout: [value, confidence, conf_lower, conf_upper, provenance_id, timestamp, revisable]. This structure captures the full spectrum of epistemic details, including precise confidence bounds, provenance identifiers for tracing origins, timestamps for temporal context, and a revisable flag for update tracking. The resulting size is 41 bytes per `Knowledge<f64>`, reflecting the richness of the metadata.

This mode is ideal for scenarios demanding maximum transparency and verifiability, such as FDA compliance in pharmaceutical modeling, scientific publications requiring detailed auditing, or forensic analysis in research pipelines. Performance-wise, it introduces approximately a 2x slowdown compared to bare types like plain `f64`, primarily due to the additional loads, stores, and computations for metadata propagation during operations.

### 3.2 Compact Mode (KnowledgeCompact)

Compact mode optimizes for space and speed while retaining essential epistemic information, using a streamlined memory layout: [value, quantized_conf (u16), provenance_hash (u32)]. Here, confidence is quantized into a 16-bit unsigned integer (0-65535), mapping linearly to a 0.0-1.0 scale, and provenance is reduced to a 32-bit hash for lightweight identification. This configuration yields a compact 14 bytes per `Knowledge<f64>`.

Suitable for production systems where storage is limited—such as embedded devices, large-scale simulations, or cloud-based analytics—Compact mode strikes a balance between utility and efficiency. Its performance characteristics show about a 1.3x slowdown versus bare types, as the reduced metadata minimizes branching and arithmetic overhead, making it viable for high-throughput applications without sacrificing core uncertainty tracking.

### 3.3 Erased Mode

Erased mode strips away epistemic metadata entirely during compilation or runtime, resulting in a memory layout that consists solely of the value (f64), matching the 8-byte size of a standard `f64`. Metadata, if needed, is tracked separately through external mechanisms or omitted altogether, effectively erasing the epistemic layer for the core computation.

This mode is best employed in release builds or performance-critical sections of code, such as real-time processing or numerical kernels where overhead must be negligible. It incurs zero additional performance cost relative to bare types, ensuring seamless integration into optimized workflows, though at the expense of built-in uncertainty awareness.

## 4. GUM Propagation Rules (Guide to the Expression of Uncertainty in Measurement)

Propagation rules in Sounio's epistemic backend draw from the Guide to the Expression of Uncertainty in Measurement (GUM), providing standardized methods for combining epistemic metadata during arithmetic and fusion operations. These rules ensure that uncertainty is propagated conservatively and mathematically grounded, maintaining the integrity of scientific computations.

### 4.1 Addition/Subtraction

For addition and subtraction, the value propagates straightforwardly as a ± b. Confidence is computed using the geometric mean, √(conf_a * conf_b), which reflects the joint reliability of inputs. The uncertainty interval width follows the quadrature formula √(w_a² + w_b²), where w represents half the interval width ((upper - lower)/2), accounting for independent uncertainties. This derivation aligns directly with GUM principles for uncorrelated linear combinations, ensuring that propagated intervals conservatively bound potential errors.

### 4.2 Multiplication/Division

In multiplication and division, the result value is a * b or a / b, respectively. Relative uncertainty is derived as √((w_a / a)² + (w_b / b)²) multiplied by the absolute value of the result, capturing how percentage errors compound in multiplicative operations. Special handling is included for division by zero, typically raising an epistemic error or defaulting to a zero-confidence state. The mathematical derivation follows GUM's approximation for relative uncertainties in products and quotients, providing a first-order Taylor expansion-based estimate suitable for small uncertainties.

### 4.3 Meet Operation (Skeptical)

The meet operation adopts a skeptical fusion strategy, selecting the value associated with the minimum confidence among inputs. This conservative approach prioritizes the most uncertain perspective, avoiding over-optimism in combined results. It is particularly useful in safety-critical systems, such as fault-tolerant control or risk assessment, where underestimating uncertainty could lead to hazardous decisions.

### 4.4 Join Operation (Credulous)

Conversely, the join operation employs a credulous strategy via inverse-variance weighted averaging, where weights w_i = 1 / variance_i and variance approximates (width/2)². The combined variance is then 1 / (w_a + w_b), yielding a fused value that favors more precise inputs. This method supports applications like sensor fusion in IoT devices or measurement averaging in experimental setups, promoting informed aggregation while respecting relative reliabilities.

## 5. Backend Implementation Details

The backend implementation in Sounio is engineered for robustness and extensibility, with components handling runtime operations, type layouts, and effect integration to support epistemic computing across modes.

### 5.1 C Runtime Functions (epistemic_runtime.rs)

Core functionality is exposed through C runtime functions with C-compatible signatures:

```c
// Full layout operations (41 bytes)
void sounio_epistemic_add_full(
    const KnowledgeFull* a,    // RDI: first operand
    const KnowledgeFull* b,    // RSI: second operand
    KnowledgeFull* result      // RDX: output
);

void sounio_epistemic_mul_full(
    const KnowledgeFull* a,
    const KnowledgeFull* b,
    KnowledgeFull* result
);

// Compact layout operations (14 bytes)
void sounio_epistemic_add_compact(
    const KnowledgeCompact* a,
    const KnowledgeCompact* b,
    KnowledgeCompact* result
);

// Fusion operations
void sounio_epistemic_meet(
    const KnowledgeFull* a,
    const KnowledgeFull* b,
    KnowledgeFull* result      // Takes minimum confidence
);

void sounio_epistemic_join(
    const KnowledgeFull* a,
    const KnowledgeFull* b,
    KnowledgeFull* result      // Inverse-variance weighted
);

// Metadata extraction
double sounio_epistemic_extract_confidence(
    const KnowledgeFull* knowledge
);
```

**System V ABI Calling Convention (x86-64):**
- First 3 pointer args in: RDI, RSI, RDX
- Return values in: RAX (integers), XMM0 (floats)
- Caller-saved: RAX, RCX, RDX, RSI, RDI, R8-R11
- Callee-saved: RBX, RBP, R12-R15

**Implementation Example (from epistemic_runtime.rs):**
```rust
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sounio_epistemic_add_full(
    a: *const KnowledgeFull,
    b: *const KnowledgeFull,
    result: *mut KnowledgeFull,
) {
    if a.is_null() || b.is_null() || result.is_null() {
        return;
    }

    unsafe {
        let a_val = (*a).value;
        let a_conf = (*a).confidence;
        let a_lower = (*a).conf_lower;
        let a_upper = (*a).conf_upper;
        let b_val = (*b).value;
        let b_conf = (*b).confidence;
        let b_lower = (*b).conf_lower;
        let b_upper = (*b).conf_upper;

        // Value: a + b
        (*result).value = a_val + b_val;

        // Confidence: geometric mean (conservative)
        (*result).confidence = (a_conf * b_conf).sqrt();

        // GUM propagation: combined width = sqrt(w_a² + w_b²)
        let a_half_width = (a_upper - a_lower) / 2.0;
        let b_half_width = (b_upper - b_lower) / 2.0;
        let combined_half_width = (a_half_width.powi(2) + b_half_width.powi(2)).sqrt();

        (*result).conf_lower = (*result).value - combined_half_width;
        (*result).conf_upper = (*result).value + combined_half_width;

        // Provenance: XOR combine
        (*result).provenance_id = (*a).provenance_id ^ (*b).provenance_id;

        // Timestamp: most recent
        (*result).timestamp = (*a).timestamp.max((*b).timestamp);

        // Revisable if either input is revisable
        (*result).revisable = (*a).revisable | (*b).revisable;
    }
}
```

### 5.2 Type Layout Engine (c_layout.rs)

The type layout engine computes sizes for `Knowledge` variants, enforces alignment requirements per platform ABI, and applies struct padding rules to prevent misalignment penalties. It also performs C compatibility checks, verifying that layouts match expected C structs for FFI safety, which is essential for embedding Sounio in mixed-language environments.

### 5.3 Effect Dispatch Integration

Epistemic effects are dispatched through Sounio's effect system, where operations on `Knowledge` trigger handlers that apply GUM rules transparently. This integration allows epistemic computations to compose with other effects, such as I/O or concurrency, without manual intervention, streamlining development in effectful codebases.

## 6. Performance Characteristics

Understanding performance is key to leveraging Sounio's epistemic backend effectively, as modes and operations introduce varying overheads in memory, computation, and optimization potential.

### 6.1 Memory Overhead

| Type                    | Size (bytes) | Alignment | Cache Lines (64B) | Ratio vs f64 |
|-------------------------|-------------|-----------|-------------------|--------------|
| `f64` (bare)            | 8           | 8         | 0.125             | 1.0x         |
| `Knowledge<f64>` Erased | 8           | 8         | 0.125             | 1.0x         |
| `Knowledge<f64>` Compact| 14          | 8         | 0.219             | 1.75x        |
| `Knowledge<f64>` Full   | 41          | 8         | 0.641             | 5.125x       |

**Cache Impact Analysis:**
- **Full mode**: 5.1x memory overhead can cause cache thrashing in tight loops. An array of 1000 Knowledge<f64> values occupies ~40KB vs ~8KB for bare f64.
- **Compact mode**: 1.75x overhead is tolerable for most workloads. Same array uses ~14KB.
- **Erased mode**: Zero overhead - identical to bare types.

**SIMD Considerations:**
- Compact and Erased modes allow vectorization of the inner `f64` values using AVX2/AVX-512
- Full mode requires scalar processing due to non-power-of-2 size
- For SIMD-friendly epistemic computing, use Compact mode with manual metadata batching

### 6.2 Computational Overhead

Operations in Full mode incur cycle estimates of 20-50 additional instructions per binary operation, depending on propagation complexity. Compact mode reduces this to 10-20 cycles, and Erased to zero. Power consumption metrics from thermal tracking indicate modest increases (e.g., 5-15% in Full mode under load), with thermal degradation potentially lowering confidence scores based on sustained computation heat.

### 6.3 Optimization Strategies

Select modes based on context: Full for compliance-heavy tasks, Compact for balanced production, and Erased for hotspots. Lazy evaluation defers provenance resolution until queried, confidence thresholding skips low-impact updates, and batch operations vectorize propagations to amortize costs across arrays.

## 7. Usage Examples

Practical usage of epistemic computing in Sounio is demonstrated through code snippets that illustrate automatic propagation and fusion.

### 7.1 Basic Arithmetic
```sio
let a: Knowledge<f64> = measure(5.0, uncertainty: 0.1)
let b: Knowledge<f64> = measure(3.0, uncertainty: 0.05)
let c = a + b  // Propagates confidence automatically
```

### 7.2 PK/PD Calculations
```sio
let dose: Knowledge<mg> = measure(500.0, uncertainty: 5.0)
let volume: Knowledge<mL> = measure(10.0, uncertainty: 0.2)
let concentration = dose / volume  // Knowledge<mg/mL>
```

### 7.3 Sensor Fusion
```sio
let sensor1: Knowledge<f64> = measure_temp(36.5, conf: 0.9)
let sensor2: Knowledge<f64> = measure_temp(36.7, conf: 0.85)
let fused = epistemic_join(sensor1, sensor2)  // Inverse-variance weighted
```

### 7.4 Safety-Critical Validation
```sio
let measured: Knowledge<mg> = lab_assay(sample)
let expected: Knowledge<mg> = theoretical_dose()
let verified = epistemic_meet(measured, expected)  // Conservative estimate
```

## 8. Debugging and Introspection

Sounio provides tools for inspecting epistemic state, aiding in validation and troubleshooting during development and deployment.

### 8.1 Extracting Metadata
```sio
let conf = measured.confidence()
let (lower, upper) = measured.interval()
let prov = measured.provenance()
```

### 8.2 Confidence Thresholds
```sio
assert_confidence(result, min: 0.95)  // FDA compliance check
```

### 8.3 Thermal Degradation Tracking

Computation cycles can degrade confidence through thermal effects, modeled via Arrhenius parameters that quantify temperature-induced error rates. Developers can view degradation history by querying provenance logs, enabling adjustments for prolonged runs in warm environments.

## 9. Integration with Other Systems

Epistemic computing extends beyond basics by integrating with Sounio's type system features, enhancing expressiveness in constrained domains.

### 9.1 Refinement Types

The `EpistemicRefinedValue` combines epistemic metadata with logical refinements, such as `PositiveEpistemic` for non-negative values or `BoundedEpistemic` for range-constrained knowledge. This fusion enforces both uncertainty and domain invariants, as in safety checks where values must remain positive post-operation.

### 9.2 Units of Measure

`QuantifiedKnowledge<T, Unit>` pairs epistemic tracking with dimensional analysis, propagating confidence in typed units like `Knowledge<mg/L>`. For instance, division of `Knowledge<mg>` by `Knowledge<L>` yields a dimensionally correct result with GUM-based uncertainty.

### 9.3 Beta-Knowledge (Advanced)

Advanced users can employ beta-knowledge for full-distribution epistemic computing, incorporating beta priors to enable Bayesian inference. This supports active inference metrics, such as updating beliefs from sequential observations in probabilistic models.

## 10. Best Practices

Adopting best practices ensures reliable and efficient use of epistemic computing, from mode selection to metadata management.

### 10.1 Choosing Runtime Mode

A decision tree guides mode choice: prioritize Full for audit needs, Compact for resource limits, and Erased for speed. Trade-offs involve compliance versus latency; migration between modes uses compile-time flags to reconfigure without code changes.

### 10.2 Confidence Management

Initial confidence should reflect measurement precision, avoiding absolutes like 1.0. Post-operation, interpret results via propagated means and intervals, applying floors (e.g., 0.0 for failures) or ceilings to bound optimism.

### 10.3 Provenance Tracking

Track full provenance only when auditing is required; otherwise, use hash-based summaries. For comprehensive trails, employ Merkle DAG structures to link operations, facilitating efficient verification in distributed systems.

## 11. Common Pitfalls

Awareness of pitfalls helps avoid subtle errors in epistemic workflows.

### 11.1 Overconfidence

Avoid assigning confidence=1.0 to empirical measurements, as it ignores inherent noise; always incorporate systematic errors through calibration validation to prevent downstream over-reliance.

### 11.2 Underestimating Uncertainty

Account for correlations in variables, as assuming independence inflates confidence; address non-independent measurements and hidden biases via sensitivity analysis or expanded GUM models.

### 11.3 Performance Issues

Excessive provenance tracking bloats memory; opt for Compact mode in loops. Full mode can cause cache thrashing in arrays—profile and switch to Erased for non-critical paths.

## 12. Future Directions

Looking ahead, enhancements to Sounio's epistemic backend include GPU acceleration for parallel propagation in large-scale simulations, distributed computing to synchronize uncertainty across nodes, integration of quantum uncertainty models for hybrid classical-quantum workflows, and real-time adaptation of confidence based on dynamic environmental factors. These developments aim to broaden applicability in emerging computational paradigms.
