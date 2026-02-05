# confidence_metadata.sio - Compiler Epistemic Confidence Metadata

**Target**: 250 lines of Sounio code
**LLM**: minimax 2.1 (cost-effective for structured boilerplate)
**Purpose**: Metadata types for tracking epistemic confidence during type checking

## Module Overview

This module defines the metadata types that attach to AST nodes, types, and unification
constraints during type checking. It bridges the runtime `Knowledge<T>` system
(stdlib/epistemic/knowledge.sio) with compile-time type checking.

## Dependencies

```sio
use core::{Option, Result, Vec}
use epistemic::knowledge::{BetaConfidence, Source, Provenance}
use compiler::types::{TypeId, TypeVar}
use compiler::ast::{NodeId, Span}
```

## Data Types

### 1. ConfidenceLevel (enum, ~20 lines)

Discrete confidence levels for type inference decisions:

```sio
/// Confidence in a type inference decision
pub enum ConfidenceLevel {
    /// Mathematically proven (e.g., literal types)
    Certain,

    /// High confidence from strong evidence (e.g., explicit annotations)
    High,

    /// Medium confidence from inference (e.g., unification with constraints)
    Medium,

    /// Low confidence from weak evidence (e.g., defaulting)
    Low,

    /// Uncertain / speculative (e.g., inference across complex boundaries)
    Uncertain,
}
```

Methods:
- `to_beta() -> BetaConfidence` - Convert to Beta distribution
- `from_beta(BetaConfidence) -> ConfidenceLevel` - Discretize from Beta
- `combine(ConfidenceLevel) -> ConfidenceLevel` - Min of two levels

### 2. InferenceSource (enum, ~30 lines)

Where did a type inference decision come from?

```sio
/// Source of a type inference
pub enum InferenceSource {
    /// Explicit type annotation by user
    Annotation { node: NodeId, span: Span },

    /// Literal value (e.g., `42` → i32)
    Literal { node: NodeId, value: string },

    /// Unification constraint
    Unification { lhs: TypeId, rhs: TypeId },

    /// Function return type propagated to call site
    ReturnType { fn_id: NodeId },

    /// Default inference (e.g., integer literals → i32)
    Default { reason: string },

    /// Refinement type constraint
    Refinement { constraint: string },

    /// Effect system constraint
    Effect { effect: string },

    /// External (stdlib, ontology)
    External { module: string },
}
```

Methods:
- `confidence_level() -> ConfidenceLevel` - Infer confidence from source
- `span() -> Option<Span>` - Extract source location if available

### 3. TypeConfidenceMetadata (struct, ~40 lines)

Epistemic metadata attached to each type during inference.

```sio
/// Confidence metadata for a type
pub struct TypeConfidenceMetadata {
    /// Type this metadata is attached to
    type_id: TypeId,

    /// Confidence level
    confidence: ConfidenceLevel,

    /// Beta posterior (continuous confidence)
    beta: BetaConfidence,

    /// Where this type inference came from
    source: InferenceSource,

    /// Provenance chain through inference steps
    provenance: Vec<InferenceStep>,

    /// Is this type user-annotated? (highest trust)
    is_annotated: bool,
}
```

Methods:
- `new(TypeId, InferenceSource) -> TypeConfidenceMetadata` - Create from source
- `certain(TypeId) -> TypeConfidenceMetadata` - Create certain metadata (literals)
- `annotated(TypeId, Span) -> TypeConfidenceMetadata` - From user annotation
- `inferred(TypeId, InferenceSource) -> TypeConfidenceMetadata` - From inference
- `update_confidence(ConfidenceLevel)` - Update confidence level
- `add_step(InferenceStep)` - Add provenance step
- `decay(f64)` - Decay confidence through transformation
- `is_certain() -> bool` - Check if confidence is Certain
- `is_reliable() -> bool` - Check if >= High

### 4. InferenceStep (struct, ~25 lines)

Single step in type inference provenance chain.

```sio
/// Single step in type inference
pub struct InferenceStep {
    /// What operation was performed
    operation: string,

    /// Source/destination types
    from_type: Option<TypeId>,
    to_type: TypeId,

    /// Confidence decay from this step
    decay_factor: f64,

    /// Why this step was taken
    reason: string,
}
```

Methods:
- `new(string, TypeId, string) -> InferenceStep`
- `with_from(TypeId) -> InferenceStep`

### 5. ConstraintConfidence (struct, ~30 lines)

Confidence in a unification constraint.

```sio
/// Confidence in a unification constraint
pub struct ConstraintConfidence {
    /// LHS type
    lhs: TypeId,

    /// RHS type
    rhs: TypeId,

    /// Confidence that lhs <: rhs
    confidence: BetaConfidence,

    /// Source of this constraint
    source: InferenceSource,

    /// Is this constraint from user code or inferred?
    is_explicit: bool,
}
```

Methods:
- `new(TypeId, TypeId, InferenceSource) -> ConstraintConfidence`
- `explicit(TypeId, TypeId, Span) -> ConstraintConfidence`
- `implicit(TypeId, TypeId) -> ConstraintConfidence`
- `combine(ConstraintConfidence) -> ConstraintConfidence` - Merge constraints
- `is_reliable() -> bool` - Check if confidence > threshold

### 6. ConfidenceContext (struct, ~40 lines)

Global context for tracking confidence across type checking.

```sio
/// Global confidence tracking context
pub struct ConfidenceContext {
    /// Metadata for each type
    type_metadata: HashMap<TypeId, TypeConfidenceMetadata>,

    /// Constraint confidences
    constraints: Vec<ConstraintConfidence>,

    /// Default confidence for untracked types
    default_confidence: ConfidenceLevel,

    /// Decay factor for inference steps
    default_decay: f64,
}
```

Methods:
- `new() -> ConfidenceContext`
- `register_type(TypeId, TypeConfidenceMetadata)` - Track new type
- `get_confidence(TypeId) -> Option<&TypeConfidenceMetadata>` - Query confidence
- `add_constraint(ConstraintConfidence)` - Add constraint
- `get_constraints_for(TypeId) -> Vec<&ConstraintConfidence>` - Query constraints
- `propagate_confidence(TypeId, TypeId)` - Propagate from source to dest
- `decay_all(f64)` - Global confidence decay

### 7. Helper Functions (~35 lines)

```sio
/// Convert source to initial confidence level
fn source_to_confidence(source: &InferenceSource) -> ConfidenceLevel

/// Combine multiple confidence levels (minimum)
fn combine_confidences(levels: &[ConfidenceLevel]) -> ConfidenceLevel

/// Decay confidence level through transformation
fn decay_confidence(level: ConfidenceLevel, factor: f64) -> ConfidenceLevel

/// Check if confidence is above reliability threshold
fn is_reliable_confidence(level: ConfidenceLevel) -> bool
```

## Design Constraints

1. **Efficiency**: Metadata should be lightweight (no expensive clones)
2. **Immutability**: Most operations return new values, not mutate
3. **Integration**: Must work with existing `TypeChecker` and `Substitution`
4. **Debugging**: All types should implement `Debug` trait

## Testing (~30 lines)

```sio
fn test_confidence_level_conversions()
fn test_metadata_creation()
fn test_provenance_chain()
fn test_constraint_combination()
```

## Notes for Code Generator

- Use `&!` for mutable references, NOT `&mut`
- Use `var` for mutable variables, NOT `let mut`
- No Rust macros (no `println!()`, use function calls)
- No attributes (no `#[derive()]`, no `#[test]`)
- Preserve Sounio style: explicit, verbose, epistemic-first
- Reference stdlib/epistemic/knowledge.sio for epistemic patterns
