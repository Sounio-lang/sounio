# type_confidence.sio - Type-Level Confidence Tracking

**Target**: 320 lines of Sounio code
**LLM**: Grok-4.1-fast (balanced speed/quality for logic-heavy code)
**Purpose**: Track and compute confidence for type-level operations (substitution, unification, generalization)

## Module Overview

This module provides the type system integration layer for epistemic confidence.
It extends the type checker's core operations (unify, substitute, instantiate) with
confidence tracking and propagation.

## Dependencies

```sio
use core::{Option, Result, Vec, HashMap}
use compiler::types::{Type, TypeId, TypeVar, Substitution}
use compiler::epistemic::confidence_metadata::{
    TypeConfidenceMetadata, ConfidenceLevel, InferenceSource,
    ConfidenceContext, InferenceStep
}
use epistemic::knowledge::BetaConfidence
```

## Data Types

### 1. TypeWithConfidence (struct, ~30 lines)

A type bundled with its confidence metadata.

```sio
/// Type with epistemic metadata
pub struct TypeWithConfidence {
    /// The type
    ty: Type,

    /// Confidence metadata
    meta: TypeConfidenceMetadata,
}
```

Methods:
- `new(Type, TypeConfidenceMetadata) -> TypeWithConfidence`
- `certain(Type) -> TypeWithConfidence` - Create with Certain confidence
- `inferred(Type, InferenceSource) -> TypeWithConfidence`
- `get_type() -> &Type`
- `get_confidence() -> &TypeConfidenceMetadata`
- `map_type<F>(F) -> TypeWithConfidence where F: fn(Type) -> Type`
- `decay(f64) -> TypeWithConfidence`

### 2. SubstitutionConfidence (struct, ~40 lines)

Track confidence through type variable substitution.

```sio
/// Confidence tracking for substitutions
pub struct SubstitutionConfidence {
    /// Base substitution [α → τ]
    subst: Substitution,

    /// Confidence for each substitution binding
    bindings: HashMap<TypeVar, TypeConfidenceMetadata>,

    /// Overall confidence in this substitution
    overall_confidence: ConfidenceLevel,
}
```

Methods:
- `new(Substitution) -> SubstitutionConfidence`
- `add_binding(TypeVar, Type, TypeConfidenceMetadata)`
- `get_binding_confidence(TypeVar) -> Option<&TypeConfidenceMetadata>`
- `apply_to_type(Type) -> TypeWithConfidence` - Apply subst with confidence
- `compose(SubstitutionConfidence) -> SubstitutionConfidence` - θ₁ ∘ θ₂
- `overall_confidence() -> ConfidenceLevel`

### 3. UnificationConfidence (struct, ~50 lines)

Track confidence through unification.

```sio
/// Confidence tracking for unification
pub struct UnificationConfidence {
    /// Unification result: MGU θ such that θ(τ₁) = θ(τ₂)
    mgu: Option<Substitution>,

    /// Confidence in the unification
    confidence: TypeConfidenceMetadata,

    /// Constraints generated during unification
    constraints: Vec<(Type, Type, ConfidenceLevel)>,

    /// Steps taken during unification
    steps: Vec<UnificationStep>,
}

/// Single unification step
pub struct UnificationStep {
    lhs: Type,
    rhs: Type,
    operation: string,
    confidence_decay: f64,
}
```

Methods for `UnificationConfidence`:
- `success(Substitution, TypeConfidenceMetadata) -> UnificationConfidence`
- `failure() -> UnificationConfidence`
- `is_success() -> bool`
- `get_mgu() -> Option<&Substitution>`
- `get_confidence() -> &TypeConfidenceMetadata`
- `add_step(UnificationStep)`
- `add_constraint(Type, Type, ConfidenceLevel)`

Methods for `UnificationStep`:
- `new(Type, Type, string, f64) -> UnificationStep`

### 4. GeneralizationConfidence (struct, ~35 lines)

Track confidence through type generalization (∀-introduction).

```sio
/// Confidence for type scheme generalization
pub struct GeneralizationConfidence {
    /// Original monotype
    mono: Type,

    /// Generalized polytype (type scheme)
    poly: Type,

    /// Quantified type variables
    quantified: Vec<TypeVar>,

    /// Confidence in generalization
    confidence: TypeConfidenceMetadata,
}
```

Methods:
- `new(Type, Type, Vec<TypeVar>, TypeConfidenceMetadata) -> GeneralizationConfidence`
- `is_polymorphic() -> bool`
- `quantifier_count() -> usize`
- `decay_confidence(f64)`

### 5. InstantiationConfidence (struct, ~35 lines)

Track confidence through type scheme instantiation (∀-elimination).

```sio
/// Confidence for type scheme instantiation
pub struct InstantiationConfidence {
    /// Original polytype
    poly: Type,

    /// Instantiated monotype
    mono: Type,

    /// Fresh variables introduced
    fresh_vars: HashMap<TypeVar, TypeVar>,

    /// Confidence in instantiation
    confidence: TypeConfidenceMetadata,
}
```

Methods:
- `new(Type, Type, HashMap<TypeVar, TypeVar>, TypeConfidenceMetadata) -> InstantiationConfidence`
- `get_substitution() -> Substitution` - Extract [α → β] for fresh vars

### 6. ConfidenceDecay (struct, ~25 lines)

Configuration for confidence decay through type operations.

```sio
/// Decay factors for type operations
pub struct ConfidenceDecay {
    /// Decay for unification
    unify_decay: f64,

    /// Decay for substitution
    subst_decay: f64,

    /// Decay for generalization
    gen_decay: f64,

    /// Decay for instantiation
    inst_decay: f64,
}
```

Methods:
- `default() -> ConfidenceDecay` - Standard decay factors
- `conservative() -> ConfidenceDecay` - Lower decay (trust inference more)
- `aggressive() -> ConfidenceDecay` - Higher decay (trust less)

## Core Functions

### Unification with Confidence (~40 lines)

```sio
/// Unify two types with confidence tracking
pub fn unify_with_confidence(
    lhs: &TypeWithConfidence,
    rhs: &TypeWithConfidence,
    ctx: &!ConfidenceContext,
    decay: &ConfidenceDecay,
) -> Result<UnificationConfidence, string>
```

Implementation sketch:
1. Unify `lhs.ty` and `rhs.ty` to get MGU θ
2. Combine confidences from lhs and rhs
3. Apply decay factor
4. Create UnificationConfidence with steps

### Substitution with Confidence (~30 lines)

```sio
/// Apply substitution with confidence tracking
pub fn substitute_with_confidence(
    ty: &TypeWithConfidence,
    subst: &SubstitutionConfidence,
    decay: &ConfidenceDecay,
) -> TypeWithConfidence
```

Implementation:
1. Apply subst.subst to ty.ty
2. Combine confidence from ty and affected bindings in subst
3. Apply decay
4. Return new TypeWithConfidence

### Generalization with Confidence (~25 lines)

```sio
/// Generalize type with confidence tracking
pub fn generalize_with_confidence(
    mono: &TypeWithConfidence,
    free_vars: &[TypeVar],
    decay: &ConfidenceDecay,
) -> GeneralizationConfidence
```

### Instantiation with Confidence (~25 lines)

```sio
/// Instantiate type scheme with confidence
pub fn instantiate_with_confidence(
    poly: &TypeWithConfidence,
    fresh_vars: HashMap<TypeVar, TypeVar>,
    decay: &ConfidenceDecay,
) -> InstantiationConfidence
```

## Helper Functions (~30 lines)

```sio
/// Combine two TypeConfidenceMetadata (take minimum confidence)
fn combine_metadata(m1: &TypeConfidenceMetadata, m2: &TypeConfidenceMetadata) -> TypeConfidenceMetadata

/// Propagate confidence through type constructor (e.g., Vec<T>)
fn propagate_through_constructor(
    constructor: string,
    args: &[TypeWithConfidence],
) -> TypeWithConfidence

/// Check if all types meet minimum confidence threshold
fn all_above_threshold(types: &[TypeWithConfidence], min: ConfidenceLevel) -> bool
```

## Testing (~20 lines)

```sio
fn test_unify_with_confidence()
fn test_substitute_with_confidence()
fn test_generalization_confidence()
fn test_decay_propagation()
```

## Integration Notes

This module should integrate with:
- `crates/souc/src/check/rankn.rs` - Rank-N type checker (unify, substitute)
- `crates/souc/src/check/mod.rs` - Main type checker (infer, check)
- Existing confidence tracking happens ALONGSIDE normal type checking

## Notes for Code Generator

- Use `&!` for mutable references, NOT `&mut`
- Use `var` for mutable variables, NOT `let mut`
- No Rust macros or attributes
- Function signatures must match Sounio style (no generic lifetime params)
- Preserve immutability where possible (functional style)
- All `Result` types use string errors (no custom error types yet)
