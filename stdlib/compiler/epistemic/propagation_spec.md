# propagation.sio - Confidence Propagation Through Type Checking

**Target**: 400 lines of Sounio code
**LLM**: Deepseek Math (best for algorithm-heavy propagation logic)
**Purpose**: Propagate epistemic confidence through the full type checking pipeline

## Module Overview

This module implements confidence propagation for all type checking operations:
- Expression type inference
- Function application
- Pattern matching
- Constraint solving
- Global type environment updates

It's the "orchestration layer" that ties together confidence_metadata and type_confidence
into the actual type checking algorithm.

## Dependencies

```sio
use core::{Option, Result, Vec, HashMap}
use compiler::types::{Type, TypeId, TypeVar, TypeEnv, Substitution}
use compiler::ast::{Expr, Pattern, NodeId, Span}
use compiler::epistemic::confidence_metadata::{
    TypeConfidenceMetadata, ConfidenceLevel, InferenceSource,
    ConfidenceContext, ConstraintConfidence, InferenceStep
}
use compiler::epistemic::type_confidence::{
    TypeWithConfidence, UnificationConfidence, SubstitutionConfidence,
    ConfidenceDecay, unify_with_confidence, substitute_with_confidence
}
use epistemic::knowledge::BetaConfidence
```

## Data Types

### 1. PropagationContext (struct, ~50 lines)

Global context for propagation through type checking.

```sio
/// Context for confidence propagation
pub struct PropagationContext {
    /// Confidence tracking
    conf_ctx: ConfidenceContext,

    /// Type environment with confidences
    env: HashMap<NodeId, TypeWithConfidence>,

    /// Decay configuration
    decay: ConfidenceDecay,

    /// Constraint graph for propagation
    constraints: Vec<PropagationConstraint>,

    /// Dependency graph: which types depend on which
    dependencies: HashMap<TypeId, Vec<TypeId>>,

    /// Minimum confidence threshold for warnings
    warning_threshold: ConfidenceLevel,
}
```

Methods:
- `new(ConfidenceDecay) -> PropagationContext`
- `bind(NodeId, TypeWithConfidence)` - Add variable binding
- `lookup(NodeId) -> Option<&TypeWithConfidence>` - Query binding
- `add_constraint(PropagationConstraint)` - Add constraint
- `add_dependency(TypeId, TypeId)` - Record dependency edge
- `propagate_all()` - Run constraint propagation to fixpoint
- `check_thresholds() -> Vec<ConfidenceWarning>` - Find low-confidence types

### 2. PropagationConstraint (enum, ~40 lines)

Constraint types for propagation.

```sio
/// Constraint for confidence propagation
pub enum PropagationConstraint {
    /// Unification: τ₁ ~ τ₂
    Unify {
        lhs: TypeId,
        rhs: TypeId,
        source: InferenceSource,
        confidence: ConfidenceLevel,
    },

    /// Subtyping: τ₁ <: τ₂
    Subtype {
        sub: TypeId,
        sup: TypeId,
        source: InferenceSource,
    },

    /// Function application: f : τ₁ → τ₂, arg : τ₁ ⊢ result : τ₂
    Application {
        fn_type: TypeId,
        arg_type: TypeId,
        result_type: TypeId,
    },

    /// Let binding: Γ ⊢ e : τ, Γ, x : ∀ᾱ.τ ⊢ ...
    LetBinding {
        var: NodeId,
        value_type: TypeId,
        generalized: bool,
    },

    /// Pattern match: scrutinee : τ, pattern : τ′ ⊢ τ ~ τ′
    PatternMatch {
        scrutinee_type: TypeId,
        pattern_type: TypeId,
        bindings: Vec<NodeId>,
    },
}
```

Methods:
- `involved_types() -> Vec<TypeId>` - Get all types in constraint
- `propagate(&!PropagationContext) -> Result<unit, string>` - Execute propagation

### 3. ConfidenceWarning (struct, ~25 lines)

Warning for low-confidence type inference.

```sio
/// Warning for low-confidence inference
pub struct ConfidenceWarning {
    /// AST node with low confidence
    node: NodeId,

    /// Span for error reporting
    span: Span,

    /// Inferred type
    ty: Type,

    /// Actual confidence
    confidence: ConfidenceLevel,

    /// Reason for low confidence
    reason: string,
}
```

Methods:
- `new(NodeId, Span, Type, ConfidenceLevel, string) -> ConfidenceWarning`
- `to_diagnostic() -> string` - Format for display

### 4. PropagationResult (struct, ~30 lines)

Result of confidence propagation.

```sio
/// Result of propagation pass
pub struct PropagationResult {
    /// Did confidences change? (for fixpoint iteration)
    changed: bool,

    /// Updated type environment
    env: HashMap<NodeId, TypeWithConfidence>,

    /// New warnings generated
    warnings: Vec<ConfidenceWarning>,

    /// Iteration count (for debugging)
    iterations: i64,
}
```

## Core Propagation Functions

### 1. Expression Type Inference with Confidence (~80 lines)

```sio
/// Infer type of expression with confidence tracking
pub fn infer_expr_with_confidence(
    expr: &Expr,
    ctx: &!PropagationContext,
) -> Result<TypeWithConfidence, string>
```

Implementation sketch:
- **Literal**: Return certain confidence (e.g., `42` → `i32` with Certain)
- **Var**: Lookup in ctx.env, add dependency
- **App**: Infer function, infer arg, unify, propagate confidence
- **Lam**: Infer body with parameter in env, decay confidence for closure
- **Let**: Infer value, generalize if appropriate, bind, infer body
- **If**: Infer condition, branches, unify branch types, combine confidences
- **Match**: Infer scrutinee, match patterns, unify with branches

Each case:
1. Recursively infer sub-expressions
2. Combine confidences from sub-expressions
3. Apply appropriate decay
4. Add provenance step
5. Return TypeWithConfidence

### 2. Pattern Matching with Confidence (~50 lines)

```sio
/// Check pattern against type with confidence
pub fn check_pattern_with_confidence(
    pat: &Pattern,
    ty: &TypeWithConfidence,
    ctx: &!PropagationContext,
) -> Result<HashMap<NodeId, TypeWithConfidence>, string>
```

Implementation:
- **PatVar**: Bind variable with full confidence from ty
- **PatLit**: Check literal type matches ty (certain if matches)
- **PatCon**: Check constructor, recurse on fields, decay confidence
- **PatTuple**: Decompose tuple type, check elements
- **PatWildcard**: Accept any type, no bindings

Returns bindings for pattern variables with their confidences.

### 3. Function Application Confidence (~40 lines)

```sio
/// Propagate confidence through function application
pub fn apply_function_with_confidence(
    fn_ty: &TypeWithConfidence,
    arg_ty: &TypeWithConfidence,
    ctx: &!PropagationContext,
) -> Result<TypeWithConfidence, string>
```

Algorithm:
1. Extract function type (should be τ₁ → τ₂)
2. Unify argument type with τ₁
3. Combine confidences from function, argument, unification
4. Apply decay (function application reduces confidence)
5. Return result type with updated confidence

### 4. Let Binding Confidence (~45 lines)

```sio
/// Handle let binding with generalization
pub fn bind_let_with_confidence(
    var: NodeId,
    value: &TypeWithConfidence,
    should_generalize: bool,
    ctx: &!PropagationContext,
) -> TypeWithConfidence
```

Algorithm:
1. If should_generalize:
   - Compute free variables
   - Generalize to ∀ᾱ.τ
   - Decay confidence (generalization is less certain)
2. Else:
   - Keep monomorphic type
   - Preserve confidence
3. Bind in ctx.env

### 5. Constraint Propagation (~60 lines)

```sio
/// Propagate confidences through constraint graph until fixpoint
pub fn propagate_constraints(
    ctx: &!PropagationContext,
) -> PropagationResult
```

Algorithm:
1. Initialize work queue with all constraints
2. While queue non-empty:
   a. Pop constraint
   b. Propagate confidence according to constraint type
   c. If any type's confidence changed, add dependent constraints back to queue
3. Check for convergence
4. Generate warnings for low-confidence types
5. Return PropagationResult

### 6. Dependency Tracking (~35 lines)

```sio
/// Add dependency edge: type_id depends on dep_id
fn add_type_dependency(
    type_id: TypeId,
    dep_id: TypeId,
    ctx: &!PropagationContext,
)

/// Get all types that depend on type_id (transitive closure)
fn get_dependent_types(
    type_id: TypeId,
    ctx: &PropagationContext,
) -> Vec<TypeId>

/// Invalidate confidence for type and all dependents
fn invalidate_confidence(
    type_id: TypeId,
    new_confidence: ConfidenceLevel,
    ctx: &!PropagationContext,
)
```

## Helper Functions (~40 lines)

```sio
/// Combine confidences from multiple sources (minimum)
fn combine_expr_confidences(exprs: &[TypeWithConfidence]) -> ConfidenceLevel

/// Decay confidence through closure capture
fn decay_for_closure(inner: &TypeWithConfidence, decay: &ConfidenceDecay) -> TypeWithConfidence

/// Check if generalization should occur (based on confidence)
fn should_generalize(ty: &TypeWithConfidence, threshold: ConfidenceLevel) -> bool

/// Generate warning if confidence below threshold
fn check_confidence_threshold(
    node: NodeId,
    span: Span,
    ty_conf: &TypeWithConfidence,
    threshold: ConfidenceLevel,
) -> Option<ConfidenceWarning>
```

## Testing (~40 lines)

```sio
fn test_infer_literal_confidence()    // 42 should be Certain
fn test_infer_var_confidence()        // Lookup should preserve confidence
fn test_apply_confidence_decay()      // Application should decay
fn test_let_generalization()          // Generalization reduces confidence
fn test_pattern_match_confidence()    // Pattern matching propagates
fn test_constraint_propagation()      // Fixpoint iteration
fn test_dependency_tracking()         // Transitive dependencies
fn test_warning_generation()          // Low confidence warnings
```

## Integration with Type Checker

This module hooks into:
- `crates/souc/src/check/mod.rs::TypeChecker::infer()`
- `crates/souc/src/check/mod.rs::TypeChecker::check()`
- `crates/souc/src/check/rankn.rs::unify()`

Strategy:
1. Run normal type checking first
2. Run confidence propagation in parallel
3. Emit warnings for low-confidence inferences
4. Optionally require minimum confidence for critical code

## Notes for Code Generator

- Use `&!` for mutable references, NOT `&mut`
- Use `var` for mutable variables, NOT `let mut`
- No Rust macros, attributes, or destructuring
- This is the most algorithmically complex module - focus on clarity
- Constraint propagation should reach fixpoint (iterate until no changes)
- Dependency graph prevents infinite loops (track visited nodes)
- All confidence updates should be monotonic (never increase, only decay)
- Reference stdlib/epistemic/knowledge.sio for patterns
