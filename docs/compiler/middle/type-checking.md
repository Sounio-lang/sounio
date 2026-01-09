# Type Checking

The type checker validates Sounio programs and produces the typed High-level IR (HIR). It is located in `compiler/src/check/`.

## Overview

The type checker performs semantic analysis including:

- **Type inference**: Bidirectional type inference with constraint-based unification
- **Name resolution**: Resolving identifiers to their definitions
- **Effect checking**: Verifying effect annotations and inferring effect sets
- **Ownership analysis**: Checking linear/affine type usage
- **Unit checking**: Dimensional analysis for physical quantities
- **Ontology validation**: Verifying ontology term references
- **Epistemic constraints**: Checking Knowledge<T> type constraints
- **Semantic compatibility**: Ontology-based type compatibility

## Entry Points

```rust
/// Type check an AST and produce HIR
pub fn check(ast: &Ast) -> Result<Hir>

/// Type check with detailed error collection
pub fn check_with_errors(ast: &Ast) -> TypeCheckResult

pub struct TypeCheckResult {
    pub hir: Option<Hir>,
    pub errors: Vec<TypeError>,
    pub warnings: Vec<String>,
}
```

## Type Checker State

The `TypeChecker` struct maintains the checking context:

```rust
pub struct TypeChecker {
    /// Type environment (variable -> type)
    env: TypeEnv,

    /// Type definitions (structs, enums, aliases)
    type_defs: HashMap<String, TypeDef>,

    /// Effect inference context
    effects: EffectInference,

    /// Unit checker for dimensional analysis
    units: UnitChecker,

    /// Fresh type variable counter
    next_type_var: u32,

    /// Fresh effect variable counter (for row polymorphism)
    next_effect_var: u32,

    /// Type constraints for unification
    constraints: Vec<TypeConstraint>,

    /// Effect variable bindings: effect param name -> EffectVar id
    effect_params: HashMap<String, EffectVar>,

    /// Errors accumulated during checking
    errors: Vec<TypeError>,

    /// Ontology alignments: (type1, type2) -> distance
    alignments: HashMap<(String, String), f64>,

    /// Function-level compatibility thresholds from #[compat] annotations
    fn_thresholds: HashMap<String, f64>,

    /// Default compatibility threshold (0.15)
    default_threshold: f64,

    /// Handler definitions: handler_name -> effect_name
    handler_effects: HashMap<String, String>,

    /// Effects masked by handlers in current context
    masked_effects: EffectSet,

    /// Declared ontology prefixes
    ontology_prefixes: HashSet<String>,

    /// Warnings accumulated during checking
    warnings: Vec<String>,
}
```

## Type Environment

The type environment tracks variable bindings with scopes:

```rust
pub struct TypeEnv {
    /// Stack of scopes
    scopes: Vec<Scope>,

    /// Module-qualified bindings for qualified path resolution
    module_bindings: HashMap<(Vec<String>, String), TypeBinding>,
}

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

## Multi-Pass Type Checking

The type checker uses multiple passes:

### Pass 1: Collection

Collect type definitions, ontology prefixes, and alignments:

```rust
// Collect ontology prefixes from `ontology X from "..."` declarations
for item in &ast.items {
    self.collect_ontology_prefix(item);
}

// Collect type definitions and alignments
for item in &ast.items {
    self.collect_type_def(item);
    self.collect_alignment(item);
    self.collect_fn_threshold(item);
}
```

### Pass 2: Validation

Check for circular types and infinite-size structs:

```rust
self.check_circular_types();
self.check_infinite_size_types();
self.check_undefined_ontology_prefixes();
```

### Pass 3: Item Checking

Type check each item:

```rust
for item in &ast.items {
    match item {
        Item::Function(f) => self.check_function(f)?,
        Item::Struct(s) => self.check_struct(s)?,
        Item::Enum(e) => self.check_enum(e)?,
        Item::Impl(i) => self.check_impl(i)?,
        // ...
    }
}
```

### Pass 4: Constraint Solving

Solve type constraints via unification:

```rust
for constraint in &self.constraints {
    self.unify(&constraint.expected, &constraint.actual, constraint.span)?;
}
```

## Bidirectional Type Inference

The type checker uses bidirectional inference, where type information flows both up (synthesis) and down (checking).

### Synthesis Mode

Compute the type of an expression:

```rust
fn synthesize(&mut self, expr: &Expr) -> Result<(HirExpr, Type)> {
    match expr {
        Expr::Literal { value, .. } => {
            let ty = self.literal_type(value);
            Ok((HirExpr::Literal(self.convert_literal(value)), ty))
        }

        Expr::Path { path, .. } => {
            let ty = self.lookup_var(&path.segments[0])?;
            Ok((HirExpr::Var(path.segments[0].clone()), ty))
        }

        Expr::Binary { op, left, right, .. } => {
            let (lhs, lhs_ty) = self.synthesize(left)?;
            let (rhs, rhs_ty) = self.synthesize(right)?;
            let result_ty = self.binary_result_type(*op, &lhs_ty, &rhs_ty)?;
            Ok((HirExpr::Binary { op: *op, lhs, rhs }, result_ty))
        }

        // ...
    }
}
```

### Checking Mode

Check an expression against an expected type:

```rust
fn check(&mut self, expr: &Expr, expected: &Type) -> Result<HirExpr> {
    match (expr, expected) {
        // Closures can be checked against function types
        (Expr::Closure { params, body, .. }, Type::Function { params: param_tys, return_type, .. }) => {
            self.check_closure(params, body, param_tys, return_type)
        }

        // Arrays can be checked against array types
        (Expr::Array { elements, .. }, Type::Array { element, .. }) => {
            for elem in elements {
                self.check(elem, element)?;
            }
            // ...
        }

        // Fall back to synthesis and unification
        _ => {
            let (hir_expr, actual) = self.synthesize(expr)?;
            self.constrain(expected.clone(), actual, self.expr_span(expr));
            Ok(hir_expr)
        }
    }
}
```

## Type Unification

Types are unified using constraint-based unification:

```rust
fn unify(&mut self, expected: &Type, actual: &Type, span: Span) -> Result<()> {
    match (expected, actual) {
        // Identical types
        (a, b) if a == b => Ok(()),

        // Type variables
        (Type::Var(v), ty) | (ty, Type::Var(v)) => {
            self.bind_type_var(*v, ty.clone());
            Ok(())
        }

        // Named types with same name
        (Type::Named { name: n1, args: a1 }, Type::Named { name: n2, args: a2 })
            if n1 == n2 && a1.len() == a2.len() => {
            for (t1, t2) in a1.iter().zip(a2) {
                self.unify(t1, t2, span)?;
            }
            Ok(())
        }

        // Function types
        (Type::Function { params: p1, return_type: r1, effects: e1 },
         Type::Function { params: p2, return_type: r2, effects: e2 }) => {
            self.unify_params(p1, p2, span)?;
            self.unify(r1, r2, span)?;
            self.unify_effects(e1, e2)?;
            Ok(())
        }

        // Ontology types - check semantic distance
        (Type::Ontology { namespace: ns1, term: t1 },
         Type::Ontology { namespace: ns2, term: t2 }) => {
            self.check_ontology_compatibility(ns1, t1, ns2, t2, span)
        }

        // Type mismatch
        _ => {
            self.error(format!(
                "Type mismatch: expected {}, found {}",
                self.type_display_name(expected),
                self.type_display_name(actual)
            ), span);
            Err(...)
        }
    }
}
```

## Fresh Type Variables

Type variables are used for inference:

```rust
fn fresh_type_var(&mut self) -> Type {
    let var = TypeVar(self.next_type_var);
    self.next_type_var += 1;
    Type::Var(var)
}
```

When type information is missing, a fresh type variable is created and later unified with concrete types.

## Type Alias Expansion

Type aliases are expanded during checking:

```rust
fn expand_type_alias(&self, ty: &Type) -> Type {
    match ty {
        Type::Named { name, args } => {
            if let Some(TypeDef::Alias(alias_ty, _, _)) = self.type_defs.get(name) {
                // Recursively expand
                self.expand_type_alias(alias_ty)
            } else {
                // Expand args recursively
                Type::Named {
                    name: name.clone(),
                    args: args.iter().map(|a| self.expand_type_alias(a)).collect(),
                }
            }
        }
        // Other types: expand recursively
        _ => ...
    }
}
```

## Semantic Compatibility

Sounio supports ontology-based semantic type compatibility:

```rust
fn check_ontology_compatibility(
    &mut self,
    ns1: &str, term1: &str,
    ns2: &str, term2: &str,
    span: Span
) -> Result<()> {
    let key = self.alignment_key(ns1, term1, ns2, term2);

    if let Some(distance) = self.alignments.get(&key) {
        let threshold = self.current_threshold();

        if *distance <= threshold {
            Ok(())  // Compatible
        } else {
            self.error(format!(
                "Semantic distance {} exceeds threshold {}",
                distance, threshold
            ), span);
            Err(...)
        }
    } else {
        // No alignment declared - require exact match
        if ns1 == ns2 && term1 == term2 {
            Ok(())
        } else {
            self.error("No ontology alignment declared", span);
            Err(...)
        }
    }
}
```

The compatibility threshold can be set per-function using attributes:

```sio
#[compat(0.1)]  // Strict threshold
fn precise_calculation(x: chebi:drug) -> f64 { ... }

#[compat(loose)]  // Loose threshold (0.25)
fn approximate_match(x: chebi:compound) -> f64 { ... }
```

## Effect Masking

When an effect is handled, it is masked from the function's external effect set:

```rust
fn check_handle_expr(&mut self, expr: &Expr, handler: &Path) -> Result<(HirExpr, Type)> {
    // Look up which effect the handler handles
    let effect_name = self.lookup_handler_effect(&handler.segments[0])?;

    // Add to masked effects
    self.masked_effects.insert(&effect_name);

    // Check the inner expression
    let (inner_hir, inner_ty) = self.synthesize(expr)?;

    Ok((HirExpr::Handle { inner: inner_hir, handler }, inner_ty))
}

/// Compute residual effects after masking
pub fn compute_residual_effects(
    inferred: &EffectSet,
    masked: &EffectSet,
) -> EffectSet {
    inferred.subtract(&masked.effects.iter().cloned().collect::<Vec<_>>())
}
```

## Error Reporting

Errors are accumulated and reported with source locations:

```rust
fn error(&mut self, message: impl Into<String>, span: Span) {
    self.errors.push(TypeError {
        message: message.into(),
        span,
        code: "E0308".to_string(),
    });
}

fn error_with_code(&mut self, code: &str, message: impl Into<String>, span: Span) {
    self.errors.push(TypeError {
        message: message.into(),
        span,
        code: code.to_string(),
    });
}
```

## Type Definitions

Type definitions are stored for lookup:

```rust
enum TypeDef {
    Struct {
        fields: Vec<(String, Type)>,
        linear: bool,
        affine: bool,
        source_module: Option<ModuleId>,
    },
    Enum {
        variants: Vec<(String, Vec<Type>)>,
        linear: bool,
        affine: bool,
        source_module: Option<ModuleId>,
    },
    Alias(Type, Span, Option<ModuleId>),
}
```

## Submodules

- `compiler/src/check/mod.rs` - Main type checker
- `compiler/src/check/compatibility.rs` - Semantic compatibility checking
- `compiler/src/check/diagnostics.rs` - Diagnostic generation
- `compiler/src/check/epistemic.rs` - Epistemic type checking

## Related Modules

- `compiler/src/types/` - Type representation and operations
- `compiler/src/effects/` - Effect system types
- `compiler/src/units/` - Unit checking for quantities
- `compiler/src/refinement/` - Refinement type constraints
