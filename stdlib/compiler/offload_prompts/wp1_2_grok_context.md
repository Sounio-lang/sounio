# WP-1.2: Type Context and Environment (Grok4 Fast)

## Task
Implement the type checking context and environment for the self-hosted Sounio compiler.

## Output Files
- `stdlib/compiler/check/context.sio` — Type checking context/state
- `stdlib/compiler/check/scope.sio` — Lexical scoping with scope stack
- `stdlib/compiler/check/binding.sio` — Variable bindings with types

## Target: ~600 LOC total

---

## CRITICAL: Sounio Syntax Rules

Sounio is NOT Rust. Follow these rules exactly:

```sio
// Mutable variable: var NOT let mut
var counter = 0

// Mutable reference: &! NOT &mut
fn modify(x: &!i32) { ... }

// Effects in signature
fn read_file(path: string) -> string with IO { ... }

// NO Rust macros: assert!(), println!(), etc.
// NO attributes: #[derive], #[test]
// NO tuple destructuring: let (a, b) = ...
// Define helpers BEFORE use (no forward references)
```

---

## Reference: Rust TypeChecker Struct

From `compiler/src/check/mod.rs`:

```rust
struct TypeChecker {
    env: TypeEnv,                           // Variable bindings and scopes
    type_defs: HashMap<String, TypeDef>,    // Struct/enum/alias definitions
    effects: EffectInference,               // Effect tracking
    units: UnitChecker,                     // Unit checking
    next_type_var: u32,                     // Fresh type var counter
    next_effect_var: u32,                   // Fresh effect var counter
    constraints: Vec<TypeConstraint>,       // Unification constraints
    errors: Vec<TypeError>,                 // Accumulated errors
    handler_effects: HashMap<String, String>,  // Handler -> effect mapping
    masked_effects: EffectSet,              // Effects handled in current scope
}

struct TypeEnv {
    scopes: Vec<Scope>,
    module_bindings: HashMap<(Vec<String>, String), TypeBinding>,
}
```

---

## context.sio Structure

```sio
extern "C" {
    fn print(s: &str) -> ();
}

// Import type system (will be in types/)
// These are placeholders - actual imports happen via file inclusion

// ============================================================================
// ERROR KINDS
// ============================================================================

fn ERR_NONE() -> i32 { 0 }
fn ERR_TYPE_MISMATCH() -> i32 { 1 }
fn ERR_UNDEFINED_VAR() -> i32 { 2 }
fn ERR_UNDEFINED_TYPE() -> i32 { 3 }
fn ERR_UNDEFINED_FN() -> i32 { 4 }
fn ERR_ARITY_MISMATCH() -> i32 { 5 }
fn ERR_NOT_CALLABLE() -> i32 { 6 }
fn ERR_NOT_INDEXABLE() -> i32 { 7 }
fn ERR_INVALID_BINOP() -> i32 { 8 }
fn ERR_INVALID_UNOP() -> i32 { 9 }
fn ERR_EFFECT_NOT_DECLARED() -> i32 { 10 }
fn ERR_LINEAR_USE() -> i32 { 11 }
fn ERR_UNIT_MISMATCH() -> i32 { 12 }

// ============================================================================
// TYPE ERROR
// ============================================================================

struct TypeError {
    kind: i32,
    span_start: i64,
    span_end: i64,
    expected_type_idx: i64,
    actual_type_idx: i64,
    message: [i8; 128],
    message_len: i64,
}

fn type_error_new() -> TypeError { ... }
fn type_error_mismatch(expected: i64, actual: i64, span_start: i64, span_end: i64) -> TypeError { ... }
fn type_error_undefined_var(name: &str, span_start: i64, span_end: i64) -> TypeError { ... }

// ============================================================================
// TYPE CONSTRAINT (for inference)
// ============================================================================

fn CONSTRAINT_EQ() -> i32 { 0 }      // T1 = T2
fn CONSTRAINT_SUBTYPE() -> i32 { 1 } // T1 <: T2

struct TypeConstraint {
    kind: i32,
    left_idx: i64,
    right_idx: i64,
    span_start: i64,
    span_end: i64,
}

fn constraint_eq(left: i64, right: i64, span_start: i64, span_end: i64) -> TypeConstraint { ... }

// ============================================================================
// TYPE DEFINITION
// ============================================================================

fn TYPEDEF_STRUCT() -> i32 { 0 }
fn TYPEDEF_ENUM() -> i32 { 1 }
fn TYPEDEF_ALIAS() -> i32 { 2 }

struct FieldDef {
    name: [i8; 32],
    name_len: i64,
    type_idx: i64,
    is_mutable: bool,
}

struct TypeDef {
    kind: i32,
    name: [i8; 64],
    name_len: i64,
    // For structs: fields
    fields: [FieldDef; 16],
    n_fields: i64,
    // For enums: variants (stored as field names)
    // For alias: target type
    target_type_idx: i64,
    // Type parameters
    type_params: [i64; 8],
    n_type_params: i64,
}

fn typedef_struct(name: &str) -> TypeDef { ... }
fn typedef_add_field(td: TypeDef, name: &str, type_idx: i64) -> TypeDef { ... }
fn typedef_enum(name: &str) -> TypeDef { ... }
fn typedef_alias(name: &str, target: i64) -> TypeDef { ... }

// ============================================================================
// FUNCTION SIGNATURE
// ============================================================================

struct FnSig {
    name: [i8; 64],
    name_len: i64,
    param_names: [[i8; 32]; 16],
    param_name_lens: [i64; 16],
    param_types: [i64; 16],
    n_params: i64,
    return_type_idx: i64,
    effect_set: i64,  // Bitmask of effects
    type_params: [i64; 8],
    n_type_params: i64,
}

fn fn_sig_new(name: &str) -> FnSig { ... }
fn fn_sig_add_param(sig: FnSig, name: &str, type_idx: i64) -> FnSig { ... }
fn fn_sig_set_return(sig: FnSig, type_idx: i64) -> FnSig { ... }
fn fn_sig_add_effect(sig: FnSig, effect: i32) -> FnSig { ... }

// ============================================================================
// TYPE CHECKING CONTEXT
// ============================================================================

struct TypeContext {
    // Type pool
    type_pool: TypePool,

    // Scopes (see scope.sio)
    scopes: [Scope; 32],
    n_scopes: i64,

    // Type definitions
    type_defs: [TypeDef; 64],
    n_type_defs: i64,

    // Function signatures
    fn_sigs: [FnSig; 64],
    n_fn_sigs: i64,

    // Constraints for inference
    constraints: [TypeConstraint; 256],
    n_constraints: i64,

    // Errors
    errors: [TypeError; 64],
    n_errors: i64,

    // Counters
    next_type_var: i64,
    next_effect_var: i64,

    // Current function being checked
    current_fn_idx: i64,
    current_return_type: i64,

    // Effect tracking
    declared_effects: i64,  // Bitmask
    inferred_effects: i64,  // Bitmask
    masked_effects: i64,    // Handled effects
}

fn context_new() -> TypeContext { ... }
fn context_push_scope(ctx: TypeContext) -> TypeContext { ... }
fn context_pop_scope(ctx: TypeContext) -> TypeContext { ... }
fn context_add_error(ctx: TypeContext, err: TypeError) -> TypeContext { ... }
fn context_add_constraint(ctx: TypeContext, c: TypeConstraint) -> TypeContext { ... }
fn context_fresh_type_var(ctx: &!TypeContext) -> i64 { ... }
fn context_add_type_def(ctx: TypeContext, td: TypeDef) -> TypeContext { ... }
fn context_add_fn_sig(ctx: TypeContext, sig: FnSig) -> TypeContext { ... }
fn context_lookup_type_def(ctx: TypeContext, name: &str) -> i64 { ... }  // Returns index or -1
fn context_lookup_fn_sig(ctx: TypeContext, name: &str) -> i64 { ... }

// ============================================================================
// TESTS
// ============================================================================

fn test_context_creation() -> bool { ... }
fn test_scope_push_pop() -> bool { ... }
fn test_type_def_registration() -> bool { ... }
fn test_fn_sig_registration() -> bool { ... }
fn test_error_accumulation() -> bool { ... }

fn main() -> i32 { ... }
```

---

## scope.sio Structure

```sio
// ============================================================================
// SCOPE
// ============================================================================

struct Scope {
    // Variable bindings in this scope
    bindings: [Binding; 32],
    n_bindings: i64,

    // Scope kind
    kind: i32,  // 0=block, 1=function, 2=loop, 3=match

    // For loops: break/continue type
    loop_break_type: i64,
}

fn SCOPE_BLOCK() -> i32 { 0 }
fn SCOPE_FUNCTION() -> i32 { 1 }
fn SCOPE_LOOP() -> i32 { 2 }
fn SCOPE_MATCH() -> i32 { 3 }

fn scope_new(kind: i32) -> Scope { ... }
fn scope_add_binding(s: Scope, b: Binding) -> Scope { ... }
fn scope_lookup(s: Scope, name: &str) -> i64 { ... }  // Returns binding index or -1

// ============================================================================
// SCOPE STACK OPERATIONS
// ============================================================================

// Look up variable in scope stack (innermost first)
fn scopes_lookup(scopes: [Scope; 32], n: i64, name: &str) -> Binding { ... }

// Find enclosing loop scope
fn scopes_find_loop(scopes: [Scope; 32], n: i64) -> i64 { ... }  // Returns scope index or -1

// Find enclosing function scope
fn scopes_find_function(scopes: [Scope; 32], n: i64) -> i64 { ... }

// ============================================================================
// TESTS
// ============================================================================

fn test_scope_binding() -> bool { ... }
fn test_scope_stack_lookup() -> bool { ... }
fn test_nested_scopes() -> bool { ... }

fn main() -> i32 { ... }
```

---

## binding.sio Structure

```sio
// ============================================================================
// BINDING KIND
// ============================================================================

fn BIND_VAR() -> i32 { 0 }       // let x = ...
fn BIND_VAR_MUT() -> i32 { 1 }   // var x = ...
fn BIND_PARAM() -> i32 { 2 }     // fn foo(x: T)
fn BIND_PARAM_MUT() -> i32 { 3 } // fn foo(x: &!T)

// ============================================================================
// BINDING
// ============================================================================

struct Binding {
    kind: i32,
    name: [i8; 32],
    name_len: i64,
    type_idx: i64,

    // For linear types: usage tracking
    is_linear: bool,
    is_consumed: bool,

    // Source location
    span_start: i64,
    span_end: i64,
}

fn binding_new(kind: i32, name: &str, type_idx: i64) -> Binding { ... }
fn binding_var(name: &str, type_idx: i64) -> Binding { ... }
fn binding_var_mut(name: &str, type_idx: i64) -> Binding { ... }
fn binding_param(name: &str, type_idx: i64) -> Binding { ... }

// Check if binding is mutable
fn binding_is_mutable(b: Binding) -> bool { ... }

// Mark as consumed (for linear types)
fn binding_consume(b: Binding) -> Binding { ... }

// Name comparison helper
fn binding_name_eq(b: Binding, name: &str) -> bool { ... }

// ============================================================================
// TESTS
// ============================================================================

fn test_binding_creation() -> bool { ... }
fn test_binding_mutability() -> bool { ... }
fn test_binding_name_lookup() -> bool { ... }

fn main() -> i32 { ... }
```
