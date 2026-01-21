# WP-1.3: Type Unification and Inference (Local LLM)

## Task
Implement type unification and constraint-based type inference for the self-hosted Sounio compiler.

## Output Files
- `stdlib/compiler/types/unify.sio` — Unification algorithm
- `stdlib/compiler/types/constraint.sio` — Constraint representation and solving
- `stdlib/compiler/types/subst.sio` — Substitution application
- `stdlib/compiler/types/infer.sio` — Type inference engine

## Target: ~1,200 LOC total

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

## Algorithm Background

### Hindley-Milner Type Inference

1. **Constraint Generation**: Walk the AST, generate equality constraints `T1 = T2`
2. **Unification**: Solve constraints by finding a substitution σ such that σ(T1) = σ(T2)
3. **Substitution Application**: Apply σ to all types in the program

### Unification Algorithm

```
unify(T1, T2):
  T1, T2 = apply_subst(T1), apply_subst(T2)  // Follow substitution chain

  if T1 == T2: return success

  if T1 is TypeVar(a):
    if a occurs in T2: return error (infinite type)
    add substitution a -> T2
    return success

  if T2 is TypeVar(b):
    if b occurs in T1: return error
    add substitution b -> T1
    return success

  if T1 = Array(E1) and T2 = Array(E2):
    return unify(E1, E2)

  if T1 = Ref(M1, I1) and T2 = Ref(M2, I2):
    if M1 != M2: return error
    return unify(I1, I2)

  if T1 = Function(P1, R1) and T2 = Function(P2, R2):
    if len(P1) != len(P2): return error
    for (p1, p2) in zip(P1, P2):
      unify(p1, p2)?
    return unify(R1, R2)

  // ... similar for Tuple, Named, etc.

  return error (type mismatch)
```

---

## unify.sio Structure

```sio
extern "C" {
    fn print(s: &str) -> ();
}

// Assume type.sio provides Type, TypePool, type_var, types_equal, etc.

// ============================================================================
// UNIFICATION RESULT
// ============================================================================

fn UNIFY_OK() -> i32 { 0 }
fn UNIFY_MISMATCH() -> i32 { 1 }
fn UNIFY_OCCURS() -> i32 { 2 }       // Infinite type (occurs check failed)
fn UNIFY_ARITY() -> i32 { 3 }        // Different number of args

struct UnifyResult {
    status: i32,
    // On error: what types couldn't unify
    left_idx: i64,
    right_idx: i64,
}

fn unify_ok() -> UnifyResult { ... }
fn unify_err(status: i32, left: i64, right: i64) -> UnifyResult { ... }

// ============================================================================
// SUBSTITUTION
// ============================================================================

// Maps type variable IDs to type indices
struct Substitution {
    var_ids: [i64; 128],
    type_idxs: [i64; 128],
    count: i64,
}

fn subst_new() -> Substitution { ... }

// Add mapping: var_id -> type_idx
fn subst_add(s: Substitution, var_id: i64, type_idx: i64) -> Substitution { ... }

// Lookup: returns type_idx or -1 if not found
fn subst_lookup(s: Substitution, var_id: i64) -> i64 { ... }

// Compose substitutions: s2 after s1
fn subst_compose(s1: Substitution, s2: Substitution) -> Substitution { ... }

// ============================================================================
// OCCURS CHECK
// ============================================================================

// Check if type variable occurs in type (would create infinite type)
fn occurs_in(var_id: i64, type_idx: i64, pool: TypePool) -> bool { ... }

// ============================================================================
// UNIFICATION
// ============================================================================

struct UnifyState {
    pool: TypePool,
    subst: Substitution,
    errors: [UnifyResult; 32],
    n_errors: i64,
}

fn unify_state_new(pool: TypePool) -> UnifyState { ... }

// Main unification function
// Returns updated state with substitution or error
fn unify(state: UnifyState, t1_idx: i64, t2_idx: i64) -> UnifyState { ... }

// Apply current substitution to a type, returning new type index
fn apply_subst_type(state: UnifyState, type_idx: i64) -> i64 { ... }

// Unify lists of types (for function params, tuple elements)
fn unify_lists(state: UnifyState, idxs1: [i64; 16], n1: i64, idxs2: [i64; 16], n2: i64) -> UnifyState { ... }

// ============================================================================
// HELPER: FOLLOW SUBSTITUTION CHAIN
// ============================================================================

// If type is a variable with a substitution, follow the chain
fn resolve_type(state: UnifyState, type_idx: i64) -> i64 { ... }

// ============================================================================
// TESTS
// ============================================================================

fn test_unify_identical() -> bool {
    // unify(i32, i32) should succeed
    ...
}

fn test_unify_var_concrete() -> bool {
    // unify(?T, i32) should succeed with ?T -> i32
    ...
}

fn test_unify_mismatch() -> bool {
    // unify(i32, bool) should fail
    ...
}

fn test_occurs_check() -> bool {
    // unify(?T, Array(?T)) should fail (infinite type)
    ...
}

fn test_unify_arrays() -> bool {
    // unify(Array(?T), Array(i32)) should succeed
    ...
}

fn test_unify_functions() -> bool {
    // unify(fn(?T) -> ?U, fn(i32) -> bool) should succeed
    ...
}

fn test_substitution_chain() -> bool {
    // ?T -> ?U, ?U -> i32 should resolve ?T to i32
    ...
}

fn main() -> i32 { ... }
```

---

## constraint.sio Structure

```sio
// ============================================================================
// CONSTRAINT KINDS
// ============================================================================

fn CONSTR_EQ() -> i32 { 0 }       // T1 = T2
fn CONSTR_SUBTYPE() -> i32 { 1 }  // T1 <: T2 (for coercion)
fn CONSTR_HAS_FIELD() -> i32 { 2 } // T has field .name : F
fn CONSTR_CALLABLE() -> i32 { 3 } // T is callable with args

// ============================================================================
// CONSTRAINT
// ============================================================================

struct Constraint {
    kind: i32,
    left_idx: i64,
    right_idx: i64,

    // Source location for error reporting
    span_start: i64,
    span_end: i64,

    // For HAS_FIELD: field name
    field_name: [i8; 32],
    field_name_len: i64,
}

fn constraint_eq(left: i64, right: i64, span_start: i64, span_end: i64) -> Constraint { ... }
fn constraint_subtype(sub: i64, super_: i64, span_start: i64, span_end: i64) -> Constraint { ... }
fn constraint_has_field(type_idx: i64, field: &str, field_type: i64, span_start: i64, span_end: i64) -> Constraint { ... }

// ============================================================================
// CONSTRAINT SET
// ============================================================================

struct ConstraintSet {
    constraints: [Constraint; 256],
    count: i64,
}

fn constraint_set_new() -> ConstraintSet { ... }
fn constraint_set_add(cs: ConstraintSet, c: Constraint) -> ConstraintSet { ... }

// ============================================================================
// CONSTRAINT SOLVING
// ============================================================================

struct SolveResult {
    success: bool,
    subst: Substitution,
    // First error if failed
    error_idx: i64,
}

// Solve all constraints in order
fn solve_constraints(cs: ConstraintSet, pool: TypePool) -> SolveResult { ... }

// ============================================================================
// TESTS
// ============================================================================

fn test_solve_simple() -> bool { ... }
fn test_solve_chain() -> bool { ... }
fn test_solve_failure() -> bool { ... }

fn main() -> i32 { ... }
```

---

## subst.sio Structure

```sio
// ============================================================================
// SUBSTITUTION APPLICATION
// ============================================================================

// Apply substitution to a single type, returning new type
// If type is a var with mapping, return the mapped type
// If type is compound (Array, Ref, etc.), recursively apply
fn apply_subst(subst: Substitution, t: Type, pool: &!TypePool) -> i64 { ... }

// Apply substitution to all types in a pool
fn apply_subst_pool(subst: Substitution, pool: TypePool) -> TypePool { ... }

// ============================================================================
// FREE VARIABLES
// ============================================================================

// Collect all free type variables in a type
fn free_vars(type_idx: i64, pool: TypePool, out: &![i64; 32], out_count: &!i64) -> () { ... }

// Check if type is ground (no type variables)
fn is_ground(type_idx: i64, pool: TypePool) -> bool { ... }

// ============================================================================
// ZONKING (final substitution application)
// ============================================================================

// Replace all remaining type variables with Error type
// Called after inference to catch unsolved variables
fn zonk(type_idx: i64, subst: Substitution, pool: &!TypePool) -> i64 { ... }

// ============================================================================
// TESTS
// ============================================================================

fn test_apply_var() -> bool { ... }
fn test_apply_compound() -> bool { ... }
fn test_free_vars() -> bool { ... }
fn test_zonk() -> bool { ... }

fn main() -> i32 { ... }
```

---

## infer.sio Structure

```sio
// ============================================================================
// INFERENCE ENGINE
// ============================================================================

struct InferState {
    pool: TypePool,
    constraints: ConstraintSet,
    next_var: i64,
    subst: Substitution,
}

fn infer_state_new() -> InferState { ... }

// Generate a fresh type variable
fn fresh_var(state: &!InferState) -> i64 { ... }

// Add equality constraint
fn constrain(state: &!InferState, t1: i64, t2: i64, span_start: i64, span_end: i64) -> () { ... }

// ============================================================================
// INSTANTIATION AND GENERALIZATION
// ============================================================================

// Instantiate a type scheme with fresh variables
fn instantiate(state: &!InferState, scheme_vars: [i64; 8], n_vars: i64, body_idx: i64) -> i64 { ... }

// Generalize a type over free variables not in environment
fn generalize(type_idx: i64, env_free: [i64; 32], env_n: i64) -> TypeScheme { ... }

// ============================================================================
// MAIN INFERENCE ENTRY POINT
// ============================================================================

// Run inference: generate constraints, solve, apply substitution
fn infer_and_solve(state: InferState) -> InferState { ... }

// ============================================================================
// TESTS
// ============================================================================

fn test_fresh_vars() -> bool { ... }
fn test_constraint_generation() -> bool { ... }
fn test_full_inference() -> bool { ... }

fn main() -> i32 { ... }
```

---

## Implementation Notes

1. **Type Pool Indices**: All types are stored in a TypePool. Functions take and return indices (i64), not Type structs directly.

2. **Substitution Representation**: Maps var_id -> type_idx. When looking up, recursively resolve chains.

3. **Occurs Check**: Essential to prevent infinite types like `?T = List<?T>`.

4. **Error Recovery**: Continue after first error to report multiple type errors.

5. **Invariant**: After successful unification, the substitution can be applied to get concrete types.
