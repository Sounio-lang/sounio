# WP-1.1: Type Representation (Minimax)

## Task
Implement the complete type system representation for the self-hosted Sounio compiler.

## Output Files
- `stdlib/compiler/types/type.sio` — Core Type enum (all 46 variants)
- `stdlib/compiler/types/type_var.sio` — Type variables for inference
- `stdlib/compiler/types/type_scheme.sio` — Polymorphic type schemes
- `stdlib/compiler/types/builtin.sio` — Built-in type definitions

## Target: ~800 LOC total

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

## Existing Pattern to Follow

The existing code uses struct + kind constants pattern (see stdlib/compiler/check/types.sio):

```sio
// Type kind constants as functions
fn TYPE_UNKNOWN() -> i32 { 0 }
fn TYPE_I32() -> i32 { 6 }
fn TYPE_VAR() -> i32 { 26 }

// Type struct with kind discriminator
struct Type {
    kind: i32,
    inner_idx: i64,    // For references, arrays
    array_size: i64,
    // ... other fields
}

// Constructor functions
fn type_i32() -> Type {
    Type { kind: TYPE_I32(), inner_idx: -1, array_size: 0 }
}

// Type pool for allocation
struct TypePool {
    types: [Type; 256],
    count: i64,
}
```

---

## Types to Implement (from Rust compiler/src/types/core.rs)

### Primitives (kinds 0-20)
```
Unit, Bool, I8, I16, I32, I64, I128, Isize, U8, U16, U32, U64, U128, Usize, F32, F64, Char, Str, String
```

### Linear Algebra (kinds 30-37)
```
Vec2, Vec3, Vec4, Mat2, Mat3, Mat4, Quat, Dual
```

### Compound Types (kinds 40-50)
```
Ref { mutable: bool, inner: Type }
RawPointer { mutable: bool, inner: Type }
Array { element: Type, size: Option<usize> }
Slice { element: Type }
Tuple { elements: [Type] }
Function { params: [Type], return_type: Type, effects: EffectSet }
Named { name: string, args: [Type] }
```

### Epistemic/Scientific (kinds 60-70)
```
Quantity { numeric: Type, unit: string }
Ontology { namespace: string, term: string }
Knowledge { inner: Type, epsilon: f64, provenance: i32 }
Tensor { element: Type, dims: [i64] }
Future { output: Type }
```

### Polymorphism (kinds 80-85)
```
Var(TypeVar)           // Type variable
Forall { vars: [TypeVar], inner: Type }
```

### Special (kinds 90-95)
```
Never, Unknown, Error, SelfType
```

---

## type.sio Structure

```sio
// Type kind constants
fn TYPE_UNIT() -> i32 { 2 }
fn TYPE_BOOL() -> i32 { 3 }
// ... all 46 types

// Main Type struct
struct Type {
    kind: i32,

    // Reference/pointer inner type
    inner_idx: i64,
    is_mutable: bool,

    // Array/slice
    array_size: i64,  // -1 for slice

    // Named types
    name: [i8; 64],
    name_len: i64,

    // Generics/args
    type_args: [i64; 8],
    n_args: i64,

    // Function
    param_types: [i64; 16],
    n_params: i64,
    return_type_idx: i64,
    effect_set: i64,  // Bitmask

    // Quantity
    unit: [i8; 32],
    unit_len: i64,

    // Knowledge
    epsilon: f64,
    provenance: i32,

    // Type variable
    var_id: i64,
}

// Constructors
fn type_unit() -> Type { ... }
fn type_bool() -> Type { ... }
fn type_i32() -> Type { ... }
// ... all primitives

fn type_ref(inner_idx: i64) -> Type { ... }
fn type_ref_mut(inner_idx: i64) -> Type { ... }
fn type_array(elem_idx: i64, size: i64) -> Type { ... }
fn type_slice(elem_idx: i64) -> Type { ... }
fn type_tuple(elems: [i64; 8], n: i64) -> Type { ... }
fn type_function(params: [i64; 16], n: i64, ret: i64, effects: i64) -> Type { ... }
fn type_named(name: &str, args: [i64; 8], n: i64) -> Type { ... }
fn type_quantity(numeric_idx: i64, unit: &str) -> Type { ... }
fn type_knowledge(inner_idx: i64, epsilon: f64) -> Type { ... }
fn type_var(id: i64) -> Type { ... }

// Type pool (256 capacity, can chain multiple pools)
struct TypePool { ... }
fn type_pool_new() -> TypePool { ... }
fn type_pool_add(pool: TypePool, t: Type) -> TypePool { ... }
fn type_pool_get(pool: TypePool, idx: i64) -> Type { ... }

// Predicates
fn is_numeric(t: Type) -> bool { ... }
fn is_integer(t: Type) -> bool { ... }
fn is_float(t: Type) -> bool { ... }
fn is_signed(t: Type) -> bool { ... }
fn is_vector(t: Type) -> bool { ... }
fn is_matrix(t: Type) -> bool { ... }
fn is_linear_algebra(t: Type) -> bool { ... }
fn is_copy(t: Type) -> bool { ... }
fn is_sized(t: Type) -> bool { ... }

// Comparison
fn types_equal(a: Type, b: Type) -> bool { ... }
fn types_compatible(a: Type, b: Type) -> bool { ... }  // Allows coercion
```

---

## type_var.sio Structure

```sio
// Type variable for polymorphism
struct TypeVar {
    id: i64,
    constraint: i32,  // 0=none, 1=numeric, 2=eq, etc.
}

fn type_var_new(id: i64) -> TypeVar { ... }
fn type_var_fresh(counter: &!i64) -> TypeVar { ... }

// Effect variable for row polymorphism
struct EffectVar {
    id: i64,
}

fn effect_var_new(id: i64) -> EffectVar { ... }
```

---

## type_scheme.sio Structure

```sio
// Polymorphic type scheme: forall a b. T
struct TypeScheme {
    type_vars: [i64; 8],
    n_type_vars: i64,
    effect_vars: [i64; 4],
    n_effect_vars: i64,
    body_idx: i64,  // Type index
}

fn scheme_new(body_idx: i64) -> TypeScheme { ... }
fn scheme_add_type_var(s: TypeScheme, var: i64) -> TypeScheme { ... }
fn scheme_instantiate(s: TypeScheme, pool: &!TypePool, counter: &!i64) -> i64 { ... }
fn scheme_generalize(ty_idx: i64, env_free_vars: [i64; 16], n: i64) -> TypeScheme { ... }
```

---

## builtin.sio Structure

```sio
// Built-in type definitions and their indices in the default pool
struct BuiltinTypes {
    unit_idx: i64,
    bool_idx: i64,
    i8_idx: i64,
    i16_idx: i64,
    i32_idx: i64,
    i64_idx: i64,
    // ... all primitives
    string_idx: i64,
}

fn builtin_types_init(pool: &!TypePool) -> BuiltinTypes { ... }

// Get built-in type by name
fn builtin_lookup(name: &str, builtins: BuiltinTypes) -> i64 { ... }
```

---

## Tests to Include

Each file should have test functions at the bottom:

```sio
fn test_type_construction() -> bool { ... }
fn test_type_equality() -> bool { ... }
fn test_type_pool() -> bool { ... }
fn test_type_predicates() -> bool { ... }

fn main() -> i32 {
    // Run all tests, return 0 on success
}
```
