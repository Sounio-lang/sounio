# WP-3.1: Effect Type System

## Sounio Syntax Rules (CRITICAL)

- Use `var` for mutable variables
- Array indexing requires `with Panic`
- NO type suffixes

## Reference Implementation

See: `compiler/src/types/effects.rs`
See: `compiler/src/effects/effect_kind.rs` (EffectKind enum)

## Target Output

**File**: `stdlib/compiler/effects/effect.sio`
**Estimated LOC**: ~800

## Specification

Implement the effect type system for tracking side effects.

### Effect Kinds

Sounio supports 8 algebraic effects:

1. **IO** - Input/output operations (file read/write, network)
2. **Mut** - Mutable reference dereference and modification
3. **Alloc** - Memory allocation (heap)
4. **Panic** - Diverging effects (array bounds check, divide by zero)
5. **Async** - Asynchronous operations (futures, coroutines)
6. **GPU** - GPU computation
7. **Prob** - Probabilistic effects (sampling, randomness)
8. **Div** - Division/divergence

### Data Structures

```sio
// Effect kind constants
fn EFFECT_IO() -> i32 { 0 }
fn EFFECT_MUT() -> i32 { 1 }
fn EFFECT_ALLOC() -> i32 { 2 }
fn EFFECT_PANIC() -> i32 { 3 }
fn EFFECT_ASYNC() -> i32 { 4 }
fn EFFECT_GPU() -> i32 { 5 }
fn EFFECT_PROB() -> i32 { 6 }
fn EFFECT_DIV() -> i32 { 7 }

// Effect set (bitmask of effects)
struct EffectSet {
    mask: i64,  // Bit i set means effect i present
}

// Effect variable (for row polymorphism)
struct EffectVar {
    id: i64,           // Unique ID
    lower_bound: i64,  // Minimum effects (bitmask)
    upper_bound: i64,  // Maximum effects (bitmask)
}
```

### Operations

```sio
// Create effect set with single effect
fn effect_singleton(kind: i32) -> EffectSet with Panic {
    EffectSet { mask: (1 << (kind as i64)) }
}

// Union of two effect sets
fn effect_union(e1: EffectSet, e2: EffectSet) -> EffectSet {
    EffectSet { mask: e1.mask | e2.mask }
}

// Intersection
fn effect_intersect(e1: EffectSet, e2: EffectSet) -> EffectSet {
    EffectSet { mask: e1.mask & e2.mask }
}

// Check if effect present
fn effect_contains(e: EffectSet, kind: i32) -> bool with Panic {
    let bit = 1 << (kind as i64);
    (e.mask & bit) != 0
}

// Empty effect set (pure)
fn effect_pure() -> EffectSet {
    EffectSet { mask: 0 }
}

// All effects
fn effect_total() -> EffectSet {
    EffectSet { mask: -1 }  // All bits set
}

// String representation
fn effect_to_string(e: EffectSet) -> &str {
    // Return comma-separated effect names
    // "IO, Mut, Panic" etc.
}
```

### Effect Variables (Row Polymorphism)

For functions with polymorphic effects:

```sio
// Example: `fn process(f: Fn() -> T with e) -> T with e`
// Here `e` is an effect variable representing unknown effects

fn effect_var_fresh(id: i64) -> EffectVar {
    EffectVar { id: id, lower_bound: 0, upper_bound: -1 }  // Unbounded
}

fn effect_var_constrain(ev: EffectVar, lower: EffectSet, upper: EffectSet) -> EffectVar {
    // Refine bounds
    EffectVar {
        id: ev.id,
        lower_bound: ev.lower_bound | lower.mask,
        upper_bound: ev.upper_bound & upper.mask
    }
}
```

### Effect Syntax

Function declarations:
```
fn read_file(path: string) -> string with IO { ... }
fn modify_ref(r: &!i32) -> i32 with Mut { ... }
fn risky_divide(a: i64, b: i64) -> i64 with Panic, Div { ... }
fn pure_function(x: i64) -> i64 { ... }  // No effects (implicitly pure)
```

### Integration

- Effect sets are part of function type signatures
- Type checker thread effects through expression checking
- Check declared effects against inferred effects
- Row polymorphism allows flexible effect propagation

### Testing

Effect inference examples:
```
fn caller() -> i64 with IO {
    let x = read_file("test.txt");  // Infers IO from read_file
    x.len() as i64
}
// Inferred effects: IO (from read_file)
```

### Key Insight

Effects are tracked as bitmasks for efficiency. Effect variables enable higher-order functions to be polymorphic over their effects, crucial for applicative programming.
