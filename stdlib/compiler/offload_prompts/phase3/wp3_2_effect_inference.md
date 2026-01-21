# WP-3.2: Effect Inference & Checking

## Sounio Syntax Rules (CRITICAL)

- Use `var` for mutable variables
- Array indexing requires `with Panic`
- While loops may need `with Div`

## Reference Implementation

See: `compiler/src/effects/inference.rs`
See: `compiler/src/effects/check.rs`

## Target Output

**File**: `stdlib/compiler/effects/infer.sio`
**Estimated LOC**: ~1,200

## Specification

Implement bottom-up effect inference and declared vs. inferred checking.

### Effect Inference Algorithm

Traverse expression/statement tree, accumulating effects:

```sio
struct EffectInferenceResult {
    inferred_effects: i64,  // Bitmask of effects
    context: TypeContext,   // Updated context
    errors: [CompileError; 32],
    n_errors: i64,
}

// Infer effects from expression
fn infer_expr_effects(ctx: TypeContext, expr: HirExpr) -> EffectInferenceResult with Panic {
    // Match on expr kind and accumulate effects
    // Base cases: literals (no effects), variables (check bindings)
    // Recursive cases: function call (union with callee effects), array index (add Panic), etc.
}

// Infer effects from statement
fn infer_stmt_effects(ctx: TypeContext, stmt: HirStmt) -> EffectInferenceResult with Panic {
    // Similar to expr, but for statements
}

// Infer effects from function body
fn infer_fn_effects(ctx: TypeContext, fn_def: FunctionDef) -> EffectInferenceResult with Panic {
    // 1. Infer effects from body statements
    // 2. Add effects from returned expression
    // 3. Compare with declared effects
}
```

### Effect Sources

1. **Function Calls**
   - Look up callee function in context
   - Get declared effects from function signature
   - Union with current effect set

2. **Array Indexing** (`arr[i]`)
   - Adds `Panic` effect (bounds check may fail)

3. **Mutable Dereference** (`*ptr = value`)
   - Adds `Mut` effect

4. **Heap Allocation** (`new T`)
   - Adds `Alloc` effect

5. **I/O Operations** (file read/write, network)
   - Adds `IO` effect

6. **GPU Operations** (kernel launch)
   - Adds `GPU` effect

7. **Randomness** (random number generation)
   - Adds `Prob` effect

8. **Division** (`a / b`)
   - Adds `Div` effect (division by zero may fail)

### Effect Checking

```sio
struct EffectCheckResult {
    declared_effects: i64,   // From function signature
    inferred_effects: i64,   // From body
    missing_effects: i64,    // Inferred but not declared
    extra_effects: i64,      // Declared but not inferred
    is_consistent: bool,     // declared ⊇ inferred
}

// Check if inferred effects match declared
fn check_effects(declared: i64, inferred: i64) -> EffectCheckResult {
    var result = EffectCheckResult {
        declared_effects: declared,
        inferred_effects: inferred,
        missing_effects: 0,
        extra_effects: 0,
        is_consistent: true,
    };

    // Missing = inferred but not declared
    result.missing_effects = inferred & ~declared;

    // Extra = declared but not inferred (OK, just overly conservative)
    result.extra_effects = declared & ~inferred;

    // Consistent if inferred ⊆ declared
    result.is_consistent = (result.missing_effects == 0);
    result
}

// Report error if effects don't match
fn check_fn_effects(fn_sig: FunctionSignature, inferred: i64) -> bool {
    let check = check_effects(fn_sig.effect_set, inferred);
    if !check.is_consistent {
        // Error: function uses effects not declared
        // Example: function claims `with IO` but uses `Mut`
        return false;
    }
    true
}
```

### Propagation Rules

Effects propagate through:
- Sequential composition: `stmt1; stmt2` → union of both
- If/match: all branches contribute effects
- Function composition: `f(g(x))` → effects of g + effects of f
- Higher-order functions: apply effect variables

### Data Flow

```sio
// Track effect constraints during checking
struct EffectConstraint {
    source_expr: HirExpr,    // Where effect comes from
    effect: i32,             // The effect (IO, Mut, etc.)
    location: SourceLocation,
}

// Store all constraints during traversal
fn collect_effect_constraints(expr: HirExpr) -> [EffectConstraint; 128] {
    // ...
}

// Solve constraints to get final effect set
fn solve_effect_constraints(constraints: [EffectConstraint; 128], n: i64) -> i64 with Panic {
    var effect_mask: i64 = 0;
    var i: i64 = 0;
    while i < n {
        let eff = constraints[i as usize].effect;
        let bit = 1 << (eff as i64);
        effect_mask = effect_mask | bit;
        i = i + 1;
    }
    effect_mask
}
```

### Testing

Example function:
```
fn process_file(path: string) -> i64 with IO, Panic {
    // Opens file (IO), may throw (Panic)
    let contents = read_file(path);  // IO from read_file
    let len = contents.len();         // No effects
    if len > 100 {                    // No effects
        error("file too large")       // Panic from error
    }
    len as i64
}

// Inferred effects: IO (from read_file) + Panic (from error)
// Declared effects: IO, Panic
// Result: Consistent ✓
```

### Key Insight

Effect inference is bottom-up: analyze leaves first, combine upward. This allows local reasoning about which operations use effects, essential for writing effect-aware code.
