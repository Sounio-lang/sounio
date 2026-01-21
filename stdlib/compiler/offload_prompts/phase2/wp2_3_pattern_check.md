# WP-2.3: Pattern Matching & Exhaustiveness Checking

## Sounio Syntax Rules (CRITICAL)

- Use `var` for mutable variables
- NO compound assignments
- Array indexing requires `with Panic`

## Reference Implementation

See: `compiler/src/check/pattern.rs`
See: `compiler/src/hir/pattern.rs` (HirPatternKind enum)

## Target Output

**File**: `stdlib/compiler/check/pattern.sio`
**Estimated LOC**: ~800

## Specification

Implement pattern matching type checking and exhaustiveness analysis.

### Pattern Kinds

1. **Wildcard** (`_`)
   - Matches anything
   - Binds no variables
   - Type is the scrutinee type

2. **Literal Pattern** (`5`, `true`, `"hello"`)
   - Matches specific literal value
   - Type must match literal type
   - Example: pattern `42` has type `i64`

3. **Variable Pattern** (`x`)
   - Binds variable to scrutinee value
   - Type is scrutinee type
   - Adds binding to scope

4. **Struct Pattern** (`Point { x, y }`)
   - Matches struct with field patterns
   - Recursively type checks field patterns
   - Binds field variables

5. **Enum Pattern** (`Some(x)`, `None`)
   - Matches enum variant
   - Type checks variant payload
   - Adds bindings from payload

6. **Array Pattern** (`[x, y, z]`)
   - Matches array of fixed size
   - Type checks each element pattern
   - Infers array element type

7. **Or Pattern** (`pat1 | pat2`)
   - Multiple alternatives
   - All alternatives must have same type
   - All bindings from all patterns in scope

### Type Checking

```sio
struct PatternCheckResult {
    type_idx: i64,       // Type of pattern
    bindings: [Binding; 64],  // Variables bound by pattern
    n_bindings: i64,
    has_error: bool,
}

// Check if pattern matches scrutinee type
fn check_pattern(ctx: TypeContext, pat: Pattern, scrutinee_type: i64) -> PatternCheckResult with Panic {
    // ...
}
```

### Exhaustiveness Checking

Analyze match arms for completeness:

```sio
struct MatchCheckResult {
    all_covered: bool,        // All cases covered
    uncovered_patterns: [Pattern; 16],  // Missing patterns
    n_uncovered: i64,
}

// Check if patterns are exhaustive for scrutinee type
fn check_exhaustiveness(ctx: TypeContext, patterns: [Pattern; 64], n_patterns: i64, scrutinee_type: i64) -> MatchCheckResult with Panic {
    // For each constructor of scrutinee type, check if covered by a pattern
}
```

### Algorithms

1. **Constructor Coverage**
   - For each type, identify all constructors:
     - Enum type: all variants
     - Struct type: one constructor
     - Bool: `true` and `false`
     - Integer: literal or wildcard

2. **Pattern Specialization**
   - Given pattern `P` and constructor `C`, compute residual patterns
   - Used to recursively check inner patterns

3. **Refinement (GADT)**
   - Pattern may refine type information
   - Example: pattern `Some(x)` refines to `x: T` if scrutinee is `Option<T>`

### Helper Functions

- `check_pattern(ctx, pat, expected_type)` → PatternCheckResult
- `check_match_arms(ctx, patterns, scrutinee_type)` → MatchCheckResult
- `infer_pattern_type(pat)` → i64 (type_idx)
- `pattern_binds_variables(pat)` → [Binding; 16]
- `are_patterns_disjoint(pat1, pat2)` → bool
- `check_pattern_exhaustiveness(scrutinee_type, patterns)` → bool

### Example

```sio
match option_value {
    Some(x) => x + 1,      // Pattern `Some(x)` binds `x: i64`
    None    => 0,          // Pattern `None` binds nothing
}
// Exhaustive: covers `Some` and `None`

match flag {
    true  => 1,
    false => 0,
}
// Exhaustive: covers both booleans

match number {
    0 => "zero",
    _ => "nonzero",
}
// Exhaustive: wildcard covers all other cases
```

### Key Insight

Pattern matching serves two purposes:
1. **Type checking**: Verify pattern type matches scrutinee
2. **Exhaustiveness**: Ensure all cases are handled (catches incomplete matches)

Missing exhaustiveness is a common bug; Sounio compiler should warn or error on non-exhaustive matches.
