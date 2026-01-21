# WP-2.1: Expression Checking (Type Checker Core)

## Sounio Syntax Rules (CRITICAL)

- Use `var` for mutable variables (NOT `let mut`)
- Use `&!T` for mutable references (NOT `&mut T`)
- Array indexing requires `with Panic` effect
- While loops may require `with Div` effect
- NO compound assignments: expand `x += 1` to `x = x + 1`
- NO type suffixes: use `0` not `0i64`
- Array repeat with helper functions: `[helper(); N]` not `[Struct{...}; N]`
- NO keyword as variable name

## Reference Implementation

See: `compiler/src/check/mod.rs` (check_expression function)
See: `compiler/src/hir/expr.rs` (HirExprKind enum)

## Target Output

**File**: `stdlib/compiler/check/expr.sio`
**Estimated LOC**: ~1,500

## Specification

Implement expression type checking for all HirExprKind variants:

### Core Expressions

1. **Literals** (Int, Float, Bool, String, Char, Byte)
   - `check_literal(ctx, lit)` → (type_idx, errors)
   - Infer type from literal kind
   - Example: `5` → `i64`, `3.14` → `f64`, `true` → `bool`

2. **Variables & Paths**
   - `check_var(ctx, name)` → type_idx
   - Look up variable in context scopes
   - Return error if undefined

3. **Binary Operations** (Add, Sub, Mul, Div, Mod, And, Or, Eq, Lt, Gt, etc.)
   - `check_binop(ctx, op_kind, left_type, right_type)` → result_type
   - Validate both operands have compatible types
   - Example: `Add` expects numeric types on both sides

4. **Unary Operations** (Neg, Not, Deref, Ref, Borrow)
   - `check_unop(ctx, op_kind, arg_type)` → result_type
   - Example: `Not` expects `bool`, `Neg` expects numeric

5. **Function Calls**
   - `check_call(ctx, fn_type, arg_types)` → return_type
   - Validate argument count matches parameters
   - Infer return type from function signature

6. **Array Operations** (Index, Slice, Concat)
   - `check_array_index(ctx, array_type, index_type)` → elem_type
   - Validate array indexing produces correct element type
   - Slice returns array type

7. **Struct/Record Operations**
   - `check_field_access(ctx, struct_type, field_name)` → field_type
   - Validate field exists on struct
   - Return field type

8. **Type Casts & Coercions**
   - `check_cast(ctx, from_type, to_type)` → bool
   - Validate cast is legal (e.g., int→float OK, float→int needs explicit cast)

9. **If/Match Expressions**
   - `check_if(ctx, cond_type, then_type, else_type)` → unified_type
   - Unify then and else branches to same type
   - Condition must be `bool`

10. **Block & Sequence**
    - `check_block(ctx, stmts, expr)` → expr_type
    - Chain effects from statements
    - Final expression type is block type

### Data Structures

```sio
struct ExprCheckResult {
    type_idx: i64,       // Inferred type index
    effect_set: i64,     // Effects (bitmask)
    has_error: bool,     // Error occurred
    error_msg: [i8; 256], // Error text
}

struct BinOpTable {
    op_kind: i32,
    left_type_idx: i64,
    right_type_idx: i64,
    result_type_idx: i64,
}
```

### Helper Functions

- `check_expr_kind(ctx, kind, args)` → ExprCheckResult
- `infer_type_from_literal(lit_kind)` → i64 (type_idx)
- `lookup_binary_op(op, left_t, right_t)` → i64 (result_type)
- `check_type_compat(ctx, type1, type2)` → bool

### Integration Points

- Calls `context_add_error()` from context.sio when type mismatch occurs
- Uses `type_pool_get()` from type.sio to retrieve type info
- Chains effects using bitwise OR: `effect_set = effect_set | callee_effects`

### Testing

Example expressions to handle:
```
// Literals
let x = 42;           // i64
let y = 3.14;         // f64
let b = true;         // bool

// Binary ops
let z = x + 5;        // i64
let w = y * 2.0;      // f64
let cond = x > 0;     // bool

// Function calls (assuming `fn add(a: i64, b: i64) -> i64`)
let result = add(x, 10);

// Array indexing
let arr: [i64; 10] = ...;
let elem = arr[0];    // i64

// Field access (assuming `struct Point { x: i64, y: i64 }`)
let p: Point = ...;
let px = p.x;         // i64
```

### Key Insight

Expression checking is bottom-up: check subexpressions first, then combine their types. This matches bidirectional type inference in `compiler/src/check/mod.rs`.
