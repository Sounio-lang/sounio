# WP-2.2: Statement & Item Checking (Type Checker Statements)

## Sounio Syntax Rules (CRITICAL)

- Use `var` for mutable variables (NOT `let mut`)
- Use `&!T` for mutable references (NOT `&mut T`)
- Array indexing requires `with Panic` effect
- NO compound assignments: expand `x += 1` to `x = x + 1`
- NO type suffixes
- Array repeat with helper functions

## Reference Implementation

See: `compiler/src/check/mod.rs` (check_statement, check_item)
See: `compiler/src/hir/stmt.rs` (HirStmtKind enum)

## Target Output

**File**: `stdlib/compiler/check/stmt.sio`
**Estimated LOC**: ~1,000

## Specification

Implement statement and top-level item type checking.

### Statements

1. **Variable Declaration** (`let x: T = expr;`)
   - `check_let(ctx, name, type_hint, init_expr)` → (ctx, type_idx)
   - Infer type from init_expr if no type hint
   - Add binding to context
   - Return context with new binding added

2. **Assignment** (`x = expr;`)
   - `check_assign(ctx, target, value)` → ctx
   - Look up target variable in context
   - Check value type matches target type
   - Return updated context

3. **Expression Statement** (`expr;`)
   - `check_expr_stmt(ctx, expr)` → (ctx, type_idx, effects)
   - Type check the expression
   - Discard result (type is unit)

4. **Return Statement** (`return expr;`)
   - `check_return(ctx, expr)` → (type_idx, effects)
   - Check expr type matches function's declared return type
   - Mark function as returning

5. **While Loop** (`while cond { body }`)
   - `check_while(ctx, cond, body_stmts)` → ctx
   - Condition must be `bool`
   - Body statements checked in same scope
   - Return updated context

6. **Break/Continue**
   - `check_break(ctx)` → ctx
   - `check_continue(ctx)` → ctx
   - Only valid inside loops (check loop nesting depth)

### Top-Level Items

1. **Function Definitions**
   - `check_fn_def(ctx, name, params, return_type, body)` → ctx
   - Check parameter types are well-formed
   - Check function body type checks
   - Add function signature to context
   - Example: `fn add(a: i64, b: i64) -> i64 { a + b }`

2. **Struct Definitions**
   - `check_struct_def(ctx, name, fields)` → ctx
   - Check field types are well-formed (no cycles)
   - Add type definition to context
   - Example: `struct Point { x: i64, y: i64 }`

3. **Enum Definitions**
   - `check_enum_def(ctx, name, variants)` → ctx
   - Check variant payloads are well-formed
   - Add type definition to context

4. **Type Alias**
   - `check_type_alias(ctx, name, target_type)` → ctx
   - Validate target_type exists
   - Add alias to context

### Data Structures

```sio
struct StmtCheckResult {
    context: TypeContext,
    effects: i64,
    has_error: bool,
}

struct Binding {
    name: [i8; 64],
    type_idx: i64,
    is_mutable: bool,
}
```

### Helper Functions

- `check_stmt_kind(ctx, kind)` → StmtCheckResult
- `check_item_kind(ctx, kind)` → StmtCheckResult
- `add_local_binding(ctx, name, type_idx, is_mutable)` → ctx
- `check_type_exists(ctx, type_idx)` → bool

### Integration Points

- Calls `context_add_error()` for type mismatches
- Calls `context_push_scope()` before function/loop body
- Calls `context_pop_scope()` after block
- Uses `context_lookup_type_def()` to verify types exist
- Chains effects from substatements

### Testing

Example program:
```
struct Point {
    x: i64,
    y: i64,
}

fn distance(p1: Point, p2: Point) -> i64 {
    let dx = p1.x - p2.x;
    let dy = p1.y - p2.y;
    dx + dy
}

fn main() -> i64 {
    let p1: Point = Point { x: 0, y: 0 };
    let p2: Point = Point { x: 3, y: 4 };
    distance(p1, p2)
}
```

### Key Insight

Statement checking manages the context as statements are processed. Each statement can introduce new bindings, which are valid for subsequent statements. This requires threading the context through a sequence of checks.
