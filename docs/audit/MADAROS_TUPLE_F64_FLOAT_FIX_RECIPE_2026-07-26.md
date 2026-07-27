<!-- docs:meta
topic_id: repo.docs.audit.madaros-tuple-f64-float-fix-recipe-2026-07-26
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-tuple-f64-float-fix-recipe-2026-07-26
-->

# Recipe: float-type `(f64, f64)` import unpack (for Claude native/lower)

**Status:** **APPLIED** (2026-07-27) — branch `fix/madaros-tuple-f64-float-20260727`  
**Witness gate:** `scripts/ci/madaros_imported_f64_mul_gate.sh` → `MADAROS_IMPORTED_F64_MUL_FIXED`  
**Evidence:** rebuilt Madaros; `tuple ga*ga=0.250000`, `tuple gv*gv=0.001412`

## Mechanism

Parser desugars:

```sounio
let (a, b) = f()
```

to:

```sounio
let __tupN = f()
let a = __tupN.0
let b = __tupN.1
```

(`self-hosted/parser/stmts.sio` tplet_*).

Tuples lower as **array-like alloc** (`lower_tuple_literal_expr_ref`).  
`.0` / `.1` are **field gets** by numeric name.

If `__tupN` is not marked **float-element local**, FieldGet of `.0`/`.1` omits
`IR_FLOAT_REG_MARKER_FLAG` → integer mul path → `ga*ga` → 0.

Scalar imported `f64` already works (`returns_float == 1` on call sites).

## Surgical fix (3 sites in `self-hosted/ir/lower.sio`)

### 1) Classify `(f64, f64)` return as float-element aggregate (`returns_float = 2`)

Insert helpers next to `lower_opt_type_is_array_of_f64` (~3903), then extend
`lower_opt_return_float_code`:

```sounio
fn lower_type_expr_is_tuple_of_all_f64(te: &TypeExpr) -> bool with Mut, Panic, Div {
    if (*te).kind != TypeExprKind::TypeTuple { return false }
    match (*te).type_args {
        Some(list0) => {
            var cur: Option<Box<TypeExprList>> = Some(list0)
            var any = false
            while true {
                match cur {
                    Some(list) => {
                        if !lower_type_expr_is_f64(&(*list).head) { return false }
                        any = true
                        cur = (*list).tail
                    }
                    None => { break }
                }
            }
            any
        }
        None => false,
    }
}

fn lower_opt_type_is_tuple_of_all_f64(opt: &Option<Box<TypeExpr>>) -> bool with Mut, Panic, Div {
    match *opt {
        Some(te) => lower_type_expr_is_tuple_of_all_f64(&(*te)),
        None => false,
    }
}

fn lower_opt_return_float_code(opt: &Option<Box<TypeExpr>>) -> i64 with Mut, Panic, Div {
    if lower_opt_type_is_f64(opt) {
        1
    } else if lower_opt_type_is_array_of_f64(opt) || lower_opt_type_is_tuple_of_all_f64(opt) {
        2
    } else {
        0
    }
}
```

Update the comment above `lower_opt_return_float_code`: `2` = f64 array **or**
all-f64 TypeTuple (same float-element local flag).

### 2) Already present: `let r = mk()` binds float-element local when returns_float==2

`expr_result_is_f64_array_ref` (~10050) + bind at ~10240 already do this for
returns_float==2. Once (1) returns 2 for f64 tuples, `let __tup = ret_pair()`
marks `__tup` as float-element.

### 3) Field access: float-element base ⇒ FieldGet is float

In `lower_field_access_expr_ref` (~12371), replace the single `let field_is_float = ...`
line with:

```sounio
var field_is_float = lo1.field_is_float_for_base_ref(&e.left, e.name) || lo1.field_is_float_by_name_simple(e.name)
// Tuple desugar: let __tup = f(); let a = __tup.0 — float-element base
// means .0/.1 are f64 even without a struct layout.
match e.left {
    Some(be) => {
        if (*be).kind == ExprKind::ExprIdent {
            if lo1.lookup_local_array_elem_float((*be).name) {
                field_is_float = true
            }
        }
    }
    _ => {}
}
```

## Acceptance

```bash
export MADAROS_RAW_BIN=... # current-source Madaros with the patch
bash scripts/ci/madaros_imported_f64_mul_gate.sh
# expect: MADAROS_IMPORTED_F64_MUL_FIXED
# (scalar a*a and tuple ga*ga both 0.25)
```

## Product mitigations already on main

- #1516: avoid vertex tuple unpack in `nonunitary_amp`  
- Witness residual marker documents open compiler fix

## Coordination

Grok claims only free files (module_frontend/vertex/amplitude).  
When you free `lower.sio` for 30 min, apply (1)+(3) above; (2) is already wired.
