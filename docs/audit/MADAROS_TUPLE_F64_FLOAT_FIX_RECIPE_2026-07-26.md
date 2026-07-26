# Recipe: float-type `(f64, f64)` import unpack (for Claude native/lower)

**Status:** residual open — product mitigates; this is the true lower fix  
**Witness gate:** `scripts/ci/madaros_imported_f64_mul_gate.sh` → want `MADAROS_IMPORTED_F64_MUL_FIXED`  
**Owner:** Claude-1 (`lower.sio` claim) — Grok cannot write that file while claimed

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

Near `lower_opt_return_float_code` (~3915):

```sounio
// After array-of-f64 check, add:
// if return type is TypeTuple and every element is f64 → return 2
```

Need a helper `lower_opt_type_is_tuple_of_all_f64(opt) -> bool` walking TypeTuple.

### 2) Already present: `let r = mk()` binds float-element local when returns_float==2

`expr_result_is_f64_array_ref` (~10050) + bind at ~10240 already do this for
returns_float==2. Once (1) returns 2 for f64 tuples, `let __tup = ret_pair()`
marks `__tup` as float-element.

### 3) Field access: float-element base ⇒ FieldGet is float

In `lower_field_access_expr_ref` (~12376), extend:

```sounio
var field_is_float = lo1.field_is_float_for_base_ref(...) || lo1.field_is_float_by_name_simple(...)
// ADD:
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
