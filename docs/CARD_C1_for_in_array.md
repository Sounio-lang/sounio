# Card C1 — for-in over arrays (lowering)  [native-v2 self-hosting punch-list]

**Owner file:** `self-hosted/ir/lower.sio` ONLY. (Coordinate with the lowerer owner — this is
the same file as Card C; do this as a sub-task, not a parallel editor.)
**Impact:** #1 self-contained `ir_bodies_failed` cause — ~47/272 programs (confirmed by a
50-file reduction sweep, 2026-06-05). Every agent independently reduced to this construct.

## The gap
`for x in <iterable>` only handles `ExprRange`. Array/collection iterables hit the `else →
self.report_error().emit_unit()` branch → `ir_bodies_failed`. See the ExprForIn case in
`lower_expr_ref` (~line 6061): `if (*iter_box).kind == ExprKind::ExprRange { ... } else {
report_error }`.

Confirmed by reduction:
- `for v in 0..1` → ELF_OK (ranges work)
- `for v in [1]` → ir_bodies_failed (array literal)
- `let a=[1]; for _ in a {}` → ir_bodies_failed (array local)
- `fn f(v:[i64]){ for x in v {} }` → ir_bodies_failed (slice param)
- `for c in "hello"` → ir_bodies_failed (string) — defer, separate

## The fix — desugar to an index loop
`for x in arr { body }`  ==>
```
i = 0
top:
  if i >= LEN goto end          // ir_binop(cmp, i, OpLt, len) ; ir_branch_false(cmp,end)
  x = arr[i]                    // ir_index_get(x_reg, base, i)  (already exists, see ExprIndex ~5974)
  body
cont:                           // push_loop_labels(cont,end) so break/continue work
  i = i + 1
  goto top
end:
```
Mirror the EXISTING range branch right above (it already does fresh_reg/fresh_label/
ir_label/ir_branch_false/push_loop_labels/lower_block_ref/ir_jump). Bind `x` via
`bind_local(e.name, x_reg, true)` and re-copy `x = arr[i]` each iteration.

## The one piece of infrastructure needed: LEN (array length)
The lowerer does NOT track array element counts. Add it, mirroring how wide-int `limbs`
are already tracked:
1. **Type AST already has it:** `TypeArray.array_size: Option<Box<Expr>>` (the `N` in `[T;N]`,
   usually an `ExprIntLit`).
2. At `let`/`var` with a `[T; N]` annotation, record the element count for the local — add an
   `elem_count` alongside the existing `bind_local_wide(... limbs)` mechanism
   (`bind_local`@2582, `bind_local_wide`@2587, `lookup_local_limbs`@2619 are the template).
   Add `bind_local_array(name,reg,is_mut,elem_count)` + `lookup_local_elem_count(name)`.
3. In the for-in desugar, get LEN by iterable kind:
   - `ExprArrayLit` → count its `args` list (static, no tracking needed) — do this case FIRST.
   - `ExprIdent` → `lookup_local_elem_count(name)`; if found (>0), use a const LEN reg.
   - slice param `[i64]` (no static size) → for now still `report_error` (runtime-length,
     separate follow-up) — but the fixed-`[T;N]` local + literal cases cover the bulk.

## Acceptance
- `for v in [1,2,3] { s=s+v }` and `let a:[i64;3]=[1,2,3]; for v in a {...}` compile to ELF and
  run with correct sum.
- `examples/algorithms/sieve.sio` and the ~47 for-in-array programs: `ir_bodies_failed` count
  drops in the census; `ELF_OK` rises.
- capgate `tests/native_v2_capgate/run.sh` stays 31/31. No other census category rises.
- Small commit, push, PR to `feat/exact-orc-machinery`.
