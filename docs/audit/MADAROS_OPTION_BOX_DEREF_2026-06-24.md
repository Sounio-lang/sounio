<!-- docs:meta
topic_id: repo.docs.audit.madaros-option-box-deref-2026-06-24
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-option-box-deref-2026-06-24
-->

# Madaros recursive data — `Some(nx) => (*nx).field` works (2026-06-24)

*Branch off `main`. `match opt { Some(nx) => (*nx).field }` on an `Option<Box<T>>` now
works; previously it crashed (139). This unblocks recursive data structures — linked lists,
trees — whose traversal goes through exactly this pattern.*

## Root cause

`Option<Box<T>>` is a nullable pointer: `Some` carries the **Box handle**, `None` is 0.
`bind_match_sub_pattern` bound the `Some(nx)` payload via `bind_local(nx, scrut)` **without
recording that `nx` is a Box**. So a body `(*nx).field` hit the `is_box_deref` check
(`lookup_local_struct_type(nx) == "Box"`), found it **false**, and lowered `*nx` as a **raw**
`IrUnaryOp(OpDeref)` — `mov rax,[rax]` on a *handle* (not a raw address) → bad load → crash.
A plain `let b = Box::new(…); (*b).v` works because the let-binding *does* tag the local
`"Box"`; the `Some`-bound payload was the one place that didn't.

## Fix
1. **`bind_match_sub_pattern`** (`lower.sio`): after binding the `Some(nx)` payload, tag it
   `"Box"` (`bind_local_struct_type(nx, "Box")`), so `*nx` lowers to the box-handle deref
   (`IrFieldGet 0`). Safe for non-Box payloads: the tag only changes how `*nx` lowers, and
   deref/field-access of a non-Box payload (e.g. `Option<i64>`) is not valid code anyway —
   `Option<i64>` `Some(v) => v` is unaffected, and `Option<struct>` direct field access does
   not compile today regardless.
2. **`lower_let_stmt`** (`lower.sio`): `let b = <ident>` now propagates the source variable's
   struct type, so a re-bound box (`let b = nx; (*b).v`) also resolves correctly.

## Verified (madaros from this source, `ulimit -s unlimited`)
- Recursive linked-list `sum(&node) → 42`; `cnt(&node) → 2`; 3-node list `→ 42`.
- `Some(nx) => (*nx).v → 42`; `let b = nx; (*b).v → 42` (ident propagation).
- No regression: `Option<i64>` `Some(v) => v+1 → 42`, `Some(40) => v+2 → 42`; 53/90 run-pass =
  prebuilt main +6, 0 regressed; madaros self-builds.

## Honest scope
- Fixes the `Option<Box<T>>` payload deref. `Option<struct>` (unboxed payload) direct field
  access is a separate, still-open gap (does not compile). Enum *payload* patterns
  (`E::A(x)`) are a separate frontend gap (E005). Array `++` (true array concat) is a
  separate codegen gap.
- The Box tag on the `Some`-payload is a structural heuristic (the payload of `Some` on an
  `Option<Box<T>>` is always a Box); it is not type-driven (the lowerer does not track field
  types), but it is sound for the reasons above.

## AI disclosure
Fix by AI agent (Claude) under human direction; root isolated by a decisive probe ladder
(construct/None/ref-param work; `Some(nx)=>(*nx)` crashes) and confirmed against the
`is_box_deref` lowering. Every claim backed by a re-runnable probe.
