<!-- docs:meta
topic_id: repo.docs.audit.madaros-impl-methods-three-layer-fix-2026-06-23
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-impl-methods-three-layer-fix-2026-06-23
-->

# Madaros impl methods — working end-to-end (three-layer fix, 2026-06-23)

*Branch:* `fix/madaros-impl-method-checker` (off `main`).
*Result:* `obj.method()` now parses, type-checks, compiles, and runs correctly under Madaros.
Refines/corrects `MADAROS_IMPL_METHOD_TWO_ROOTS_2026-06-23.md` (which mis-attributed the
crash to a checker by-value miscompile).

## Summary

Method support was broken at **three independent layers**, each masking the next. The
triage's "checker by-value miscompile" hypothesis was wrong — instrument-and-observe
(per-step tracing, the same method that cracked the struct-field bug) showed the first
crash is in the **parser**, before the checker ever runs. Three small fixes make methods
work end-to-end:

| Layer | Symptom | Root | Fix |
|---|---|---|---|
| 1. Parser | `impl C { fn … }` SIGSEGV at parse | `parse_impl_item` called `parse_fn_item(false, false)` — the **bool `false` was passed for the `visibility: Visibility` struct param** (ABI mismatch; built only because the type error is a non-fatal warning) | pass `visibility_private()` |
| 2. Checker | `obj.method()` → **E011** (method not found) | the `*mut` collect spine **no-op'd `impl`** (method sigs never registered) | add `checker_collect_impl_item_bridge` and wire `ItemImpl` into `checker_collect_item_inplace` (mirrors the existing check bridge) |
| 3. Codegen | type-checks, then SIGSEGV at lowering | `lower_method_call_expr_ref` resolved the call by the **bare** method name, but impl methods are lowered as **mangled** free fns (`ir_mangle_method_name`) → call to a body-less fn | derive the receiver's struct type and look up the **mangled** name |
| 3b. Receiver type | `self.method()` / `param.method()` SIGSEGV (layer-3 crash) | function **params were bound with an empty `struct_type`** (both the functional `lower_fn_params_ref` and the `*mut lowerer_lower_fn_params_mut`), so the receiver type was unknown for `self`/struct params → bare-name fallback | register each param's struct type at binding (`lower_param_struct_type_name`, unwrapping `&`/`&!`; `*mut` setter `lowerer_bind_local_struct_type_mut`) |

Note layer 1's exit-139 was briefly confounded with ROOT 1's stack overflow (testing the
raw binary without `ulimit`); with `ulimit -s unlimited` (which `bin/madaros` sets) the
layers separate cleanly.

## Files

- `self-hosted/parser/items.sio` — `parse_fn_item(visibility_private(), false)` at the two
  `(false, false)` sites (impl methods + study-block fns).
- `self-hosted/check/check.sio` — `checker_collect_impl_item_bridge` + `ItemImpl` arm in
  `checker_collect_item_inplace`.
- `self-hosted/ir/lower.sio` — `lower_method_recv_type` helper + mangled call name in
  `lower_method_call_expr_ref`.

## Verified (madaros built from this exact source)

- `impl C { fn m() {} }` → `--check` OK (was SIGSEGV).
- Local receiver: `c.five() → 5`; `c.get()/c.add() → 40 42`; `method1 → 42`;
  nested `a.plus(a.val()) → 200`.
- **`self.method()`** (sibling call `a.twice() → 42`) and **`param.method()`**
  (`relay(x).get() → 99`) — both were SIGSEGV before the param-struct-type fix.
- No regression: the param-struct-type change (the only broad one — it touches all params)
  is **measured neutral** — identical 32/60 run-pass exit-0 on `madaros` with vs without it,
  zero changed tests. The other three changes are gated to impl/method/method-call code
  paths (unreachable from non-method programs). madaros self-builds.

## Honest scope

- Receiver-type resolution handles **local / `self` / struct-param** receivers
  (via the now-registered param/local struct types). Method calls on arbitrary
  sub-expression results (e.g. `f().method()`) fall back to the bare name and are out of
  scope here.
- Independent of (and composes with) the struct-field-index fix #413 — `self.<field>` on a
  by-ref receiver already worked; local struct-field reads still need #413.

## AI disclosure
Fix by AI agent (Claude) under human direction; every claim backed by re-runnable
`madaros compile/--check/run` commands and the per-step traces used to localise each layer.
