<!-- docs:meta
topic_id: repo.docs.handoff.continuity.wp-a2
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.continuity.wp-a2
-->

# WP-A2 — Madaros: trait-method dispatch on PRIMITIVE receivers [Opus] (dep: WP-A0; parallel with A1/A3)

## Problem
After the phase-1 AST specializer, a generic fn `fn use2<F: R>(a: F, b: F) -> F { a.radd(b) }` instantiated at `<i64>` leaves a method call `a.radd(b)` with `a: i64`. `impl R for i64 { fn radd(self, o: Self) -> Self { self + o } }` parses (phase-1 grammar) and its method registers, but the checker rejects the CALL: `error[E019]: method calls are not supported for this type` (bare primitive receiver), and `trait_bounded_dispatch_multi_call.sio` also shows `E011 no method named for this type` / `E009 argument type mismatch`. Struct receivers work (`trait_bounded_dispatch_struct.sio` is GREEN) — the gap is primitives only.

This mirrors the lean_single fix that landed in PR #650 (there: primitive receiver hash fallback at the method-lookup site + bare `self` typed with the primitive VAR_TY instead of struct). Madaros needs the same semantics in its own architecture.

## Where to look (verified anchors, post-phase-1 tree)
- Checker method-call path: `self-hosted/check/check.sio` — method resolution around `checker_check_call_inplace`/method lookup; E019/E011/E009 message sites via `print_error_message` (`check.sio:~11314-11488`; E011 = "no method named for this type"). Find the branch that gates method calls on the receiver being a struct (`TyNamed` + `struct_table_find`) and extend it: when the receiver type is a primitive (i64/i32/f64/bool), look the method up against impls registered for that primitive's type NAME (how `impl ExactRing for i64` registers is visible in the phase-1 parser work: `parser/items.sio` `parse_impl_item` — receiver name flows like an inherent impl; `ImplDef.trait_name` at `parser/ast.sio:1153`).
- `Self` in impl bodies resolves via `current_impl_type` (`check.sio:1681, 14031`) — verify it also resolves when the impl target is a primitive name (`impl R for i64` → `Self` = i64), and that bare `self` params get the primitive type, not a struct type.
- IR lowering method mangling: `self-hosted/ir/lower.sio` — `lower_method_recv_type` + `ir_mangle_method_name` (see also the phase-1 println fix nearby at `expr_result_scalar_kind_ref`, `lower.sio:~4893`). Ensure a method on an i64 receiver lowers to the same mangled symbol the impl's fn was emitted under.

## Witnesses (exit-code, NOT println — see BOOTSTRAP guardrail)
W1 (asymmetric — catches operand swaps):
```
trait R { fn rmix(self, o: Self) -> Self }
impl R for i64 { fn rmix(self, o: Self) -> Self { self * 10 + o } }
fn use2<F: R>(a: F, b: F) -> F { a.rmix(b) }
fn main() -> i64 with IO { return use2::<i64>(2, 3) }   // rc MUST be 23
```
and a variant returning `use2::<i64>(7, 4)` → rc 74.
W2 (two impls of one trait — disambiguation):
```
trait E { fn ea(self, o: Self) -> Self }
impl E for i64 { fn ea(self, o: Self) -> Self { self + o } }
struct Wrap { v: i64 }
impl E for Wrap { fn ea(self, o: Self) -> Self { Wrap { v: self.v * o.v } } }
fn g<F: E>(a: F, b: F) -> F { a.ea(b) }
// main #1: return g::<i64>(20, 3)                        → rc 23
// main #2: return g::<Wrap>(Wrap{v:20}, Wrap{v:3}).v     → rc 60
```
W3: `tests/run-pass/trait_bounded_dispatch.sio` → prints `5` / `spike PASS`, rc=0 (this one may use println of a call result — that path was fixed in phase 1 and is allowed).
W4: `tests/run-pass/trait_bounded_dispatch_multi_call.sio` → `11`/`5`/`6`/`multi_call PASS`.

## Validation battery
- Build cycle: `bash scripts/ci/build_modular_madaros.sh /tmp/madaros-a2`; run witnesses with actual rc/stdout checks.
- Still green: `trait_bounded_dispatch_struct.sio` (10/struct PASS), `impl_trait_for_type{,_multi}.sio`, `turbofish.sio` 3/3, `generic_struct_basic.sio`, compile-fail `tests/compile-fail/turbofish_type_arg_arity.sio` still REJECTED (E010).
- Umbrella before/after: `bash scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh` — pre-existing reds per SCOREBOARD.md; zero NEW reds.
- 10-test diverse regression sample vs your pre-change build, byte-identical.

## Done criteria
W1–W4 pass output-verified; battery green; PR merged; scoreboard + handoff updated.
