<!-- docs:meta
topic_id: repo.docs.handoff.exact-engine-prereqs
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.exact-engine-prereqs
-->

# Exact Cayley-Dickson `<F>` engine — full compiler prerequisite set

**Author:** Claude (exact-algebra-core lane), 2026-07-05
**Consumer:** `stdlib/algebra/cayley_dickson_exact.sio` (Phase-2 skeleton, does not compile yet)
**Purpose:** the operator chose "fix the compiler first" so the exact engine can be *generic in the coefficient field `F`*. Drafting the skeleton revealed that generic-in-`F` needs **three** compiler features, not one. This doc enumerates them (spike-verified), maps who covers what, and states the zero-prerequisite fallback that ships the same science today.

---

## Spike-verified capability matrix (souc v0.80.0, 2026-07-05)

All spikes are `//@ run-pass` micro-files; both engines exit rc=0 on failure (compact-stub false-green), so the signal is stdout (`parse error` / `error[E0xx]` / `preflight failed`), never the exit code.

| # | Construct the `<F>` engine needs | Status | Evidence |
|---|---|---|---|
| — | generic struct decl + inline instantiation + field access | ✅ works | `generic_struct_instantiate.sio`, `generic_struct_nested.sio` pass |
| — | inherent `impl Type {…}` + `Self`-as-type | ✅ works | spikes p2/p3; used throughout stdlib (`Complex`, `HeapVec`) |
| — | empty `trait Name {}` (no methods) | ✅ parses | spike `tr_empty.sio` |
| **1** | **fn param/return of a generic struct** (`fn cd_mul_exact<F>(a: CDElementExact<F>) -> CDElementExact<F>`) | ❌ **BROKEN** | `turbofish.sio` known-failure; spike → `expected CDExact, found CDExact__T` + lean_single "tail type mismatch"@27281 |
| **2a** | **bodyless trait-method signatures** (`trait ExactRing { fn er_add(self,o:Self)->Self }`) | ❌ **BROKEN** | spike `tr_sig.sio` → `parse error expected=184` at the method-sig line; `stdlib/quantum/vqe.sio`'s `trait Ansatz` extracted standalone fails identically. CORRECTION: an earlier full-file check wrongly reported traits "parse" — full-file error-recovery masked it; isolated files are the only trustworthy signal. |
| **2b** | **`impl Trait for Type`** (needed to give `i64`/`Rational` the `ExactRing` methods) | ❌ **BROKEN** | spike → `parse error` at the `for` keyword (`expected=184 actual=23`). NB: occurrences DO exist in `vqe.sio`/`pac.sio`/`examples/*` but all are aspirational/non-compiling. |
| **3** | **trait-bounded generic method dispatch** (`fn f<F: ExactRing>(x: F) { x.er_mul(y) }`) | ❌ **BROKEN** | spikes t1b/t8b (multi-line, correctly formatted) still fail; depends on 2a + 2b |

Rejected alternative — **fn-pointer vtable** (`struct RingOps<F> { mul: fn(F,F)->F, … }`): avoids traits but fails independently with `E016 expected fn#0 found fn#2` **even for a concrete, non-generic `OpsI64`** (spikes t4/t5) — the checker won't unify a named function with a `fn(...)->...` field type. So it needs a *fourth* fix (fn-type-identity unification) **plus** #1. Not cheaper than traits.

## Who covers what

- **#1 is in flight** — `docs/handoff/compiler_generic_struct_return_fix_prompt.md` (fable5 / Claude #2 lane), PR #631 lands the prompt.
- **#2 (2a+2b) and #3 are NOW commissioned** — `docs/handoff/compiler_impl_trait_for_type_fix_prompt.md` (covers both parser bugs 2a bodyless-trait-method-sigs + 2b `impl Trait for Type`) and `docs/handoff/compiler_trait_bounded_dispatch_fix_prompt.md` (#3, depends on #2). These are separate, larger features than #1; the fable5 generic-struct-return prompt does **not** cover them. So generic-in-`F` needs **#1 + 2a + 2b + #3** — strictly more than first stated.

**Net:** landing only #1 does **not** unblock `cayley_dickson_exact.sio`. The generic-in-`F` path needs #1 **and** #2 **and** #3.

## The zero-prerequisite fallback (recommended to ship the science now)

Zero-divisor detection is homogeneous over ℤ (±1 sign algebra — see Phase-1 report / `formal/lean4/SounioZeroDivisorBridge.lean`). So a **hand-monomorphized concrete-`i64` engine** — `struct CDElementExactI64 { c: [i64; 2048], bits }` with non-generic `cd_mul_exact_i64` / `cd_associator_exact_i64` / `zd_exact_i64` — needs **none** of #1/#2/#3 (it is the same shape as the existing f64 `CDElement`/`cd_mul`, which compiles today). It **fully proves the n=4 sedenion 168-class census** and the entire annihilation Definition-of-Done. `Rational` coefficients (norms only, explicitly out of the annihilation task) would be a second hand-monomorphized twin if ever needed.

The `<F>` engine is strictly a *code-sharing / ergonomics* win over the concrete engine — it changes no scientific result.

## Decision for the operator

Two coherent paths:

- **A — Ship concrete-i64 now, keep `<F>` as the north star.** Implement `CDElementExactI64` today (Phases 2–4 proceed immediately, no compiler dependency); let #1 land via fable5; commission #2/#3 later; migrate the concrete engine to `<F>` when all three are green. Fastest to the 168-census artifact; the skeleton (`cayley_dickson_exact.sio`) is the migration target.
- **B — Hold for the full generic stack.** Commission #2 + #3 alongside #1 before writing any engine. Cleanest end state, but the algebra work stays parked across three compiler features instead of one.

Recommendation: **A** — the concrete-i64 engine is the same code the skeleton would monomorphize to, so nothing is wasted, and the headline (Sounio *executes* the exact product and *proves* annihilation at n=4) lands without waiting on a three-feature compiler stack.
