<!-- docs:meta
topic_id: repo.docs.handoff.compiler-trait-bounded-dispatch-fix-prompt
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.compiler-trait-bounded-dispatch-fix-prompt
-->

# Prompt — Trait-bounded generic method dispatch (unblocks exact-algebra `<F: Trait>`, prerequisite #3 of 3)

**For:** a fresh compiler-lane agent (fable-style), the follow-on to the agent that lands prerequisite #2
**Authored by:** Claude (exact-algebra-core lane, `coord/exact-algebra-core`), 2026-07-05
**Priority:** blocking prerequisite for the generic-in-`F` Exact Algebraic Core, **secondary to** the concrete-i64 engine which ships the same science without it
**Type:** compiler-internals (self-host + Madaros) — serialized surfaces, fixed-point + output-verified gates mandatory

---

## 0. One-line task

Make a generic function whose type parameter carries a **trait bound** (`fn f<F: Trait>(...)`) resolve calls like `a.method(b)` (where `a: F`) to the concrete trait-impl for whatever type `F` is monomorphized to, and emit correct per-instantiation code — e.g. `fn use2<F: R>(a: F, b: F) -> F { a.radd(b) }` called as `use2::<i64>(2, 3)` must return `5`. This is prerequisite **#3 of 3** in `docs/handoff/exact_engine_prereqs.md`. **Hard dependency: prerequisite #2 (`docs/handoff/compiler_impl_trait_for_type_fix_prompt.md`) must be merged first, in full** — that task actually covers two coupled parser bugs (verified 2026-07-05): (Bug A) a trait declaration with bodyless method signatures — the only useful form, and the one `R`/`ExactRing` use — fails to parse even with no `impl` in sight, and (Bug B) `impl Trait for Type` is rejected at the `for` keyword. Every repro in *this* task fails at Bug A/Bug B's exact parse sites until #2 lands both fixes. This task also assumes prerequisite **#1** (generic-struct-return monomorphization, `docs/handoff/compiler_generic_struct_return_fix_prompt.md`) is available, since the real consumer (`cd_mul_exact<F>(a: CDElementExact<F>, ...) -> CDElementExact<F>`) combines a trait-bounded type parameter with a generic-struct return.

**Before starting: verify #2 is actually merged (both Bug A and Bug B) and its acceptance criteria are green** (check `artifacts/omega/agent_handoff.log.md` for a RELEASE entry, and re-run this task's own spike below — if it still fails with the Bug A/Bug B parse errors, STOP, #2 is not actually done, do not attempt to route around it).

## 1. Why this matters (the consumer)

The operator asked for an **exact algebraic layer** beneath the f64 Cayley–Dickson/sedenion runtime: zero-divisor annihilation as decidable integer equality (`ab == 0` over ℤ) instead of tolerance-gated float. The full generic design (operator-approved) is:

```
trait ExactRing {
    fn er_add(self, o: Self) -> Self
    fn er_sub(self, o: Self) -> Self
    fn er_mul(self, o: Self) -> Self
    fn er_is_zero(self) -> bool
    fn er_eq(self, o: Self) -> bool
}
impl ExactRing for i64 { ... }        // prerequisite #2
impl ExactRing for Rational { ... }   // prerequisite #2

fn cd_mul_exact<F: ExactRing>(a: CDElementExact<F>, b: CDElementExact<F>) -> CDElementExact<F> {
    // ... calls a.c[i].er_mul(b.c[j]), a.c[i].er_add(...), etc.
}
```

`impl ExactRing for i64` (prerequisite #2) makes `i64` *carry* the methods. This task, prerequisite #3, is what lets **generic code** (`cd_mul_exact<F: ExactRing>`) actually **call** `a.er_mul(b)` when `a: F` and `F` is a bound type parameter — i.e. the checker must accept the bound, and the monomorphizer must, for each instantiation `F = i64` / `F = Rational`, resolve `er_mul` to the right concrete impl and emit correct code. Without this, the generic `ExactRing`-based engine cannot be called at all, even once #1 and #2 are both done. **Until #1+#2+#3 are all green, the exact-algebra lane ships a hand-monomorphized concrete-i64 engine** (`stdlib/algebra/cayley_dickson_exact_i64.sio`) that proves the same n=4 sedenion 168-class census with none of these three compiler features. Landing #3 is a code-sharing/ergonomics win at that point, not a new scientific result — see `docs/handoff/exact_engine_prereqs.md` for the full decision record.

## 2. The bug — exact reproduction

Reproduction is on `souc v0.80.0`, worktree off `origin/main`, **assuming #2 is merged** (if #2 is not merged, this repro fails identically to #2's own repro and tells you nothing new — see §0). Exit codes on this compiler are known to false-green at rc=0 on some stages; always read stdout, never trust rc alone.

**Repro — minimal** (source committed next to this doc as `docs/handoff/spike_trait_bounded_dispatch.sio`; also inline in §6):

```
trait R { fn radd(self, o: Self) -> Self }
impl R for i64 {
    fn radd(self, o: Self) -> Self { self + o }
}
fn use2<F: R>(a: F, b: F) -> F {
    a.radd(b)
}
fn main() with IO {
    let r = use2::<i64>(2, 3)
    println(r)
}
```

Run today (pre-#2, for the record): `./bin/souc run docs/handoff/spike_trait_bounded_dispatch.sio` → stdout (line numbers reflect the committed file's header comment; re-run to reconfirm if you edit the file):
```
parse error: expected token at line 27:1     (Bug A: trait R's own closing brace)
 expected=184
 actual=177
parse error: expected token at line 29:8     (Bug B: the `for` on `impl R for i64 {`)
 expected=184
 actual=23
parse error: expected token at line 42:1     (cascaded EOF)
 expected=185
 actual=0
Parse failed for module 0: 6 errors
error: parser reported 6 syntax errors
Compilation failed!
  error: type checking preflight failed
```
This is **#2's failure** (both Bug A and Bug B), not #3's — neither `trait R { fn radd(self, o: Self) -> Self }` nor `impl R for i64 {` have parsed yet. **Re-run this exact spike after #2 lands.** Whatever new failure mode appears at that point (most likely: `<F: R>` bound syntax rejected by the generic-fn parser, or the bound is parsed/ignored but `a.radd(b)` fails type-checking with an unresolved-method error since `F` is not statically known to have `radd`, or monomorphization emits code that calls the wrong/no concrete `radd`) is **this task's real repro** — document it in this section of a working copy of this prompt (or in the CLAIM log entry) before starting the fix, since it cannot be predicted exactly until #2's grammar is in place.

**Second fixture to add once #2 is in** (struct-typed `F`, not just a builtin): a struct implementing `R`, to make sure the fix isn't specific to builtin-type impls — see §4.2.

## 3. Root-cause map (starting points — verify, don't assume)

Engine 1 — `self-hosted/compiler/lean_single.sio` (35,897 lines):
- **Bound parsing:** the generic function parameter-list parser accepts `<F>` today (bare type params — used throughout, e.g. `make_wrapper<T>`). Extend it to accept `<F: TraitPath>` — parse an optional `: TraitPath` suffix per type parameter and record it (a new field alongside wherever type-param names are currently stored; do not silently discard it as it would today if the parser is lenient about trailing `:`).
- **Bound checking (minimal, not full trait-coherence):** at the call site `a.radd(b)` inside the generic function body, where `a: F` and `F` is a bound type parameter, the checker must NOT try to resolve `radd` as if `F` were a concrete type (it isn't, yet) — it should look up `radd` on the **bound trait** (`R`), confirm the trait declares a method of that name/signature, and accept the call provisionally. This is the same shape of problem "unresolved method on a type parameter" that any trait-bound system solves; keep it minimal — full trait-signature unification is not required, just enough to let the call through without breaking existing generic-fn type-checking for **unbounded** type params (must not regress `make_wrapper<T>`-style code with no bound).
- **Monomorphization / dispatch:** reuse the existing generic-fn instantiation machinery: `mono_mangle(ns,ne,ts,te)` @ **5740**, `MONO_GEN / MONO_TY_S / MONO_TY_E / MONO_IS_ST`, `mono_find_inst`, the generic-fn discovery loop @ **~25150–25325**, Pass 0c+0d monomorphization @ **26178**. For each concrete instantiation (`F = i64`), when emitting the body, resolve `a.radd(b)` to the **specific `impl R for i64`'s `radd`** (found via prerequisite #2's method-registration table — confirm what that table's lookup key is: presumably `(type_hash, method_name_hash)`; #3 should look it up the same way an inherent-method call would, since #2 registered trait-impl methods indistinguishably from inherent ones) and emit a direct call, not any kind of runtime vtable/dispatch (no dynamic dispatch is needed — this is monomorphization, everything is known at compile time per instantiation).
- If #2 registered trait-impl methods in the exact same table as inherent methods (as instructed in #2's task, §3), then in the *already-monomorphized* body, `a.radd(b)` where `a`'s concrete type is now `i64` should resolve through the **ordinary method-call path** with no new dispatch logic needed at all — the only genuinely new work may be (a) accepting `<F: Trait>` bound syntax in the parser/checker without producing a type error on the as-yet-unresolved `a.radd(b)` call *before* monomorphization substitutes the concrete type, and (b) making sure monomorphization actually substitutes `F -> i64` early enough that the ordinary method-resolution pass sees `a: i64`, not `a: F`, by the time it looks up `radd`. Verify empirically rather than assuming which of these is the actual blocker — instrument or bisect from the spike's new error message once #2 is merged.

Engine 2 — Madaros: `self-hosted/compiler/module_frontend.sio`, same shape: bound parsing + call resolution through instantiation. Confirm current failure mode there separately (may differ from lean_single's).

## 4. Acceptance criteria (ALL required; both engines / default lane)

1. `docs/handoff/spike_trait_bounded_dispatch.sio` (promote a copy into `tests/run-pass/trait_bounded_dispatch.sio`) → runs, prints `5` then `spike PASS`.
2. Add `tests/run-pass/trait_bounded_dispatch_struct.sio`: same shape but `F` bound to a **struct** type (e.g. a 2-field `{num: i64, den: i64}` implementing `R`), to prove dispatch isn't builtin-type-specific — mirrors the real consumer's `Rational` case.
3. Add `tests/run-pass/trait_bounded_dispatch_multi_call.sio`: the bounded function calls **two different** trait methods on `F` (e.g. `radd` and a second method `rsub`), and is instantiated at **two different** concrete types in the same program (`use2::<i64>(...)` and `use2::<StructF>(...)`), to catch any single-instantiation-only bug in the monomorphizer.
4. Self-host fixed point preserved: canonical compiler gate green; gen2 == gen3 (bit-identical). Use the **canonical** gate — per the 2026-07-05 handoff, `lean_single_fixed_point_gate.sh` has a pre-existing harness break (targets the `bin/souc` wrapper).
5. No regression: full `tests/run-pass` fail-count not worse than baseline; existing unbounded generic-fn fixtures (anything using bare `<T>` with no `: Trait` bound, e.g. `closure_generic_hof`, `turbofish.sio` once #1 lands) still pass — bound parsing must be additive, not a breaking change to the existing bare-`<T>` grammar.
6. Madaros: equivalent acceptance; **output-verified** (assert the printed values, not just rc — the compact stub backend false-greens on exit code).
7. **End-to-end smoke test (stretch, encouraged but not blocking):** if #1 has also landed by this point, attempt compiling a minimal 2-field `CDElementExact<F>`-shaped struct with a trait-bounded `cd_add_exact<F: ExactRing>(a: CDElementExact<F>, b: CDElementExact<F>) -> CDElementExact<F>` calling `.er_add()` element-wise — this is the actual shape of the real consumer and is the strongest possible signal that #1+#2+#3 compose correctly. If it doesn't fit in scope/time, document the attempt and hand off to `coord/exact-algebra-core` rather than blocking the release of #3 alone.

## 5. Protocol / coordination (non-negotiable in this repo)

- Fresh worktree off `origin/main` (which must already contain #2, and ideally #1) — append a **CLAIM** to `artifacts/omega/agent_handoff.log.md` before editing, and in the CLAIM note explicitly which of #1/#2 are present in the base you branched from (commit SHA).
- **Serialized surfaces:** `self-hosted/compiler/lean_single.sio` and `bin/souc-linux-x86_64` — coordinate with **Lane 4 (nv2-compiler-hardening)**; hold the `bin/souc` token per the 6-lane doc. `module_frontend.sio` for the Madaros half.
- Heed the 2026-07-05 entries: two-level indexed RMW is **safe again** on trees containing `06409ecb9` (resynced seed); the `&!`-of-boxed-element class is still a separate live defect. Rebuild to fixed point; **output-verify** every witness.
- On green: **RELEASE** with `commit=<sha>` + checks, and ping `coord/exact-algebra-core` (this lane) via the log — with #1+#2+#3 all green, the generic-in-`F` engine (`stdlib/algebra/cayley_dickson_exact.sio`) can finally be attempted; note in the RELEASE entry that this closes out the full three-feature prerequisite set from `exact_engine_prereqs.md`.

## 6. Minimal repro source (spike)

Committed as `docs/handoff/spike_trait_bounded_dispatch.sio`:

```sounio
//@ run-pass
trait R {
    fn radd(self, o: Self) -> Self
}

impl R for i64 {
    fn radd(self, o: Self) -> Self { self + o }
}

fn use2<F: R>(a: F, b: F) -> F {
    a.radd(b)
}

fn main() with IO {
    let r = use2::<i64>(2, 3)
    println(r)
    println("spike PASS")
}
```

## 7. Out of scope

- Do **not** touch the exact-algebra consumer files (`stdlib/algebra/cayley_dickson_exact.sio`, `stdlib/algebra/cayley_dickson_exact_i64.sio`, `stdlib/math/sedenion_verdict.sio`, `tests/run-pass/sedenion_zd_*`) — owned by `coord/exact-algebra-core`. The §4.7 smoke test is a throwaway spike file, not a change to those owned files.
- Do **not** touch Lane 3 paper-168 files (`examples/cocycle_*`, `examples/*168*`, `docs/papers/main/168-*`).
- Do **not** re-implement or second-guess prerequisite **#1** (generic-struct-return) or **#2** (`impl Trait for Type` parsing) — depend on them as merged, do not route around a missing one by inventing a parallel mechanism.
- No dynamic/runtime trait-object dispatch (`dyn Trait`, vtables, trait objects as a first-class value) — everything here is static monomorphization, resolved entirely at compile time per instantiation. If a spike seems to need runtime dispatch, that means the spike is wrong, not that this task's scope should grow.
- No trait inheritance, default trait-method bodies, negative bounds, or multi-trait bounds (`<F: A + B>`) — single-bound, methods-only traits only, matching `ExactRing`'s actual shape.
