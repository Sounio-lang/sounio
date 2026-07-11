<!-- docs:meta
topic_id: repo.docs.handoff.compiler-generic-struct-return-fix-prompt
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.compiler-generic-struct-return-fix-prompt
-->

# Prompt — Fix souc generic-struct-return monomorphisation (unblocks exact-algebra `<F>`)

**For:** fable5 agent on the **Claude #2 lane**
**Authored by:** Claude (exact-algebra-core lane, `coord/exact-algebra-core`), 2026-07-05
**Priority:** blocking prerequisite for the operator-initiated Exact Algebraic Core work
**Type:** compiler-internals (self-host + Madaros) — serialized surfaces, fixed-point + output-verified gates mandatory

---

## 0. One-line task

Make the compiler accept a **function that returns a generic struct parameterized by the function's own type parameter** — e.g. `fn make<F>(x: F) -> Wrapper<F>` and `fn cd_mul_exact<F>(a: CDElementExact<F>, b: CDElementExact<F>) -> CDElementExact<F>`. Today both compiler engines reject this. It is a documented completeness gap on **valid** code.

## 1. Why this matters (the consumer)

The operator asked for an **exact algebraic layer** beneath the f64 Cayley–Dickson/sedenion runtime: zero-divisor annihilation as decidable integer equality (`ab == 0` over ℤ) instead of tolerance-gated float. The design (operator-approved) is **generic in the coefficient field `F`**:

```
struct CDElementExact<F> { c: [F; 2048], bits: i64 }
fn cd_mul_exact<F>(a: CDElementExact<F>, b: CDElementExact<F>) -> CDElementExact<F> with ...
fn cd_associator_exact<F>(a, b, c: CDElementExact<F>) -> CDElementExact<F>
```

with `F = i64` first and `F = Rational` (stdlib/math/rational.sio) second. A capability spike (below) proved every one of these functions hits the generic-struct-return gap. **Until this is fixed, the exact-algebra lane cannot use `<F>` and must fall back to a hand-monomorphized concrete i64 engine.** Fixing this unblocks the clean generic design.

## 2. The bug — exact reproduction

Both reproductions are on `souc v0.80.0`, worktree off `origin/main`. Both exit **rc=0** — that is the known false-green pattern (compact stub backend); the real signal is `Compilation failed!` / `type checking preflight failed` on **stdout**. Do not trust exit codes; grep stdout.

**Repro A — existing known-failure fixture** `tests/run-pass/turbofish.sio`:
```
fn make_wrapper<T>(v: T) -> Wrapper<T> { Wrapper { val: v } }   // struct Wrapper<T> { val: T }
...
let w = make_wrapper::<i64>(99)   // rejected
```
Its own annotation: *"generic-struct return monomorphisation gap — `make_wrapper<T> -> Wrapper<T>` is rejected by both engines (lean_single 'tail type mismatch'; Madaros E004/E008/E009). Completeness gap, valid code; needs compiler generic-struct-return support."*
Run: `./bin/souc run tests/run-pass/turbofish.sio` → stdout ends `type checking preflight failed`.

**Repro B — minimal, mirrors the real consumer** (source committed next to this doc as
`docs/handoff/spike_generic_struct_return.sio`; also inline in §6):
`fn cd_add<F>(a: CDExact<F>) -> CDExact<F>` with a by-value `c: [F; 4]` field, called `cd_add::<i64>(...)`.
Run → stdout:
```
: argument type does not match parameter
   = expected CDExact
   = found CDExact__T
```
**The tell:** the monomorphizer mangles the instantiation to `CDExact__T` — it leaked the **type-parameter name `T`** into the mangled struct name instead of substituting the call-site concrete type (should be `CDExact_i64`). So the argument's type never equals the declared parameter/return type, and lean_single raises **"tail type mismatch"** on the returned expression.

**Control (works today):** non-generic instantiation and field access are fine —
`tests/run-pass/generic_struct_instantiate.sio` and `generic_struct_nested.sio` pass. The gap is *specifically* generic structs flowing through **function parameter/return positions** where the type arg is another function's type param.

## 3. Root-cause map (starting points — verify, don't assume)

Engine 1 — `self-hosted/compiler/lean_single.sio` (35,897 lines):
- `mono_mangle(ns,ne,ts,te)` @ **5740** — builds the mangled name `"name_type"`. Confirm what `ts..te` (the type span) is when the "type" is itself an unresolved type-param `F`/`T`: that is almost certainly where `__T` originates.
- Monomorphisation instance registry + dispatch: `MONO_GEN / MONO_TY_S / MONO_TY_E / MONO_IS_ST`, `mono_find_inst`; the generic-fn and generic-struct discovery loops @ **~25150–25325**; **Pass 0c+0d monomorphization** @ **26178**; "Re-scan monomorphized struct declarations" @ **25484–25489**.
- `"tail type mismatch"` emitter @ **27281** (the return-expr vs declared-return check that fires here).
- Generic registries: search from the `// Generic function registry (for monomorphization)` comment @ **187**.

Engine 2 — Madaros: `self-hosted/compiler/module_frontend.sio` (+ the two-level RMW hazards noted in the 2026-07-05 handoff entries), error codes **E004/E008/E009**. Diagnostic strings around `self-hosted/interop/contract.sio:285` (`DIAG_E008_INVALID_TYPE`).

**Working hypothesis:** when substituting a function's mono type-arg into the function's signature, the substitution is not applied to **nested generic-struct type references** in parameter/return positions (`CDExact<F>`), so `F` stays symbolic, `mono_mangle` emits `..__T`, no corresponding struct instantiation is registered, and the arg/return types compare unequal. The fix must (a) resolve the fn's type-param to the concrete call-site type inside nested generic-struct type refs, (b) register/emit that struct instantiation (reuse the struct-mono path @ 25172+/25484), and (c) make the return-expr type unify with the declared return type so 27281 no longer fires. Apply the mirror fix in Madaros.

## 4. Acceptance criteria (ALL required; both engines / default lane)

1. `tests/run-pass/turbofish.sio` → **run-pass**: remove the `known-failure` annotation, all three asserts print `PASS`.
2. `tests/run-pass/generic_struct_return.sio` (promote the spike, §6) → runs, prints `6` then `spike PASS`.
3. Struct-`F` case: a `CDExact` instantiated at a **2-field struct `F`** (Rational-like `{num:i64,den:i64}`), a fn returning it, by-value `[F; N]` field → compiles **and** runs correctly. Add as `tests/run-pass/generic_struct_return_structf.sio`.
4. Self-host fixed point preserved: canonical compiler gate green; gen2 == gen3 (bit-identical). Use the **canonical** gate — per the 2026-07-05 handoff, `lean_single_fixed_point_gate.sh` has a pre-existing harness break (targets the `bin/souc` wrapper).
5. No regression: `generic_struct_basic/nested/instantiate/knowledge`, `closure_generic_hof` still pass; full `tests/run-pass` fail-count not worse than baseline.
6. Madaros: E004/E008/E009 no longer emitted for these shapes; **output-verified** (assert the printed values, not just rc — the compact stub backend false-greens on exit code).

## 5. Protocol / coordination (non-negotiable in this repo)

- Fresh worktree off `origin/main`; append a **CLAIM** to `artifacts/omega/agent_handoff.log.md` before editing.
- **Serialized surfaces:** `self-hosted/compiler/lean_single.sio` and `bin/souc-linux-x86_64` — coordinate with **Lane 4 (nv2-compiler-hardening)**; hold the `bin/souc` token per the 6-lane doc. `module_frontend.sio` for the Madaros half.
- Heed the 2026-07-05 entries: two-level indexed RMW is **safe again** on trees containing `06409ecb9` (resynced seed); the `&!`-of-boxed-element class is still a separate live defect. Rebuild to fixed point; **output-verify** every witness.
- On green: **RELEASE** with `commit=<sha>` + checks, and ping `coord/exact-algebra-core` (this lane) via the log so the exact-algebra work resumes on `<F>`.

## 6. Minimal repro source (spike)

Committed as `docs/handoff/spike_generic_struct_return.sio`:

```sounio
//@ run-pass
// SPIKE: fn returning a generic struct parameterized by F, with a by-value [F;N] field.
// This is exactly what cd_mul_exact<F> -> CDElementExact<F> needs.
struct CDExact<F> { c: [F; 4], bits: i64 }
fn cd_add<F>(a: CDExact<F>, b: CDExact<F>) -> CDExact<F> {
    var out = CDExact { c: a.c, bits: a.bits }
    out
}
fn main() with IO {
    let a = CDExact { c: [1, 2, 3, 4], bits: 4 }
    let b = CDExact { c: [5, 6, 7, 8], bits: 4 }
    let r = cd_add::<i64>(a, b)
    println(r.c[0])       // expect 6 after you implement real add; 1 with the copy stub above
    println("spike PASS")
}
```

## 7. Out of scope

- Do **not** touch the exact-algebra consumer files (`stdlib/algebra/cayley_dickson_exact.sio`, `stdlib/math/sedenion_verdict.sio`, `tests/run-pass/sedenion_zd_*`) — owned by `coord/exact-algebra-core`.
- Do **not** touch Lane 3 paper-168 files (`examples/cocycle_*`, `examples/*168*`, `docs/papers/main/168-*`).
- No trait system / higher-kinded types / generic-method monomorphisation beyond what criteria 1–3 require. Smallest change that makes generic-struct **return + param** positions monomorphize correctly.
