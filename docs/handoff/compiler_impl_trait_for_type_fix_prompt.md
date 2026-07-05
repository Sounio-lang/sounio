<!-- docs:meta
topic_id: repo.docs.handoff.compiler-impl-trait-for-type-fix-prompt
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.compiler-impl-trait-for-type-fix-prompt
-->

# Prompt — Parse `impl Trait for Type` (unblocks exact-algebra `<F: Trait>`, prerequisite #2 of 3)

**For:** a fresh compiler-lane agent (fable-style), analogous to the fable5 agent on prerequisite #1
**Authored by:** Claude (exact-algebra-core lane, `coord/exact-algebra-core`), 2026-07-05
**Priority:** blocking prerequisite for the generic-in-`F` Exact Algebraic Core, **secondary to** the concrete-i64 engine which ships the same science without it
**Type:** compiler-internals (self-host + Madaros) — serialized surfaces, fixed-point + output-verified gates mandatory

---

## 0. One-line task

Make the compiler **parse** `impl <TraitPath> for <Type> { <methods> }` — a trait implementation block — and attach its methods to `<Type>` as callable methods (`value.method(...)`), the same way inherent `impl Type { ... }` methods already work. This is prerequisite **#2 of 3** in `docs/handoff/exact_engine_prereqs.md`; it depends on nothing else, but prerequisite **#3 (trait-bounded generic dispatch)** depends on this one.

**Scope correction from the original survey (verified 2026-07-05, see §2): this task is actually TWO coupled parser bugs, not one.** A trait declaration whose methods have **no body** (bodyless signatures — the only useful form for an interface like `ExactRing`, and the form used everywhere in the stdlib) fails to parse *on its own*, with no `impl` in sight. `impl Trait for Type` is a second, independent bug on top of that. `ExactRing` needs bodyless methods, so **both** must be fixed for this task's own acceptance spike to compile — see §3 (Bug A / Bug B) and §4.

## 1. Why this matters (the consumer)

The operator asked for an **exact algebraic layer** beneath the f64 Cayley–Dickson/sedenion runtime: zero-divisor annihilation as decidable integer equality (`ab == 0` over ℤ) instead of tolerance-gated float. The clean design (operator-approved) is **generic in the coefficient field `F`**, via a trait:

```
trait ExactRing {
    fn er_add(self, o: Self) -> Self
    fn er_sub(self, o: Self) -> Self
    fn er_mul(self, o: Self) -> Self
    fn er_is_zero(self) -> bool
    fn er_eq(self, o: Self) -> bool
}
impl ExactRing for i64 { ... }
impl ExactRing for Rational { ... }
```

A **trait declaration with zero methods** parses today (`trait Empty {}`), but `trait ExactRing { ... }` as actually written — five bodyless method signatures, the same style as `stdlib/quantum/vqe.sio`'s `trait Ansatz`/`trait FermionMapping` — does **not** parse in isolation (Bug A, §2). Layered on top, the **implementation** half, `impl ExactRing for i64 { ... }`, is rejected independently at the `for` keyword (Bug B, §2). Without both fixes, `i64` and `Rational` cannot be given `ExactRing` methods, so the generic-in-`F` engine (`stdlib/algebra/cayley_dickson_exact.sio`, currently a DESIGN SKELETON that does not compile) is stuck. **Until this is fixed, the exact-algebra lane cannot use `<F: ExactRing>` and continues on a hand-monomorphized concrete-i64 engine.** Fixing this (plus #3) unblocks the clean generic design as a code-sharing win — it does not change any scientific result (see `exact_engine_prereqs.md`).

## 2. The bug — exact reproduction (TWO bugs, verified independently)

Reproduction is on `souc v0.80.0`, worktree off `origin/main`. All repros exit **rc=1** (the parser stage fails outright, before the compact-stub-backend false-green would even apply) — but treat rc as untrustworthy in general on this compiler and always read stdout; other stages of this codebase are known to false-green at rc=0.

### Bug A — bodyless trait-method signatures don't parse (no `impl` involved at all)

```
trait Tr { fn g(self) -> i64 }     // note: NO body on `g`
fn main() with IO { println("t") }
```
→ stdout: `parse error: expected token, expected=184 actual=177` at the trait's own closing `}` (the unexpected token), plus a cascaded EOF error. **Control:** `trait Empty { }` (zero methods) parses fine (`Compilation successful!`) — so the bug is specifically the **method-signature-with-no-body** grammar inside a trait, not traits in general. This directly contradicts `exact_engine_prereqs.md`'s claim "trait DECLARATION ✅ parses (`stdlib/quantum/vqe.sio` has `trait Ansatz`)" — verified by extracting `trait Ansatz { ... }` verbatim (4 bodyless methods) into a standalone file: it does **not** parse (`./bin/souc run` on the extracted snippet → the same class of error). `ExactRing` (§1) is written the same way — 5 bodyless methods — so Bug A blocks it directly, independent of Bug B below.

### Bug B — `impl Trait for Type` rejected at the `for` keyword

Isolate this from Bug A by giving the trait method a throwaway default body:
```
trait Tr { fn g(self) -> i64 { 0 } }
struct Q { v: i64 }
impl Tr for Q { fn g(self) -> i64 { self.v } }
```
→ stdout: `parse error: expected token, expected=184 actual=23` — right at the `for` keyword, a single, clean error (no cascade). The main parser's `impl` grammar only accepts `impl <Ident> {` (and `impl<G> <Ident><G> {`); it expects `{` immediately after the first identifier and rejects `for`.

### Combined repro (the real shape of the consumer)

Committed as `docs/handoff/spike_impl_trait_for_type.sio` (bodyless method, matching `ExactRing`'s actual style, so both bugs fire together — also inline in §6):
```
trait Tr { fn g(self) -> i64 }
struct Q { v: i64 }
impl Tr for Q { fn g(self) -> i64 { self.v } }
```
Run: `./bin/souc run docs/handoff/spike_impl_trait_for_type.sio` → stdout (line numbers reflect the committed file's header comment; re-run to reconfirm if you edit the file):
```
parse error: expected token at line 47:1     (Bug A: trait's own closing brace)
 expected=184
 actual=177
parse error: expected token at line 51:9     (Bug B: the `for` keyword)
 expected=184
 actual=23
parse error: expected token at line 60:1     (cascaded EOF)
 expected=185
 actual=0
Parse failed for module 0: 6 errors
```

**Important correction to an earlier survey** (`exact_engine_prereqs.md` said "zero occurrences [of `impl Trait for Type`] in the entire stdlib" — not quite accurate): the *syntax* `impl X for Y` does appear several places — `stdlib/quantum/vqe.sio` (`impl FermionMapping for JordanWigner`, `impl Ansatz for HardwareEfficient`, etc.), `stdlib/ml/pac.sio` (`impl Learnable for DecisionStump`), and `examples/*.sio` (`impl Default for ...Config`). **None of these files currently parse** — verified: `./bin/souc run stdlib/quantum/vqe.sio` and `./bin/souc run stdlib/ml/pac.sio` both fail (among other, unrelated errors in those files — `vqe.sio` also has an unrelated char-literal-match issue at lines 117–120). So the syntax is aspirational/forward-declared in several stdlib files awaiting exactly this fix — treat those files as **bonus regression fixtures**, not just spikes, once the fix lands (see §4.3). Note also that in the full `vqe.sio` file, the parser's error-recovery masks some individual trait/impl sites (e.g. `trait Ansatz` itself doesn't show up as a separately-reported error line in the full-file run, even though the identical snippet fails in isolation) — do not use "no error reported at line N in the full file" as evidence that construct N parses; always isolate to a minimal standalone file to get a trustworthy signal.

**Also note:** there is already dead-reckoning code in `self-hosted/compiler/lean_single.sio` around **line 25988** (`// Check for "for" keyword — if present, this is impl Trait for Type`) inside the **Pass 0a receiver-hash scan**. This code runs on the raw token stream *after* the main parser has already produced an AST (or failed to) — it does not help here because the **primary parser** (the stage that emits the errors above) rejects the construct before Pass 0a ever runs. Do not confuse the two: Pass 0a's `for`-detection is a real, working piece of machinery for something else (receiver-hash tracking) — the fix in this task is in the earlier, primary grammar stage, for BOTH Bug A and Bug B.

## 3. Root-cause map (starting points — verify, don't assume)

Engine 1 — `self-hosted/compiler/lean_single.sio` (35,897 lines):

**Bug A (bodyless trait-method signatures):** search for the trait-item parser (item-level `trait` keyword handler, distinct from `impl`). It currently appears to parse a trait body as a sequence of items each expected to have a `{ ... }` body (like a normal top-level `fn`), which is why `trait Empty {}` (no methods) parses but `trait Tr { fn g(self) -> i64 }` (one bodyless method) doesn't — the parser is presumably falling into ordinary-fn parsing for `fn g(self) -> i64`, expects a body next, and instead finds the trait's closing `}`. Fix: inside a trait body, after parsing a method's name/params/return-type, accept **either** a `{ ... }` body (default-method — nice-to-have, not required for this task) **or** a bare newline/`}`-terminated signature (the required, minimal case). Verify with the isolated Bug A repro (§2) before moving to Bug B.

**Bug B (`impl Trait for Type`):** the primary item/impl parser — search for the `impl` keyword-dispatch in the top-level statement/item parser (NOT the Pass 0a scan at line 25988, which is a later, separate token-stream sweep). It currently recognizes `impl` → optional `<G>` generic params → type-name identifier → expects `{` directly. Add: after the type-name identifier, check for a `for` keyword; if present, treat the identifier just consumed as the **trait path** and parse a second type-name identifier (optionally with its own `<G>` generic args) as the **implementing type**, then expect `{`.
- AST representation: `self-hosted/parser/ast.sio:1152` already has a comment `// For "impl TraitName for Type { }" — trait_name.len > 0 when present` — there is very likely an existing field on the impl-block AST node reserved for exactly this (`trait_name` or similar). Find it and confirm whether it is populated anywhere today (likely: nowhere, since nothing produces it). Populate it from the new grammar branch.
- Method attachment: once parsed, the block's methods must be registered against the **implementing type** (not the trait) using whatever registry inherent `impl Type { ... }` methods use today, so `q.g()` dispatches exactly like an inherent method. For prerequisite #2 alone, it is acceptable (and simplest) to register the methods as if they were inherent methods on the implementing type, ignoring the trait identity — trait-bound *checking* (does this impl satisfy trait X's signature) and trait-bound *dispatch* (`fn f<F: Trait>`) are prerequisite **#3**, out of scope here. Do not build #3's machinery in this task; just make the block parse and the methods callable.
- Also handle `impl<G> Trait for Type<G> { ... }` (generic-parameterized) if it falls out cheaply from reusing the existing `<G>`-skip logic already present for inherent impls — but this is not required for acceptance (see §4); do not scope-creep into it if it's nontrivial.

Engine 2 — Madaros: `self-hosted/compiler/module_frontend.sio` — find the mirror of the primary impl-item parser there and apply the equivalent grammar extension. Confirm current failure mode (parse error / E00x) before changing anything.

**Working hypothesis:** this is a narrowly-scoped grammar addition (recognize an optional `<TraitPath> for` prefix before the impl type name) plus wiring the already-reserved AST field, not a new subsystem. The bulk of the risk is in *method registration* — making sure `q.g()` resolves through whatever table inherent impls populate, so no second, parallel "trait-impl method" lookup path needs to be invented.

## 4. Acceptance criteria (ALL required; both engines / default lane)

1. `docs/handoff/spike_impl_trait_for_type.sio` (promote a copy into `tests/run-pass/impl_trait_for_type.sio`) → runs, prints `3` then `spike PASS`. This requires **both** Bug A and Bug B fixed (the spike uses a bodyless trait method, per §2).
2. A second run-pass fixture, `tests/run-pass/impl_trait_for_type_multi.sio`, covering: (a) a type with **two** trait impls (`impl TraitA for T { ... } impl TraitB for T { ... }`, both callable), and (b) a trait impl **alongside** an inherent `impl T { ... }` block on the same type (both sets of methods callable, no collision).
3. A standalone regression fixture for Bug A alone, `tests/run-pass/trait_decl_bodyless_methods.sio`: a trait with **two or more** bodyless methods (no impl needed) parses and an empty-trait control (`trait Empty {}`) still parses too — do not regress the zero-method case while fixing the nonzero-method case.
4. **Bonus regression check (not required to fully compile end-to-end, but must not regress further):** full-file error recovery on this compiler is not trustworthy as a pass/fail signal (verified: `trait Ansatz` produces no reported error in the full `vqe.sio` run, yet the identical snippet fails when isolated — see §2). So do **not** rely on `./bin/souc run stdlib/quantum/vqe.sio`'s line-numbered error list; instead, extract each real trait/impl block verbatim into its own standalone file (`trait Ansatz`@467, `trait FermionMapping`@261, `impl FermionMapping for JordanWigner`@270, `impl FermionMapping for BravyiKitaev`@327, `impl Ansatz for HardwareEfficient`@506, `impl Ansatz for UCCSD`@615 in vqe.sio; `trait Learnable`@86, `impl Learnable for DecisionStump`@102 in pac.sio) and confirm each isolated extraction now parses (may still fail to fully *compile*/typecheck for unrelated reasons — that's fine; just confirm no Bug-A/Bug-B-class parse error remains). Do not attempt to fix unrelated errors in those files or in the full multi-thousand-line originals.
5. Self-host fixed point preserved: canonical compiler gate green; gen2 == gen3 (bit-identical). Use the **canonical** gate — per the 2026-07-05 handoff, `lean_single_fixed_point_gate.sh` has a pre-existing harness break (targets the `bin/souc` wrapper).
6. No regression: full `tests/run-pass` fail-count not worse than baseline; existing inherent-impl fixtures (`generic_struct_basic/nested/instantiate`, anything exercising `impl Type { ... }`) still pass.
7. Madaros: equivalent parse acceptance; **output-verified** (assert the printed values, not just rc — the compact stub backend false-greens on exit code).
8. Do **not** implement trait-bound checking, trait-bound generic dispatch, or any monomorphization — that is prerequisite #3, a separate task. This task's methods may be registered and dispatched exactly like inherent-impl methods; nothing here needs to know a "trait" exists at type-check/codegen time.

## 5. Protocol / coordination (non-negotiable in this repo)

- Fresh worktree off `origin/main`; append a **CLAIM** to `artifacts/omega/agent_handoff.log.md` before editing.
- **Serialized surfaces:** `self-hosted/compiler/lean_single.sio` and `bin/souc-linux-x86_64` — coordinate with **Lane 4 (nv2-compiler-hardening)** and with whoever is landing prerequisite **#1** (`docs/handoff/compiler_generic_struct_return_fix_prompt.md`) — both tasks touch the same monomorphization/parser neighborhood in the same file; check the handoff log for an active claim before starting, and hold the `bin/souc` token per the 6-lane doc. `module_frontend.sio` for the Madaros half.
- Heed the 2026-07-05 entries: two-level indexed RMW is **safe again** on trees containing `06409ecb9` (resynced seed); the `&!`-of-boxed-element class is still a separate live defect. Rebuild to fixed point; **output-verify** every witness.
- On green: **RELEASE** with `commit=<sha>` + checks, and ping `coord/exact-algebra-core` (this lane) via the log — prerequisite **#3 (trait-bounded dispatch)** is queued to start immediately after, and needs this one merged first.

## 6. Minimal repro source (spike)

Committed as `docs/handoff/spike_impl_trait_for_type.sio`:

```sounio
//@ run-pass
trait Tr {
    fn g(self) -> i64
}

struct Q { v: i64 }

impl Tr for Q {
    fn g(self) -> i64 { self.v }
}

fn main() with IO {
    let q = Q { v: 3 }
    println(q.g())
    println("spike PASS")
}
```

## 7. Out of scope

- Do **not** touch the exact-algebra consumer files (`stdlib/algebra/cayley_dickson_exact.sio`, `stdlib/algebra/cayley_dickson_exact_i64.sio`, `stdlib/math/sedenion_verdict.sio`, `tests/run-pass/sedenion_zd_*`) — owned by `coord/exact-algebra-core`.
- Do **not** touch Lane 3 paper-168 files (`examples/cocycle_*`, `examples/*168*`, `docs/papers/main/168-*`).
- Do **not** implement prerequisite **#1** (generic-struct-return) or **#3** (trait-bounded generic dispatch) — separate tasks, separate prompts (`docs/handoff/compiler_generic_struct_return_fix_prompt.md`, `docs/handoff/compiler_trait_bounded_dispatch_fix_prompt.md`).
- Do not attempt to fully fix `stdlib/quantum/vqe.sio` or `stdlib/ml/pac.sio` end-to-end — only confirm the specific Bug-A/Bug-B error classes disappear from them (§4.4).
- No associated-type support (`type Model;`, seen in `pac.sio`'s `Learnable` trait) — that is a distinct, unrelated feature; not needed by `ExactRing` and not in scope here.
