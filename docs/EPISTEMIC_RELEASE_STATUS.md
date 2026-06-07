# Sounio / Madáres v0.80.0 — Epistemic Release Status

> An epistemic programming language must know what it knows.
>
> Every capability below carries a **confidence** and a **provenance** — the exact
> gate, repro, or observed exit code that backs it. We do not assert above our
> evidence. Where the compiler is wrong, we say so, loudly, with a reproduction.
> This document is itself written in the discipline the language espouses: it is a
> set of `Knowledge<capability>` values — `{ value, ε, provenance }` — not a
> marketing claim.
>
> Honesty note, stated first because it matters most: the language *proposes*
> compile-time confidence gates (`ε ≥ …`). The current compiler **does not yet
> enforce** them (see Unverified, below). So this release earns its honesty by
> *practice* — adversarial verification — not by a feature it does not yet have.

Reference tip: `integration/native-v2-honest` @ `a53511813`.
Method of provenance: every "verified" claim was checked by building the modular
compiler `mc` from `self-hosted/compiler/main.sio` via the bootstrap `./bin/souc`,
then running the program and observing the exit code / output — never trusted from
a report. The aggregate honest gate is `scripts/ci/release_gate.sh` (11 gates).

---

## What the compiler builds, certified

`scripts/ci/release_gate.sh` — **11/11 honest gates green** at this tip. These are
the only gates that build `mc` from `main.sio` and assert *observed* behavior:

| gate | result | what it proves |
|---|---|---|
| native_v2_calls_arity | PASS | 5- and 6-arg calls (r8/r9 ABI) |
| native_v2_e2e_codegen_suite | PASS | scalar/call/fnptr/control/arith/f64/sret IR→ELF→exit |
| native_v2_e2e_exit_code | PASS | emit→exit-code contract |
| native_v2_recovered_source | PASS | struct/array/enum/logical/nested/method **source→ELF** |
| capgate | 32/32 | single-file source → native ELF across core shapes |
| native_v2_soundness | 7/7 | ill-typed single-module source is **rejected**, not miscompiled |
| native_v2_enum | 15/15 | enum struct-variant construction/dispatch/binding |
| native_v2_checker_crash | 26/26 | the checker no longer crashes on these shapes |
| native_v2_literal_coercion | 19/19 | numeric-literal expected-type coercion, **only literals** |
| native_v2_backend_soundness | 40/40 (+1 tracked) | field/discriminant lowering correctness |
| native_v2_multimodule | 27/27 | multi-module link + ill-typed cross-module rejection |
| parser sweep | 525/525 | the run-pass corpus still parses |

Two **tracked, by-design residuals** are listed in the gate output and never folded
into a green count (per our anti-self-deception rule): the `aa/ae` field-hash bucket
collision, and (historically) the import-typecheck slips — both pinned by identity.

---

## VERIFIED — high confidence (compiles AND runs to the correct value)

29 language features, each re-run by hand against `mc`:

literals i64 · explicit `as` casts i32↔i64 · f64 literals & comparison · arithmetic
with precedence · `==` `!=` `&&` · `if`/`else` · `while` · `for`‑in ranges ·
`let`/`var`/`mut` · struct field get/set · nested struct access · array index/set ·
unit enums + `match` · **enum struct-variants + field-bound `match`** · match guards ·
function calls (1–6 args) · recursion · function pointers · higher-order
function-typed params · methods (`self` and `&self` receivers) · tuple return &
projection · linear types (use-exactly-once, double-use **rejected**) · `with Mut`.

**Numeric-literal coercion** (new this release): `fn main() -> i32 { 0 }` now
type-checks and runs — a bare numeric literal adopts the expected type (i32/i64/
f32/f64) from context. Soundness is preserved by construction: coercion is gated on
the literal **AST node**, never on the expected type, so a non-literal of the wrong
width still rejects (`let y:i64=5; let x:i32=y` → error), out-of-range narrow
literals reject (`let x:u8=300` → error), and literals to non-numeric types reject
(`let b:bool=0` → error). This made **+223** previously-rejected files type-check.

**Multi-module** programs link and run for the tested layout (`use` + an imported
`pub fn`/`struct`/`enum`), and ill-typed cross-module use (wrong-typed argument to
an imported function) is **rejected** with the right type error.

---

## PARTIAL — medium confidence (works in the verified shape; edges unproven)

- **Multi-module native compilation is structure-dependent.** The canonical
  `use helper::util::*` layout links and runs (verified, exit-checked). Some other
  module layouts fail in the native back half (`ir_summary_failed`). Treat
  multi-module native output as working for simple shapes, not yet hardened.
- **Cross-module type checking** rejects wrong-typed arguments to imported
  functions; a let-binding from an imported call result (`let x:bool = i64_fn()`)
  may not yet be caught in every shape — under verification.

---

## KNOWN-WRONG — we know the compiler is wrong here (reproductions included)

These are stated loudest because, for an epistemic language, a *silent* wrong
answer is the worst possible failure. None of these should be relied on.

**Silent miscompiles (well-typed source → wrong runtime value):**
- `f32` comparison: `let x:f32=1.5 as f32; if x > 1.0 as f32 {12} else {0}` → exits
  **0**, should be 12. (`f64` comparison is correct; `f32` is not.)
- Closures without capture: `let f = |x:i64| x+1; f(41)` → exits **1**, should be 42.

**Crashes during `--native-v2-compile` (after a clean type-check):**
closures with capture · generic functions (`id<i64>(42)`) · string `len`/`eq`/
`concat` · `with IO` / `println`.

**Rejected (feature not yet accepted by the checker):** enum tuple-variant
match-binding · `Option` + `match` · `Box` deref · generic structs (parse).

The field-hash residual: two struct fields sharing a first letter *and* a name-hash
bucket still alias (`struct{aa,ae}`) — rare, tracked in the backend gate, **no worse
than before**, never silently green.

---

## UNVERIFIED / ASPIRATIONAL — claimed by the design, not by this binary

Stated plainly so the design vision is not mistaken for shipped capability:
- **Compile-time ε-confidence gates are NOT enforced.** The `Knowledge(…, ε=…)`
  constructor syntax does not yet type-check on `mc`; the confidence-gate is a design
  goal. (The epistemic *core* — `KCoreKnowledge`, GUM propagation — exists in
  `packages/epistemic-core` as a library, not as an enforced compiler surface.)
- **Effect enforcement is absent on the checked path:** `with Div` works, but the
  same code *without* `with Div` also compiles — declared effects are parsed, not
  enforced.
- **The modular compiler does not yet self-compile** `main.sio` to native ELF. The
  bit-identical bootstrap fixed point belongs to the legacy `lean_single` lane.
- `bin/souc` is the bootstrap (`mini_native`); it uses the positional
  `souc <source> <output>` interface, **not** `check`/`run`/`build` subcommands.

---

## Corpus honesty (where the language stands, by the numbers)

Measured by running `mc --check` over the whole tree at the consolidated tip (not
asserted — counted by me):
- **stdlib + examples combined: 638 / 2003** type-check today (≈32%). The
  numeric-literal coercion in this release is what moved the largest single share
  (+223 files over the prior tip); most remaining failures use not-yet-supported
  features (closures, generics, the unsupported shapes listed above) or are stale
  programs written for a retired dialect.
- For reference, the pre-coercion split was examples 252/860 and stdlib 515/1143;
  100%-green stdlib modules include `constants`, `functional`, `particle_physics`,
  `string`, `research`.

These numbers are deliberately published. A release that hid them would be
unworthy of a language whose entire premise is honest uncertainty.

---

## The release ledger (what closes next, in order)

1. the two silent miscompiles — `f32` comparison, no-capture closures
2. the `--native-v2-compile` crashes — strings, IO, capturing closures, generics
3. packaged-import resolution (`SOUNIO_STDLIB_PATH`) + multi-module hardening
4. effect enforcement — enforce, or document as unenforced in the type system
5. package the real `mc`, not the bootstrap, in the distributable

Each will be landed the way everything above was: a fix, an adversarial audit, and
an independent re-run — and only what survives all three is claimed here.
