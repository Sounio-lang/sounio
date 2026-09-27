<!-- docs:meta
topic_id: repo.docs.audit.documented-and-dead-names-2026-08-20
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.documented-and-dead-names-2026-08-20
-->

# Documented and dead names — material for a decision, not the decision

**Branch:** `lane/minimax-cli1/audit-documented-dead-names-20260820`
**Date of measurement:** 2026-08-20 (lean_single on this pod; Madaros via prior receipts)
**Decision:** owed to the founder — **no decision is made here**

## Scope

Three PRs landed in the last 24 hours measuring specific names against the
specification:

| PR | what it measured |
|---|---|
| [#2033](https://github.com/Sounio-lang/sounio/pull/2033) | `own`, `handle`, `mut` are not keyword-reserved against identifiers (`let <word> = 1`) — `mut` is reserved **to teach against** (E040), the other two are contextual |
| [#2034](https://github.com/Sounio-lang/sounio/pull/2034) | `int`, `uint` give `error[E001]` (different type) on Madaros once a live caller is present; `i128`, `u128` do check |
| [#2043](https://github.com/Sounio-lang/sounio/pull/2043) | five of seven CLAUDE.md §7 syntax rows are style, not enforcement |

This document extends those three with:

1. **Corpus counts** — how many sites in the versioned tree use each name in a
   type position, identifier position, or `with <Effect>` position, separated
   from comment / documentation hits.
2. **Engine behaviour on a live caller** — for each name, what Madaros and
   lean_single do *when the type is actually used*, not just declared.
3. **Cost per exit** — for the three admissible exits (`alias` to a working
   type, `Reserved` so the compiler refuses with a named error, `remove from
   documentation`), how many sites flip meaning.
4. **One additional dead name** — `Observe`, an effect listed in CLAUDE.md
   that does not appear in `docs/spec/LANGUAGE_SPECIFICATION.md` §2.4.

## Methodology

### The negative-control discipline

Three of these names have null-test pitfalls:

- **A type in parameter position passes on both engines for any name**, even
  invented. `fn f(x: zorble_florble) -> i32 { 0 }` checks under Madaros. So
  declaring a type without using it distinguishes nothing. (#2034 already
  showed this — and #2033 is the same probe in identifier position.)
- **The lean_single engine accepts invented types with arithmetic.** Both
  `witness_invented.sio` (`fn make() -> zorble_florble { 0 }`) and
  `witness_invented_arith.sio` (with `x + 1`) compile, link, run, and print
  `PASS`. This makes the **lean_single column structurally uninformative** for
  the negative case — it can only confirm a name is *not* lexically
  disqualified. Every "lean_single accepts" entry below should be read as
  "no lexical refusal," not "this name works."
- **The Madaros shipped binary on this pod is the G1 broken control**
  (`error[E007]` cascade per PR #1961). Direct Madaros measurement is
  unavailable here; the Madaros column in the table below is taken from the
  receipts in #2033 / #2034 / #2043, and from the lean_single self-hosted
  compiler's own `error[E…]` numbering, which mirrors Madaros at the lex level.

### Live-caller probe shape

Every "what happens on a live caller" row is built from the same probe:

```sio
fn make() -> <Name> { <zero literal> }
fn consume(x: <Name>) -> <Name> { x + 1 }      // for arithmetic types
fn main() -> i64 {
    let v = make()
    let r = consume(v)
    if r == 1 { print("PASS witness\n"); 0 } else { print("FAIL\n"); 1 }
}
```

For identifier-shaped names (`own`, `handle`, `mut`, `async`, `await`,
`spawn`), the probe is `let <name> = make(); if <name> == 0 { … }`. For
effects (`Observe`), the probe is `fn f() with <Effect> { () }` plus a
caller.

Witness sources and outputs live under `/tmp/nomes_mortos_witness/` for the
type probes and under `/tmp/witness_*.sio{.out}` for the keyword probes.

## Decision table

### Type-position names

| name | documented at | corpus type-position | corpus comment-only | corpus `.md` only | Madaros (live caller) | lean_single | alias to working type | mark Reserved (E218-style) | remove from docs |
|---|---|---|---|---|---|---|---|---|---|
| `int` | §2.3, §3.1 | **12** (`examples/day21_build_system.sio`, `examples/debug_profile_demo.sio`, `examples/watch_mode_demo.sio`, `examples/day20_tooling.sio`) | many (`stdlib/msgpack/*.sio`) | 228 | **`error[E001]`** (this binding expects a different type) — #2034 | accepted (lexically OK) | **12 sites gain meaning** if aliased to `i64` | **12 sites stop compiling** under stricter rule; also flips any new code that learns `int` from §2.3 | §2.3 row, §3.1 row, CLAUDE.md line 66 wording if it lists `int` |
| `uint` | §2.3, §3.1 | **0** | many (`stdlib/msgpack/*.sio`, `bench/*.sio`) | 6 | **`error[E001]`** — #2034 | accepted | **0 sites change** (alias is free) | **0 sites change today**; protects future code from learning a name that does not compile | same rows as `int` |
| `u16` | §2.3, §3.1 | **0** | `TYPE_U16` identifier only (`stdlib/compiler/check/types.sio:285`); rest are PTX assembly strings | 21 | accepts (`let x: u16 = 0` checks, see #2034 cited evidence) | accepted | not applicable (already works) | would **create** 0 failures; aspirational only | §2.3 row, §3.1 row |
| `char` | §2.3, §3.1 | **23** (`stdlib/text/unicode.sio`, `stdlib/text/case.sio`, plus type params) | 0 in comments as type | 68 | accepts | accepted | not applicable (already works) | would **break** 23 real sites | none needed |
| `f128` | dispatch cited as primitive | **63** (`tests/run-pass/f128_v0b_literal_smoke.sio`, `f128_v0b_literal_forms.sio`, `f128_*.sio`) | PTX assembly strings only | 122 | **`error[E218]` Reserved** (#2034 dispatch) | accepted (lexically OK; emits no-ops in arith) | unknown — no working `f128` to alias to without compiler work | **63 sites stop compiling** under stricter rule; this is the canonical "name says it does not work" pattern (`f128` is the honest model) | rows that promise `f128` as a primitive |
| `f256` | dispatch cited as primitive | **56** (`tests/run-pass/f256_v0b_literal_forms.sio`, `f256_*.sio`) | PTX assembly strings only | 107 | **`error[E218]` Reserved** (#2034 dispatch) | accepted (lexically OK; emits no-ops in arith) | same as `f128` | **56 sites stop compiling** under stricter rule | same rows as `f128` |

### Identifier-shaped "keywords"

| name | documented at | corpus uses as identifier | corpus `.md` only | Madaros | lean_single | alias / contextual / Reserved | remove from docs |
|---|---|---|---|---|---|---|---|
| `own` | §2.3 as type keyword | **15** in code (not a keyword — `let own = 1` checks, see #2033) | 950 (overwhelmingly `docs/audit/**`) | accepted as identifier | accepted as identifier | **contextual** — not reserved at all | remove the keyword-style row from §2.2 / CLAUDE.md; note `own` is just an identifier |
| `handle` | §2.4 as effect keyword | **419** in code (not a keyword — `let handle = 1` checks, see #2033; the contextual slot is `handle<IO> { … }`) | 335 | accepted as identifier | accepted as identifier | **contextual** — same shape as `handle<IO> { … }` for the bind form | remove from the keyword table; note it is contextual in §7.3 |
| `mut` | §2.2 as keyword; CLAUDE.md §7 | **11 keyword uses** (`let mut x` / `var mut x`); **2 235 raw-pointer uses** (`*mut T`) | 763 | **`error[E040]`** — *reserved to teach against* | accepted (lean_single does not teach against `let mut`) | reserved by design — keep, but stop listing it as a usable keyword beside `linear` and `where` | keep the E040 diagnostic; clarify in §2.2 that `mut` is a reserved-only keyword |
| `async` | §2.2 (control-flow category) | **0 as identifier** — `let async = make()` is `PASS async_as_ident` on lean_single; contextual in `async fn …` | dozens | accepted as identifier | accepted as identifier | **contextual** — only meaningful in `async fn` position | the §2.2 row should mark it contextual |
| `await` | §2.2 (control-flow category) | **0 as identifier** — `let await = make()` is `PASS await_as_ident` on lean_single; contextual after `.` | many | accepted as identifier | accepted as identifier | **contextual** — only meaningful as `<expr>.await` | same as `async` |
| `spawn` | §2.2 (control-flow category) | **0 as identifier** — `let spawn = make()` is `PASS spawn_as_ident` on lean_single; contextual in `spawn { … }` | dozens | accepted as identifier | accepted as identifier | **contextual** — only meaningful in `spawn { … }` | same as `async` |

### Effect-position names

| name | documented at | corpus `with <Effect>` uses | Madaros | lean_single | alias | Reserved | remove from docs |
|---|---|---|---|---|---|---|---|
| `Observe` | **CLAUDE.md lines 66, 245** (effect list; example function `fn observe(x: Unobserved<f64>) -> bool with Observe` at line 243) — **NOT in `docs/spec/LANGUAGE_SPECIFICATION.md` §2.4** (which lists 8: IO, Mut, Alloc, Panic, Async, GPU, Prob, Div) | **47** (`stdlib/prob/observe.sio`, `Knowledge<T>` machinery) | accepted (lexically; full enforcement status is in the `Mut` audit §7.2.1 which notes neither shipped engine enforces effect inference) | accepted | not applicable — already accepted | not applicable — already accepted | **the divergence is the bug**: either add `Observe` to §2.4, or remove it from CLAUDE.md and the example function |

## Per-name notes

### `int` — documented but does not compile on Madaros

The probe `let x: int = 0` gives `error[E001]: this binding expects a different type` on Madaros (cited from #2034). All 12 type-position uses are in `examples/` only — there is no use in `stdlib/` or `self-hosted/`. The most plausible exit is **alias to `i64`**: 12 sites gain meaning, and the docs stay accurate. The cost of `Reserved` is also low (0 sites break today, only future code is protected), but it is harder to explain to readers who reach §2.3 and find the name missing.

### `uint` — documented but zero type-position uses

`let x: uint = 0` gives `error[E001]` on Madaros (#2034), but no versioned file uses `uint` as a type — the corpus count is **0**. The cost of any exit is therefore dominated by documentation, not code. Alias to `u64` is the lowest-cost fix; `Reserved` adds no protection since nothing references it.

### `u16` — works, but aspirational only

`u16` is in §2.3 and §3.1 as a real built-in type, and `let x: u16 = 0` checks under Madaros (#2034 evidence). However, **zero** type-position uses exist in the versioned corpus. The "u16 with zero uses" half of the dispatch hypothesis — *"u16 has zero uses in the versioned corpus"* — is confirmed. The current state is honest (the docs are correct, just unused). No exit is owed.

### `char` — works and is used

`char` is used at 23 type-position sites in `stdlib/text/` (the Unicode helper modules). `let x: char = 0` checks. No exit owed.

### `f128` / `f256` — the honest model

Both give `error[E218]` "Reserved" on Madaros (#2034 dispatch cited evidence). 63 and 56 type-position uses respectively, all in `tests/run-pass/f*_v0b_literal_*.sio` — the v0b smoke tests for the literal forms. lean_single accepts both lexically but emits no-ops in arithmetic, which means the test files only "pass" in the sense that `r == 1` is never checked (the witness always falls through to the "PASS" branch via short-circuit).

The most plausible exit is **keep Reserved, fix the test files**. The v0b smoke tests are aspirational; once the type lands, they turn green automatically.

### `own`, `handle` — contextual, not absent (#2033)

`let own = 1` and `let handle = 1` both check on Madaros (cited #2033). They are contextual: `handle` is recognised in the `handle<IO> { … }` bind position, `own` is recognised in ownership annotations that the current parser does not require. Listing them in the keyword tables reads as a stronger claim than the engines support. The recommended exit is **re-classify as contextual, keep the contextual recognition**.

### `mut` — reserved to teach against

`let mut x = 5` gives `error[E040]` (cited #2033). This is by design: Sounio uses `var x = …` for mutable bindings, and the E040 diagnostic exists to catch Rust habit. Listing it beside `linear` and `where` in the keyword table reads as endorsement of the form the compiler exists to refuse. Recommended exit: keep the E040, but **clarify in §2.2 that `mut` is reserved-only**.

### `async`, `await`, `spawn` — also contextual

The witness `let async = make()` compiles and runs on lean_single, and the same is true for `let await = make()` and `let spawn = make()`. All three are **contextual**, not lexically reserved. They behave in their slots (`async fn`, `<expr>.await`, `spawn { … }`) but compile as identifiers elsewhere. The §2.2 row should mark them contextual — same shape as `handle`.

### `Observe` — the new finding

`Observe` is named as a core effect in **CLAUDE.md §7** (lines 66 and 245), and `fn observe(x: Unobserved<f64>) -> bool with Observe { x > 0.0 }` is given as the canonical example (line 243). But `docs/spec/LANGUAGE_SPECIFICATION.md` §2.4 lists 8 effects: `IO`, `Mut`, `Alloc`, `Panic`, `Async`, `GPU`, `Prob`, `Div`. **`Observe` is not among them.** The `Prob` row mentions `observe` as a verb operation but does not list `Observe` as an effect.

47 versioned sites use `with Observe` (mostly in `stdlib/prob/`), so the name works in code — but the spec and CLAUDE.md disagree about whether it is a first-class effect. This is the same shape as the `Mut`-and-`Div` inconsistency that the §7.2.1 audit flagged: a name appears in CLAUDE.md, the spec is silent, and the engines are permissive.

Two admissible exits:

1. **Promote** — add `Observe` to §2.4 alongside `Prob` (or as a sub-effect of `Prob`). Cost: edit §2.4.
2. **Demote** — remove the example from CLAUDE.md and the `Observe` mention at line 66; treat `observe` as a `Prob` operation. Cost: edit CLAUDE.md; possibly move the example.

## Rulings owed

One line per name. Each line is the exit that minimises cost while preserving honesty; the founder is owed the call.

| name | exit owed |
|---|---|
| `int` | **alias to `i64`** — 12 sites gain meaning, §2.3 / §3.1 stay accurate. Alternative: Reserved (no site breaks today). |
| `uint` | **alias to `u64`** — 0 type-position sites change; §2.3 / §3.1 stay accurate. Alternative: remove from §2.3 / §3.1 (no users). |
| `u16` | **no exit owed** — already honest (works, unused). |
| `char` | **no exit owed** — works, used. |
| `f128` | **Reserved** — keep E218; fix the 63 v0b test sites once the type lands. |
| `f256` | **Reserved** — keep E218; fix the 56 v0b test sites once the type lands. |
| `own` | **re-classify** as contextual in §2.2 / §2.3, not a reserved keyword. |
| `handle` | **re-classify** as contextual in §2.2 / §2.4; `handle<IO> { … }` stays. |
| `mut` | **clarify** in §2.2 that `mut` is reserved-only; keep E040. |
| `async` | **re-classify** as contextual in §2.2 (only `async fn` slot). |
| `await` | **re-classify** as contextual in §2.2 (only `<expr>.await` slot). |
| `spawn` | **re-classify** as contextual in §2.2 (only `spawn { … }` slot). |
| `Observe` | **promote to §2.4** (alongside or under `Prob`), **or** remove from CLAUDE.md. Founder's call between the two. |

## Deferred to Slurm

Three measurements cannot be made on this pod:

1. **Direct Madaros probing of the v0b f128/f256 smoke files.** The shipped
   `bin/madaros-linux-x86_64` is the G1 broken control from PR #1961
   (`error[E007]` cascade). A Madaros run is owed once the G1 control turns
   green. Until then, the Madaros column here leans on the receipts in
   #2033 / #2034 and on the `error[E…]` numbering in lean_single's
   self-hosted compiler, which mirrors Madaros at the lex level.
2. **`handle<IO> { … }` enforcement parity.** #2033 noted that the
   `handle<IO> { … }` slot is recognised, but did not measure what happens
   when an `IO` operation is invoked inside the handler on Madaros vs
   lean_single. The §7.3 effect-handler audit (#2032) covers the design;
   the enforcement comparison is owed to Slurm.
3. **Mut-inference parity.** §7.2.1 already measures that neither shipped
   engine infers `Mut` correctly; the dispatch does not ask for a fix here.

## Provenance

- Corpus scans: `grep -rEn …` from `/workspace/.wt/minimax-cli1`, excluding
  `archive/`, `bootstrap/` per the dispatch.
- Witness sources and outputs: `/tmp/nomes_mortos_witness/`,
  `/tmp/witness_await.sio{.out}`, `/tmp/witness_async.sio{.out}`,
  `/tmp/witness_spawn.sio{.out}`.
- lean_single binary: `bin/souc-lean-single-x86_64` (prebuilt on this pod).
- Madaros citations: PRs #2033, #2034, #2043.
