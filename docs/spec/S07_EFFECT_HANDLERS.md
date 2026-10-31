<!-- docs:meta
topic_id: repo.docs.spec.s07-effect-handlers
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.s07-effect-handlers
-->

# §7 — Effect handlers

Spec-Section: `SOUNIO-SPEC-07`
Frame: `docs/spec/E2E_SPECIFICATION_FRAME.md`

Status: **ruled 2026-10-31** (founder): Implemented, deep handlers, linear one-shot continuations. Backend landing on PR #3371 (phases P0-P5); until P5 closes, do not describe handlers as shipping.

## 7.1 `handle` reaches the type checker and stops there

Measured on `origin/main`, 2026-08-19.

The front end is complete:

| stage | evidence |
|---|---|
| token | `TokenKind::Handle`, `TokenKind::Handler` (`parser/parser.sio:775,809`) |
| parse | `parse_handle_expr` (`parser/exprs.sio:564`), produces `ExprKind::ExprHandle` |
| check | `check_handle_expr` (`check/check.sio:24749`) |

The back end is absent. `ExprHandle` occurs **zero times** across
`self-hosted/ir/`, `self-hosted/native/` and `self-hosted/enir/`. Positive
control from the same command, same directories: `ExprCall` 23, `ExprBinary` 17,
`ExprIf` 4.

A handler expression is therefore parsed, type-checked, and does not appear in
any intermediate representation.

This is stronger than the frame's earlier reading. The frame recorded that *the
CPS path* has no execution semantics. Measured, **no path does**.

## 7.2 The checker's own two silent losses

`check_handle_expr` resolves the handled effect name and admits it with

    if eff_id >= 0 && c.current_effect_count < 8 {

so `handle` inherits both silences of §6.2 at the point where an effect is
supposedly *discharged*: an unrecognised effect name is ignored without a
diagnostic, and a function already carrying eight effects has the handled effect
dropped.

## 7.3 Measured: the expression is erased in silence on one engine and unknown on the other

Runtime witness, 2026-08-19, both engines, on Slurm
(`docs/audit/HANDLE_EXECUTION_WITNESS_2026-08-19.md`):

| engine | result |
|---|---|
| **Madaros** (default) | **Silent erasure.** `handle<IO>` *and* `handle<NotARealEffect>` both check OK, emit an ELF, exit 0, and print nothing. The whole expression is discarded. |
| **lean_single** (the seed) | **Refuses** — `error[E200]: undefined identifier \`handle\`` |

Controls close the other three outcomes. A control program differing only by the
absence of `handle` compiled and printed `BODY_MARK` on **both** engines, so the
empty cells are the construct and not the instrument. `HANDLER_MARK` never
appears, so this is not *works*. Madaros exits 0, so this is not *crash*.

**The negative control is the sharper finding.** On Madaros an invented effect is
**indistinguishable from `IO`**. The checker does not separate a real effect from
a fake one — independently confirming §6.2's `eff_id >= 0` guard and #1953's
`Foo` → `check: OK` / silence.

### 7.3.1 lean_single's refusal is ignorance, not design

`E200: undefined identifier` is the diagnostic the seed gives any name it does
not know. It does not treat `handle` as a keyword at all. This is **not** a typed
`Reserved` refusal.

The consequence for 7.4 is direct: **neither engine refuses `handle` by
decision.** One erases it silently, the other has never heard of it. The
*Reserved* option below is therefore not the cheap one already half-present — it
does not exist anywhere and must be built like the others.

### 7.3.2 The corpus contains no live handler at all

Census, 2026-08-19 (`docs/audit/HANDLE_GREEN_CENSUS_2026-08-19.md`): **zero**
`tests/run-pass/` files contain a live algebraic-effect `handle` expression and
pass under Madaros. The perverse cell — a green that is really an erased
construct — is empty.

The files that appear to use handlers are **name traps**:

| site | what it actually is |
|---|---|
| `tests/run-pass/closure_linear.sio:14` | `let handle = open_file(42)` — a variable |
| `tests/run-pass/linear_correct_consume.sio:13` | `let handle = FileHandle { fd: 42 }` |
| `tests/run-pass/async_spawn.sio:11` | `let handle = spawn {` |
| `examples/showcase/linear_file_server.sio` | `handle` as a field and a parameter of a linear `FileHandle` |
| `self-hosted/ir/lower.sio:4810` | `lower_handle_buf_arg_index` — the runtime handle table, unrelated |

In `examples/effects*.sio` the handler code sits **inside comments**; the live
program below is a stub.

> **Correction, 2026-08-19.** An earlier revision of this section listed four of
> those files as programs whose algebraic-effect parts silently do nothing. That
> was wrong: none contains a handler. The claim came from a pattern that matched
> the identifier `handle` — the exact trap the census's own negative control was
> written to exclude.

**This narrows §7.4 rather than widening it.** Silent erasure has, today, no
victims in the corpus: there is nothing to break by refusing `handle`, and
nothing to fix by implementing it. Whichever way 7.4 is ruled, the cost of
ruling is the lowest it will ever be.

### 7.3.3 One green tests the message, not the mechanism

The census records that `handler_discharge.sio` is green on both engines
**because it was rewritten to `println("handler: PASS")`**. The file's name
claims a discharge test; what it asserts is a string. Reclassifying it is a
founder decision and the census did not touch it.

### 7.3.4 Neither form the language specification documents is implemented

`docs/spec/LANGUAGE_SPECIFICATION.md` §4.8 documents two effect expressions.
Both were tested on both engines.

**§4.8.1 — `with handler { ... perform E::op() ... }`**

| engine | result |
|---|---|
| Madaros v0.80.0 | `module failed to parse` |
| lean_single | `error[E200]: undefined identifier \`handler\``, `error[E200]: undefined identifier \`perform\`` |

`perform` is listed as a keyword at `LANGUAGE_SPECIFICATION.md:114`. lean_single
resolves it as an ordinary identifier and fails to find it, so on that engine it
is not a keyword at all.

**§4.8.2 — `handle name for Effect { on op(arg) -> { resume(v) } }`**

| engine | result |
|---|---|
| Madaros v0.80.0 | `module failed to parse` |
| lean_single | compiles, emits an ELF, no diagnostic |

**The lean_single result is silence, not support.** Negative control: replacing
the whole block with nonsense that keeps only its shape —

    zorble qwertyuiop for MyEffect {
        blargh operation(arg) -> { frobnicate(arg + 1) }
    }

— also compiles and emits an ELF. An unrecognised top-level item is skipped
without a diagnostic, so a green here distinguishes nothing. Without that control
the honest-looking conclusion is *"lean_single implements the documented handler
syntax"*, and it is false.

The form §7.3 measures — `handle<IO> { ... }` — is a third spelling, present in
exactly one test. So the specification documents two forms neither engine
implements, the corpus exercises a third, and §7.3 has already measured that
`ExprHandle` reaches `ir/`, `native/` and `enir/` **zero** times.

## 7.4 Ruling (2026-10-31, founder)

**Implemented.** `handle` exists with execution semantics. Discipline:

- **Deep** handlers: resuming re-installs the handler around the resumed
  computation, and a `perform` issued inside a handler clause is handled by
  the next enclosing (outer) handler, never by the clause's own handler.
- **Linear one-shot continuations**: the resume capability must be used at
  most once on every path through a clause. This is enforced by the checker;
  any runtime guard is an assertion backstop only, never the mechanism.
- **Zero resume is affine, with defined teardown** (see FX-5). Strictly-linear
  (zero resume = error) was rejected: it forbids abort/escape handlers.
  Multi-shot continuations and shallow handlers are non-goals.

### Grammar (canonical surface)

    effect E { op1(a: T1, ...) -> R1   op2(...) -> R2 }   // top-level declaration
    handle<E> { body } [with { on op1(x, ...) -> expr
                               on _ -> expr                // optional catch-all
                               return(x) -> expr }]        // optional return clause
    perform E.op1(args)
    resume(value)        // only inside `on` clauses; linear

The two `LANGUAGE_SPECIFICATION.md` section 4.8 spellings remain unimplemented
(non-goal). `resume(e)` has the handle answer type: it evaluates to whatever
the resumed computation produces when it next performs or completes the
handle. `on` clauses may appear in any order; a duplicate clause for one op
is a compile error.

### Operational semantics (small-step)

Notation: `F = (E, H, k)` is a handler frame for effect `E`, clause map `H`,
outer continuation `k`. Answer type `A` is the handle expression's type.

- **FX-1 (install)** `handle<E>{B} with H` pushes frame `F` and evaluates
  `B`. Without `with`, `H` is empty (any perform of `E` in `B` then fails
  FX-8 at compile time unless `B` performs nothing of `E`).
- **FX-2 (capture)** `perform op(v)` with nearest enclosing frame `F` for
  `E` captures the delimited continuation `K` (the rest of `B` up to `F`),
  suspends the body stack segment, and transfers control to `F`'s clause.
- **FX-3 (dispatch/deep)** `F` is disarmed while its clause runs: a
  `perform` inside the clause is dispatched to the next enclosing frame
  for that effect (FX-2 applies there), or is an unhandled perform (FX-8
  makes this unreachable statically when types are known).
- **FX-4 (resume)** `resume(w)` consumes `K`, re-installs `F` around the
  resumed segment, and continues `B` with `w` as the value of the
  `perform`. `resume` evaluates to the next captured operation (dispatched
  per FX-3) or, when the resumed `B` completes under FX-6, to the
  handle's final value.
- **FX-5 (zero resume, affine teardown)** If a clause completes with
  value `u` and `K` unconsumed: `K`'s stack segment is freed
  deterministically, `u` is the handle's value, and FX-6 does not apply.
  Linear obligations owned by the suspended computation must be
  discharged at the perform point; the checker refuses a clause that
  would drop one.
- **FX-6 (return)** When `B` completes with value `v` (initially or via
  FX-4) and a `return(x) -> R` clause exists, the handle's value is
  `R[x:=v]`; otherwise it is `v`. The frame is popped.
- **FX-7 (linearity)** Statically: the resume capability is used 0..1
  times on every path through each clause. Violation is a compile error.
- **FX-8 (exhaustivity, no silent effects)** Every op of `E` performable
  in `B` must have a clause or a catch-all; otherwise compile error.
  Unknown effect names and unknown ops in `handle`/`perform` are compile
  errors (this closes the section 7.2 silence at handler sites).
- **FX-9 (nesting)** Frames nest dynamically; FX-2 always selects the
  nearest enclosing frame for the performed effect.

Implementation note: the runtime model chosen in phase P0 is one stack
segment per handler (OCaml-5 style), proven by probe; copying the body
stack in place is excluded. Each FX rule above has at least one derived
test in phase P5 (`tests/run-pass/fxN_*.sio`).

## Claims forbidden

- Until phase P5 closes, do not describe Sounio as shipping algebraic effect
  handlers; the section 7.4 semantics are ruled and landing on PR #3371.
- Do not cite `examples/_attic/effects.sio` or `examples/effects/*.sio` as evidence
  that handlers run. That a file uses the syntax says only that the syntax
  parses.
