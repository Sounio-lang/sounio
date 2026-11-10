<!-- docs:meta
topic_id: repo.docs.spec.s07-effect-handlers
authority: repo_only
audience: users
last_validated: 2026-10-29
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.s07-effect-handlers
-->

# §7 — Effect handlers

Spec-Section: `SOUNIO-SPEC-07`
Frame: `docs/spec/E2E_SPECIFICATION_FRAME.md`

Status: **implemented in Madaros**, following the founder ruling of 2026-10-31:
deep handlers with linear one-shot continuations. PR #3371 merged into `main`
as `4a49fe25c4` on 2026-11-02. See the [current implementation status](#status-2026-11-06-sio-417-p5-merged)
for the supported subset and limits. Sections 7.1–7.3 preserve earlier dated
measurements; their descriptions of missing registration and execution no
longer describe current source.

## 7.1 `handle` reaches the type checker and stops there

Measured on `origin/main`, 2026-08-19. The line references and the `ExprHandle`
count below were re-checked on 2026-10-29 (`dbe8b6d2d`).

The front end is complete:

| stage | evidence |
|---|---|
| token | `TokenKind::Handle`, `TokenKind::Handler` (`parser/parser.sio:992,1020`) |
| parse | `parse_handle_expr` (`parser/exprs.sio:683`), produces `ExprKind::ExprHandle` |
| check | `check_handle_expr` (`check/check.sio:32728`), reached only on the by-value checker path, not on the in-place path Madaros uses |

The back end is absent. `ExprHandle` occurs once in `self-hosted/ir/`
(`lower.sio:10282`, only to flag a handler as control flow the inline
re-lowering cannot follow) and not at all in `self-hosted/native/` or
`self-hosted/enir/`. Positive
control from the same command, same directories, re-run on 2026-10-29
(`grep -rw <name> self-hosted/ir self-hosted/native self-hosted/enir | wc -l`):
`ExprCall` 83, `ExprBinary` 53, `ExprIf` 47, `ExprHandle` 1.

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

## 7.3 Measured: the expression is refused by accident on one engine and unknown on the other

> **Re-measured, 2026-10-29 (`dbe8b6d2d`).** The 2026-08-19 witness below
> recorded silent erasure on Madaros. That is no longer the behaviour: Madaros
> now stops at the checker. The lean_single row is unchanged.

Runtime witness, 2026-08-19, both engines, on Slurm
(`docs/audit/HANDLE_EXECUTION_WITNESS_2026-08-19.md`):

| engine | result |
|---|---|
| **Madaros** (default) | **Refuses, without a code.** `handle<IO>` *and* `handle<NotARealEffect>` both fail `check` and `run` with exit 1 and `error: expression kind was not checked in place (Expr.kind dispatch missed)`. No ELF is emitted. The in-place checker has no `ExprHandle` arm (`check/check.sio:7814`, `checker_note_kind_dispatch_miss`). On 2026-08-19 the same programs checked OK, exited 0 and printed nothing. |
| **lean_single** (the seed) | **Refuses** — `error[E200]: undefined identifier \`handle\`` |

Controls close the other three outcomes. A control program differing only by the
absence of `handle` compiled and printed `BODY_MARK` on **both** engines, so the
empty cells are the construct and not the instrument. `HANDLER_MARK` never
appears, so this is not *works*. On 2026-08-19 Madaros exited 0; it now exits 1
with an internal dispatch message, which is a refusal by omission, not a typed
diagnostic.

**The negative control is still the sharper finding.** On Madaros an invented
effect remains **indistinguishable from `IO`**: both now fail with the same
dispatch message, before any effect is looked at. On 2026-08-19 the checker
did not separate a real effect from a fake one — independently confirming §6.2's `eff_id >= 0` guard and #1953's
`Foo` → `check: OK` / silence.

### 7.3.1 lean_single's refusal is ignorance, not design

`E200: undefined identifier` is the diagnostic the seed gives any name it does
not know. It does not treat `handle` as a keyword at all. This is **not** a typed
`Reserved` refusal.

The consequence for 7.4 is direct: **neither engine refuses `handle` by
decision.** One stops on an internal dispatch gap, the other has never heard of it. The
*Reserved* option below is therefore not the cheap one already half-present — it
does not exist anywhere and must be built like the others.

### 7.3.2 The 2026-08-19 corpus census found no live handler

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

**Update, 2026-10-29.** Two later tests do contain live `handle<IO> { ... }`
expressions: `tests/run-pass/handle_if_f32_literal_join.sio` and
`tests/run-pass/handle_match_f32_literal_join.sio` (both `//@ check-only`, both
`//@ requires: madaros`). They verify the type of the handler body, not
execution. On `dbe8b6d2d` both fail `check` under Madaros with exit 1 and
`error: expression kind was not checked in place (Expr.kind dispatch missed)`,
as does the known-failure `tests/run-pass/handler_discharge.sio`. They do not
break CI because the suite skips `requires: madaros` tests unless
`SOUNIO_MADAROS_AVAILABLE` is set (`scripts/dev/run_sio_test_suite_v2.sh:576`;
no setter found under `.github/`).

**This narrows §7.4 rather than widening it.** No program in the corpus depends
on handler *execution*, so nothing breaks by refusing `handle` on that account,
and nothing is fixed by implementing it. The two check-only tests above are the
exception: Madaros refuses them today, and the skip hides it.
Whichever way 7.4 is ruled, the cost of ruling is still low.

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

## Status (2026-11-06, SIO-417 P5 merged)

The section 7.4 semantics are implemented in Madaros. PR #3371 merged into
`main` on 2026-11-02 (`4a49fe25c4`), including:
parser/AST (`ExprPerform`/`ExprResume`, `on`/`return` clauses), checker
(op resolution FX-8, linear resume FX-7, diagnostics 290-298), lowering
(synthetic clause functions, handler frame), and the native backend
(builtins `__sounio_fx_install/perform/resume/body_done`: one stack segment
per handler, deep dispatch, affine zero-resume teardown). Every FX rule has
a derived test (`tests/run-pass/fx{1,2,2b,3,4,5,6,9}_*.sio`,
`tests/compile-fail/fx{7,8}_*.sio`), and `tests/run-pass/handler_discharge.sio`
prints `handler: PASS` for the reason its name claims.

v1 limits (named compile-time refusals, never silent miscompiles): scalar
i64 ops and answers only (no f64, no string: E299), perform only in the
handled body of the same function, no captured locals in clauses, at most
5 perform arguments, at most 12 clauses per handle, handler nesting depth
at most 8 (E299). All caps fail closed at compile time; none may silently
drop a clause, a perform record, or a frame (review round on PR #3371:
record table deduplicates per (frame, op) and refuses past 64 distinct
performed ops per function region, clause keys colliding under the name
hash are refused at lowering, and the runtime scratch/drive-top slots
live at frame words 59/60, above the 12-clause table at words 15..50).

Implementation note (regression guard): the fx lowering methods in
`self-hosted/ir/lower.sio` must stay inside the existing `impl Lowerer`
block. Wrapping them in a second `impl Lowerer` block makes the seed
mis-register the methods that follow it; the resulting compiler segfaults
lowering any closure (bisected 2026-11-01, SIO-417 P5). Do not reintroduce
a separate impl block for these methods.

- Do not cite `examples/_attic/effects.sio` or `examples/effects/*.sio` as evidence
  that handlers run. That a file uses the syntax says only that the syntax
  parses.
