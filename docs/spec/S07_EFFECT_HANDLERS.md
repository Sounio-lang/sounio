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

Status: **undefined.** No normative statement has been ruled.

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

### 7.3.2 The corpus already contains programs this affects

`examples/effects.sio`, `examples/effects/basic_handler_continuation.sio`,
`examples/showcase/linear_file_server.sio` and `tests/run-pass/closure_linear.sio`
use the construct. Under Madaros they compile, run, and their algebraic-effect
parts do nothing, with no warning. Under lean_single none of them compiles.

## 7.4 Rulings owed

- **Does `handle` exist?** Three answers are coherent and they are very
  different. *Reserved*: the surface is refused with a named diagnostic until
  implemented — honest, and cheap. *Implemented*: handlers acquire lowering and
  execution semantics. *Withdrawn*: the surface is removed. What is not coherent
  is the present state, in which a program that appears to use algebraic effects
  type-checks.
- **If implemented, which discipline?** One-shot or multi-shot continuations;
  deep or shallow handlers; whether a handler may itself perform effects. None
  of these is decidable from the code, because no execution exists to read them
  off.

## Claims forbidden

- Do not describe Sounio as having algebraic effect handlers. The surface exists
  and reaches no backend.
- Do not cite `examples/effects.sio` or `examples/effects/*.sio` as evidence
  that handlers run. That a file uses the syntax says only that the syntax
  parses.
