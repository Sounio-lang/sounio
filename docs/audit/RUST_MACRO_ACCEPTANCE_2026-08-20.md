<!-- docs:meta
topic_id: repo.docs.audit.rust-macro-acceptance-2026-08-20
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.rust-macro-acceptance-2026-08-20
-->

---
title: Four Rust macros are accepted, and three of them SIGSEGV (SUPERSEDED by #2247)
status: superseded
date: 2026-08-20
last_validated: 2026-08-20
engines: Madaros v0.80.0 (default), lean_single
---

# Four Rust macros are accepted, and three of them SIGSEGV

> **SUPERSEDED 2026-08-28 by #2247. Everything below this box is a historical
> record of what was true on 2026-08-20; none of it describes the compiler now.**
>
> Re-measured on a Madaros built from current source (the committed prebuilt ELF
> predates the fix and still answers `check: OK`, so measuring through
> `./bin/souc` reproduces the OLD result and is how this page could look current
> while being false):
>
> | written | Madaros (source-built) | lean_single |
> |---|---|---|
> | `println!("x")` | **`error[E043]`** | `error[E200]` |
> | `print!("x")` | **`error[E043]`** | `error[E200]` |
> | `panic!("x")` | **`error[E043]`** | `error[E200]` |
> | `assert!(1 == 2)` | **`error[E043]`** | `error[E200]` |
>
> All four are now refused before lowering, so neither the SIGSEGV nor the inert
> `assert!` this page documents can still occur. The table below is kept because
> it is the evidence that motivated the fix, not because it is current.
>
> **Two things this page's successor should carry, both still open:**
>
> 1. **The engines disagree on the identity, not the verdict.** Madaros says
>    `E043`; lean_single says `E200` (undefined identifier). Both refuse, which
>    is the safety property — but a language-level grammar rule should not have
>    a different diagnostic identity depending on which frontend observed it.
>    This matters directly to whether `DiagnosticId` can be relied on by an LSP
>    or by tooling.
>
> 2. **E043 is narrower than its message says.** It reads *"Sounio does not use
>    Rust macros"*, but it fires on lexical adjacency alone. Measured on the
>    source-built compiler:
>
>    ```
>    x!(y)     -> error[E043]
>    x! (y)    -> accepted, rc=0
>    x !(y)    -> accepted, rc=0
>    x ! (y)   -> accepted, rc=0
>    foo::bar!(z) -> error[E004], NOT E043
>    ```
>
>    So it means "bare identifier followed by a tightly-adjacent bang-call", and
>    the path-qualified macro form — the shape most Rust code actually uses for
>    anything namespaced — is not recognised as a macro at all.
>
> `tests/compile-fail/parser_rust_macro.sio` remains on
> `tests/vacuous_expect_baseline.txt` for a reason that follows from (1): the
> Full Test Suite runs lean_single, which produces E200, not the message the
> fixture pins. The fixture becomes honest on the day the engines agree.


`CLAUDE.md` §7 lists `println!("hi")` under *"Critical differences — these are
compile errors"*. `tests/compile-fail/parser_rust_macro.sio` exists to enforce it,
with `//@ error-pattern: macro` and the description *"Rust macros are not
supported in Sounio"*.

## The compile-fail test does not fail

| engine | `tests/compile-fail/parser_rust_macro.sio` |
|---|---|
| lean_single | `error[E200]` — refused |
| **Madaros v0.80.0** | **`check: OK`** |

A sixth engine-divergent `compile-fail` test, alongside the five frozen by
`scripts/ci/epsilon_engine_parity_gate.sh`.

## Exactly four names are accepted

| macro | Madaros `check` |
|---|---|
| `println!` `print!` `assert!` `panic!` | **`check: OK`** |
| `format!` `vec!` `write!` `eprintln!` `todo!` `unreachable!` `debug_assert!` | `error[E137]` |
| `zorble!` (control) | `error[E137]` |

The control matters: the acceptance is **not** blanket silence. A macro name is
resolved, and these four resolve. They are exactly the four whose bare forms exist
in Sounio — `println`, `print`, `assert`, `panic`.

## What they do at run time, and they do not agree

| written | `check` | run |
|---|---|---|
| `println!("x")` | `check: OK` | **`rc=139`** (SIGSEGV) |
| `print!("x")` | `check: OK` | **`rc=139`** |
| `panic!("x")` | `check: OK` | **`rc=139`** |
| `assert!(1 == 2)` | `check: OK` | **runs through** — the next statement executes |
| `assert(1 == 2)` (the correct form) | `check: OK` | halts, as it should |

Two distinct defects wearing one syntax:

- **`println!`/`print!`/`panic!` crash.** A program that typechecks clean
  segfaults with no diagnostic, and the statement after it never runs. Measured
  with `println("ANTES")` before and `println("DEPOIS")` after: only `ANTES`
  reaches stdout.
- **`assert!` is inert.** `assert!(1 == 2)` does not halt — the line after it
  prints. Written by anyone with a Rust habit, it is a safety check that compiles
  clean and never fires. Its correct sibling `assert(1 == 2)` does halt, so the
  difference is exactly one character.

## Corpus exposure: latent

| macro | sites |
|---|---:|
| `assert!` | **0** |
| `format!` | 17 (all refused by `E137`) |
| `print!` | 6 |
| `panic!` | 3 |
| `println!` | 3 |

The `println!`/`print!`/`panic!` sites are in `examples/` and `tests/`. One is in
`stdlib/pbpk/regulatory.sio` — checked, and that file **does not parse** at all, so
it is not live code producing a wrong result.

`assert!` at zero is the number that matters: the most dangerous of the four has
not been written yet.

## What this does not claim

Not that the macro syntax should be supported. `CLAUDE.md` is right about the
intent — the defect is that the refusal is absent on the default engine, and that
three of the four fail as a crash rather than a diagnostic. Not that any current
result is wrong: exposure is measured above and it is latent.

## Reproduce

    export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
    printf 'fn main() -> i32 with IO {\n    println("A")\n    println!("B")\n    println("C")\n    0\n}\n' > /tmp/m.sio
    ./bin/souc check /tmp/m.sio        # check: OK
    ./bin/souc run   /tmp/m.sio        # prints A, then rc=139
    ./bin/souc check tests/compile-fail/parser_rust_macro.sio                          # check: OK
    SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check tests/compile-fail/parser_rust_macro.sio  # error[E200]
