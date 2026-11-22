<!-- docs:meta
topic_id: repo.docs.decisions.adr-013-format-printing
authority: repo_only
audience: users
last_validated: 2026-10-23
validated_by: tests/run-pass/print_format_placeholders.sio
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-013-format-printing
-->

# ADR-013: Format printing with `{}` placeholders

**Status**: accepted
**Date**: 2026-10-22

## Context

`print` and `println` took one argument on both engines and ignored the rest
without a diagnostic. `println("x={}", x)` compiled, ran, and printed `x={}`.
Measured on 2026-10-22, before the change (escape-aware search for a call whose
first argument is a string literal followed by a comma; 47 files):

- 33 of them passed `souc check` on Madaros and so dropped their arguments when
  run. 25 are under `examples/_attic/`. Among the others,
  `stdlib/chemistry/kinetics.sio` printed every line of its GUM budget and
  literature check as `lit={} act={} tol={}`, and six stdlib files
  (`medical/sedenion_eeg`, `medical/sedenion_imaging`, `medical/sedenion_pbpk`,
  `ode/pbpk3_demo`, `test/assert_advanced`, `types/refinement`) used a
  concatenating style, `println("x = ", x)`, that printed only `x = `. In
  `assert_advanced` that included the expected and actual values of every
  failed assertion.
- On lean_single, three of them compiled: `kinetics.sio`,
  `tests/regression/linalg_qr_alias_test.sio` and
  `examples/wave2_causal_simpson.sio`.
- 580 single-argument string literals in 118 files contain `{` or `}` and are
  meant to print those characters.

There were two ways to close the silent drop: refuse a second argument, or give
it a meaning. The founder chose to give it a meaning.

## Decision

With **two or more** arguments, the first argument of `print` / `println` is a
format:

1. It must be a string literal. Its placeholders are checked when the program is
   compiled, not at run time.
2. Each `{}` is replaced by the next argument, printed exactly as `print(arg)`
   prints it on that engine.
3. `{{` and `}}` print a single `{` and `}`.
4. The number of `{}` must equal the number of arguments after the format.
5. Anything else that starts with `{` is refused: format specifiers such as
   `{:.1}`, and positional or named placeholders such as `{0}` and `{x}`. So is
   a lone `}`.
6. Arguments are evaluated left to right, each one when its `{}` is reached.
   Literal text before a placeholder is printed before that placeholder's
   argument is evaluated.
7. A trailing comma after the last argument is accepted, as in any call.

With **one** argument nothing changes: `println("{}")` prints `{}`. The 580
literals above keep their meaning, and no existing single-argument call needs
escaping.

Every refusal is a located compile error. The tests pin these messages:

| case | message contains |
|---|---|
| count differs | `placeholder(s) but` |
| specifier | `format specifiers` |
| lone `}` | `unmatched` |
| format not a literal | `must be a string literal` |

## Implementation

- **Madaros**: the parser rewrites the call into a block of single-argument
  calls, `{ print("x="); print(x); println() }`
  (`self-hosted/parser/exprs.sio`, `fmt_print_kind` and
  `rewrite_format_print`). Checking, effects, the `Unobserved` boundary and
  lowering then treat every piece exactly as a call the programmer wrote.
- **lean_single**: `compile_format_print` in
  `self-hosted/compiler/lean_single.sio` emits the same sequence directly, on
  the x86-64 and aarch64 paths. The pieces between placeholders are printed by
  `emit_print_src` / `emit_print_a64_src`, which decode escapes as the
  single-argument path does.

## Consequences

- The output of a placeholder is whatever `print` produces for that type. The
  engines already differ on some of these, for example the digits printed for
  an `f64`. The format inherits those differences and adds none of its own;
  `scripts/ci/engine_parity_gate.sh` reports them.
- Precision and width specifiers are a possible later extension. Rule 5 refuses
  them today, so adding them later breaks no program that compiles now.
- The files listed under Context were rewritten to the format form: `{:.1}`
  became `{}` (`kinetics.sio`, `wave2_causal_simpson.sio`), and the
  concatenating calls became formats (`println("x = {}", x)`). They now print
  the values their lines were written to report. In `kinetics.sio` this shows
  results the braces had hidden: several literature checks print `LIT FAIL`,
  and lean_single prints `0` for every `chebi_map[i].id`.

## Grounded in

- `tests/run-pass/print_format_placeholders.sio`. Under both engines before the
  change it printed `fmt x={} s={} f={}`.
- `tests/compile-fail/format_placeholder_count_mismatch.sio`,
  `format_specifier_refused.sio`, `format_lone_brace_refused.sio` and
  `format_not_literal_refused.sio`.
