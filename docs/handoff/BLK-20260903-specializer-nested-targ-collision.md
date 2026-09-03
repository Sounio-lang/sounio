<!-- docs:meta
topic_id: repo.docs.handoff.blk-20260903-specializer-nested-targ-collision
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.blk-20260903-specializer-nested-targ-collision
-->

# Blocker: BLK-20260903-specializer-nested-targ-collision

```text
Blocker-ID: BLK-20260903-specializer-nested-targ-collision
Status: open
Severity: B1
Class: compiler-semantics
Owner: lang-limits-20260903
Lane: lang-limits-20260903
Worktree: /workspace/.wt/lang-limits
Branch: fix/language-limitations
Files-Owned: self-hosted/check/specializer.sio
Repro: tests/run-pass/specializer_nested_targ_distinct.sio
Observed: two instantiations of one generic template whose type arguments are
  themselves generic mangle to the same name, hash equal, and share a single
  specialization.
Expected: distinct type arguments produce distinct instantiations, or the
  template is left unspecialized — never silently shared.
Acceptance-Gate: scripts/ci/madaros_specializer_nested_targ_gate.sh
Evidence-Level: E3
Fallback-Path: fail closed — refuse the compilation when a poisoned template's
  instantiations carry non-scalar type arguments.
LLM-Offload: not-required
Next-Action: rebuild with the fail-closed refusal and re-run the generics
  differential plus the repro.
```

## Context

`self-hosted/check/specializer.sio` is the only live monomorphization pass
(`check/monomorph.sio` is not imported by `main.sio` or `module_frontend.sio`).
It names every instantiation by mangling the template name with its turbofish
type arguments, and keys both its instantiation cache (`SPEC_GF_INST_HASH`) and
its emitted-struct cache (`SPEC_EMITTED_HASHES`) on `ast_name_hash` of that
mangled name.

`spec_render_type_name` rendered a type argument by taking the head name of a
single-segment named type and **dropping its own type arguments**, falling back
to the literal name `T` for every other shape. Nesting therefore did not reach
the mangled name at all.

## Observed

Madaros v0.80.0, `bin/souc check`.

Function lane — `pick::<Cell<i64>>` and `pick::<Cell<f64>>` both mangle to
`pick__Cell`:

```
error[E009
] at 0
..1018
: argument type does not match parameter
   |
   = expected Cell__i64
   = found Cell__f64
```

The `f64` call site is checked against the clone specialized for `i64`, because
the second instantiation was treated as identical to the first and so never
created its own clone — and never tripped the second-distinct-instantiation
poison guard either.

Struct lane — `Cell<Cell<i64>>` and `Cell<Cell<f64>>` both mangle to
`Cell__Cell`, so `spec_get_or_create_struct_instance` emits one struct and
`spec_emitted_has` hands the same name to both:

```
   = expected Cell__Cell
   = found Cell__Cell__i64
```

In the function lane the collision surfaced as a false rejection. In the struct
lane a single emitted struct is shared between two different field types, which
is a wrong-code shape rather than a rejected program.

## Second, larger defect found while fixing this one

Correct mangling makes two nested instantiations distinct, which then trips the
existing second-distinct-instantiation guard (`SPEC_GF_POISONED`). Poisoning
keeps the template unspecialized — and lowering has no generics concept, so a
poisoned template with struct-typed arguments evaluates to zero.

Measured on both the committed `bin/madaros-linux-x86_64` and a local build,
via the `bin/souc` wrapper:

| program | committed binary | local build |
|---|---|---|
| one nested instantiation | `7` | `7` |
| two nested instantiations | rejected, E009 | rc=0, prints `0` and `0.000000` |
| two scalar instantiations (`pick::<i64>` / `pick::<f64>`) | `7`, `1.5` | `7`, `1.5` |
| two struct instantiations (`pick::<P>` / `pick::<Q>`, plain structs) | **rc=0, prints `0`, `0`** | **rc=0, prints `0`, `0`** |

The last row is the important one: it reproduces identically on the committed
binary, with no nesting involved. **Poisoning already miscompiles struct-typed
instantiations on `main` today.** The mangling fix does not introduce that path;
it moves the nested case into it, which is why the fix cannot ship without the
refusal below.

## Fix

`self-hosted/check/specializer.sio`:

- `spec_render_type_string` renders type arguments recursively, encoding
  nesting as `_Lb_` / `_Cm_` / `_Rb_` so the result stays in `[A-Za-z0-9_]`.
  Shapes with no injective spelling here — qualified paths, tuples, fn types,
  arrays (whose size lives in an `Expr`), `Knowledge`/refinement types and the
  ZD wrappers — render as the empty string.
- `spec_render_type_name` and `spec_mangle_name` return `empty_name()` as the
  "cannot mangle" sentinel, also when the mangled string would exceed 120 bytes
  (`make_name` sets `Name.len` from the string length while copying at most 128
  bytes into `Name.buf`, so a longer name reads out of bounds in `name_eq` and
  `ast_name_hash`).
- The three call sites check the sentinel: `spec_ensure_instance` poisons the
  template, `spec_get_or_create_struct_instance` emits nothing and returns the
  template name, and the generic-struct path rewrite is skipped.

Fail-closed refusal, same file plus the two frontends:

- `spec_type_args_all_scalar` classifies an instantiation's arguments against an
  explicit primitive allowlist. Scalars survive an unspecialized body (measured);
  anything else does not.
- Poisoning a template whose instantiations are not all scalar sets
  `SPEC_UNSOUND_POISON`. `spec_report_unsound_poison()` prints the diagnostic and
  returns true.
- `self-hosted/compiler/main.sio` and `self-hosted/compiler/module_frontend.sio`
  fold that call into their existing verdict checks, so the message appears only
  where the program would otherwise have been accepted.

## Residual — NOT claimed closed

- Multi-instantiation monomorphization. A template used with two distinct
  type-argument lists is still refused when any argument is non-scalar. This
  fix makes the failure honest; it does not add the capability.
- Scalar-argument poisoning keeps its current behaviour, which is measured
  correct on the shapes tested and is *not* proved correct in general.
- Marker aliasing. An identifier that literally contains `_Lb_`, `_Cm_` or
  `_Rb_` can still alias another instantiation's mangled name. Far narrower
  than dropping arguments entirely, but not zero.
- Unrenderable shapes are refused, not supported.
