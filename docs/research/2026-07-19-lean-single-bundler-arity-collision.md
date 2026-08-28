<!-- docs:meta
topic_id: repo.docs.research.2026-07-19-lean-single-bundler-arity-collision
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.2026-07-19-lean-single-bundler-arity-collision
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# lean_single bundle collision — silent miscompile on same-name / different-arity functions

**Status:** complete (2026-07-19)
**Component:** `self-hosted/compiler/lean_single.sio` (the seed / fallback engine)
**Surfaced by:** the SciPy↔Sounio stats-distribution parity vertical (PR #1218),
which could not co-import `stats::densities` and `stats::distributions` and had
to fall back to one-module-per-emitter.

## Symptom

Compiling a program that bundles two modules which each define a `pub fn` of the
**same name but a different arity** (e.g. `stats::densities::normal_pdf(x, mu,
sigma)` — 3 args — and `stats::distributions::normal_pdf(dist, x)` — 2 args)
produced a **binary at `rc = 0` that silently computed the wrong answer**, or,
depending on which definition was registered first, ran a program whose visible
output happened to be correct while internal calls were miscompiled.

Minimal reproduction (two modules, one free `foo` each):

```
// bza/m.sio:  pub fn foo(x: f64) -> f64 { x + 1.0 }
//             pub fn use_a() -> f64 { foo(10.0) }        // wants bza::foo -> 11
// bzb/m.sio:  pub fn foo(x: f64, y: f64) -> f64 { x + y }
//             pub fn use_b() -> f64 { foo(10.0, 20.0) }  // wants bzb::foo -> 30
// main: use bza::m::*; use bzb::m::*;  use_a(); use_b()
```

Result before the fix: `use_a() = 11`, `use_b() = 11` (the call in `use_b`
resolved to `bza`'s 1-arg `foo`, dropping the second argument) — compiled at
`rc = 0`, **no error surfaced to the exit code**, even though the compiler
printed `error: arity mismatch` to stdout.

## Root cause

Three interacting facts:

1. **Call resolution is name-only, first-match-wins.** `fn_find(ns, ne)` scans
   the flat function table and returns the first entry whose name matches
   (`name_eq`), with no arity or signature check. Two same-named functions
   collapse to whichever was registered first.

2. **Duplicate detection keys on `(name, receiver-hash)`, but resolution ignores
   the receiver-hash.** A free function whose first parameter is a struct gets a
   UFCS receiver-hash (so it can be dot-called), which made the registration-time
   duplicate check *miss* the `stats` collision entirely — while `fn_find` still
   collapsed the two by name.

3. **A deliberate `from_import` guard swallowed the resulting error.**
   `tc_mark_failed()` returns early without setting `TYPECHECK_FAILED` when the
   erroring function is imported (effect-bit 2048). This guard exists so that a
   large imported stdlib module does not fail your compile because some unused
   function deep inside it does not type-check in isolation. But it also swallowed
   the *arity-mismatch* error raised when a mis-resolved call was actually
   compiled — turning a hard error into a silent miscompile and an emitted binary.

## Fix

Two surgical changes, both in `lean_single.sio`:

- **Honor the real mis-resolution (hard failure).** `tc_wrong_arity_hard` — the
  one call-site arity check (line ~14774) — now sets `TYPECHECK_FAILED` directly
  after `tc_mark_failed()`, bypassing the `from_import` guard. A call whose
  argument count does not match the resolved definition is a genuine
  mis-resolution, not a tolerable standalone-typecheck artifact, so it fails hard
  wherever it lives. Every other tolerated class (effects, generics, type
  mismatches in dead imported code) stays tolerated — the guard is untouched.

- **Name the collision (warning).** At function registration, a same-name /
  different-arity pair that is *not* two genuine impl-block methods on distinct
  receivers now emits `warning: conflicting definitions of the same function name
  with different arity across the bundle …`. This is a *warning*, not a failure:
  if the shadowed definition is never actually called with its own arity it is
  dead and harmless, and the bundle still compiles. The warning gives the
  cross-module context ("two `foo` of different arity") that the bare
  "arity mismatch at line N" hard error, raised later in some library file,
  otherwise lacks. A new per-function flag `FN_IN_IMPL` distinguishes true
  `impl`-block methods (dot-called, receiver-aware — legitimately overloadable
  across types) from UFCS free functions (name-only-resolved — collide).

Net behavior: benign coexistence compiles with a warning; an actual wrong-arity
mis-resolution fails hard with a clear exit code.

## Verification

- **Repro matrix:** the two-module `foo` case and the real `stats` bundle now
  fail hard (`rc = 1`); a benign coexistence (two modules whose shadowed
  definition is never called) compiles green with only the warning; two modules
  that both define a 0-arg `run_tests` (every module does) stay a soft duplicate
  warning, unchanged.
- **Fixed point:** the edited compiler still self-compiles to a bit-identical
  fixed point (stage2 == stage3), so the bootstrap is intact.
- **Full multi-module corpus sweep** (687 files with ≥2 imports, comparing the
  fixed compiler against a current-source build *without* the fix, to isolate the
  change from the stale shipped binary): **exactly one net-new hard failure**,
  `tests/stdlib/chemistry/test_lib_surface.sio` — a genuine pre-existing latent
  bug (a private `rk4_step` in `stdlib/epistemic/ode.sio` mis-resolving to the
  glob-imported public `ode::rk4::rk4_step` of a different arity). Fixed in this
  change by renaming the private helper to `eode_rk4_step` (local, private, one
  self-call). Post-fix the sweep reports zero net-new failures.

## Scope and residual

- This makes the collision **fail loudly**; it does **not** add function
  overloading or `module::fn` qualified calls. Sounio has no overloading by
  design, so two modules that expose the same-named-but-different function are
  still not co-usable — you rename one or import selectively. That is unchanged.
- **Not covered:** same-name / **same-arity** / different-body (or different-type)
  collisions still resolve first-wins silently. Catching those would require
  honoring the type-mismatch (E001) path or adding overloading — a larger,
  separate change. The 139 `run_tests` (all 0-arg) are the benign population that
  this residual deliberately keeps compiling.
