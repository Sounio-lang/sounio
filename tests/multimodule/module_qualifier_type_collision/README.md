# A real `use`d module vs. an unrelated same-named lowercase type

Regression fixture for `checker_check_expr_call_inplace`'s `Type::method`
associated-function arm (`self-hosted/check/check.sio`). Driven by
`scripts/ci/madaros_module_qualifier_type_collision_gate.sh`, via `--check`.

| case    | what it pins |
|---------|--------------|
| `basic` | `main.sio` `use`s the real module `zz.sio` (exports `f() -> i64 { 42 }`) and ALSO declares its own unrelated lowercase `struct zz` with a method `f`. Before round 3's fix, the associated-function lookup consulted the global struct/enum tables before ever checking whether the qualifier was a module the caller actually `use`d, so `zz::f()` always bound the struct's method -- with zero call arguments for a method that takes `self`, which typechecked the receiver as an inferred-default type rather than refusing the call, and then failed inside the method body with a spurious "no field n" error. |

**History -- accepting this shape was itself wrong, now refused instead:**

Round 3's fix made `checker_use_module_known` win unconditionally when the
qualifier names a real `use`d module, so `--check` reported OK for
`zz::f()`. Comment 4114680413 (a later round) caught the problem with
that: the checker's fix is a *checker-only* disambiguation -- the LOWERER
(`self-hosted/ir/lower.sio`'s `callee_path_module_stripped_name`, backed by
the global, no-per-import-provenance `LOWER_IMPL_METHOD_HASH`) has no
equivalent asymmetric signal and can still select the unrelated struct's
method body for this identical call. Confirmed directly: the native-
compiled binary for this fixture segfaults at runtime even though `--check`
reported OK. A checker pass the compiler then crashes on is worse than a
checker refusal -- `--check OK` has to mean "this will run", not "this
typechecks under a guess the rest of the compiler cannot fully honor."

Closing this for real (accepting AND correctly running `zz::f()` in the
presence of the collision) needs the same shared "qualifier ->
target-module-id" resolution table this PR already documents as a genuine
architectural addition elsewhere (comment 4110379040 and others) -- there
is no locally-safe disambiguation available in the lowerer the way there
is in the checker (the checker's asymmetry works because a caller's own
`use` is a strictly-stronger signal than an incidental global type-table
hit; no such asymmetry exists between two equally-real global registry
entries in the lowerer). Until that table exists, the checker now fails
closed instead: when a qualifier is BOTH a real `use`d module AND collides
with an unrelated struct/enum that defines the same method name, the call
is refused outright (error E263) rather than resolved either way. This
gate asserts the refusal, not a pass -- see its own comment for the exact
E263 check. The common case (a used module with no colliding same-named
type+method) is unaffected and unambiguous; every other fixture in this PR
(`af_mod::AfNum::quad`, `pkg_mod::Widget::make`) already compiles and runs
correctly.

Note: the qualifier/type name is spelled `zz`, not a single letter --
single-letter lowercase names (e.g. `m`) hit an unrelated, pre-existing
checker quirk in this exact scenario (a spurious "no field n on type f64"
inside the method body) that is not part of what this fixture is pinning.
