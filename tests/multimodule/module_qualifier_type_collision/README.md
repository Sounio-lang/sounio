# A real `use`d module vs. an unrelated same-named lowercase type

Regression fixture for `checker_check_expr_call_inplace`'s `Type::method`
associated-function arm (`self-hosted/check/check.sio`). Driven by
`scripts/ci/madaros_module_qualifier_type_collision_gate.sh`, via `--check`.

| case    | what it pins |
|---------|--------------|
| `basic` | `main.sio` `use`s the real module `zz.sio` (exports `f() -> i64 { 42 }`) and ALSO declares its own unrelated lowercase `struct zz` with a method `f`. Before this fix, the associated-function lookup consulted the global struct/enum tables before ever checking whether the qualifier was a module the caller actually `use`d, so `zz::f()` always bound the struct's method -- with zero call arguments for a method that takes `self`, which typechecked the receiver as an inferred-default type rather than refusing the call, and then failed inside the method body with a spurious "no field n" error. `checker_use_module_known` is now consulted first, so the caller's own import wins and `--check` reports OK. |

**Why `--check` only, not compile-and-run:** this identical ambiguous shape
-- a real module and an unrelated impl method sharing one qualifier --
separately trips the STILL-OPEN gap this same review round raised in the
LOWERER (comment 4113834083, `self-hosted/ir/lower.sio`'s
`callee_path_module_stripped_name`): `LOWER_IMPL_METHOD_HASH` is global
with no per-import provenance either, so codegen for this exact call can
still target the wrong body even once the checker accepts it correctly
(confirmed directly: the native-compiled binary for this fixture segfaults
at runtime). Closing that needs the same shared "qualifier ->
target-module-id" resolution table this PR's other comments (4110379040
and others) already document as a genuine architectural addition -- there
is no locally-safe disambiguation available there the way there is in the
checker (the checker's fix works because a caller's own `use` is a
strictly-stronger, asymmetric signal than an incidental global type-table
hit; no such asymmetry exists between two equally-real global registry
entries in the lowerer). This fixture pins the checker's half of the fix
honestly, rather than claiming a full run-to-completion pass the compiler
cannot yet deliver for this exact shape.

Note: the qualifier/type name is spelled `zz`, not a single letter --
single-letter lowercase names (e.g. `m`) hit an unrelated, pre-existing
checker quirk in this exact scenario (a spurious "no field n on type f64"
inside the method body) that is not part of what this fixture is pinning.
