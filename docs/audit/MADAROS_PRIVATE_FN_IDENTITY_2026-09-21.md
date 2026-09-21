<!-- docs:meta
topic_id: repo.docs.audit.madaros-private-fn-identity-2026-09-21
authority: repo_only
audience: users
last_validated: 2026-09-21
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-private-fn-identity-2026-09-21
-->

# Madaros: same-named private functions in different modules

## Claim

> Under **current-source Madaros**, two modules that each define a private
> `fn helper` keep **separate bodies**. Each module's callers run that module's
> `helper`, in either import order. If it cannot tell them apart it **refuses to
> compile**; it never emits an executable in which they share one body.

Before this change the program compiled clean and printed the wrong answer.

## Symptom

```
hmod_a.sio   fn helper() -> i64 { 1 }    pub fn get_a() -> i64 { helper() }
hmod_b.sio   fn helper() -> i64 { 20 }   pub fn get_b() -> i64 { helper() }
main.sio     use hmod_a::{get_a}  use hmod_b::{get_b}   ... print get_a(), get_b()
```

| | before | after |
|---|---|---|
| as written | `a=1 b=1` | `a=1 b=20` |
| `use` lines swapped | `a=20 b=20` | `a=1 b=20` |

The winner was **import order**. The checker had accepted the program correctly
(`fn_sig_table_find_prefer_module`, see
[MADAROS_DUAL_GUM_KNOWLEDGE_IMPORT_2026-07-19.md](MADAROS_DUAL_GUM_KNOWLEDGE_IMPORT_2026-07-19.md));
lowering did not honour the same distinction.

## Root cause

The merged IR identifies a function by its **unqualified name**:
`lowerer_lookup_fn_id_by_name_ref` and `lowerer_find_or_add_fn_id_mut` are linear
`ir_name_eq` scans, and the dep merge, the DCE mark set and the FO transfer
registry are name-keyed as well. Two private `helper`s therefore share one slot;
the first-loaded body wins; every caller in every module runs it.

It was worse than the reported repro: the `rich` fixture showed a call from `main`
to a **pub** `helper` binding to another module's *private* one (`pub_helper=50`,
expected `5004`).

## Decision: module-aware identity (a), not detect-and-refuse (b)

A hard error was the obvious minimum, and it was measured and rejected:

* **The class is common in-tree.** A closure census over 1708 multi-module test
  roots found 77 (4.5%) with same-named private-involving fns whose bodies
  *differ* (`sqrt_f64`, `sin_approx`, `exp_approx`, `dual_*`, `near`, `chk`, ...).
  A default error would newly fail dozens of currently-passing roots.
* **Nothing exists to exempt benign copies.** There is no AST equality or
  fingerprint in the compiler, and the buffer printer is a roundtrip subset.
* **The bug is live in stdlib.** `chemistry/equilibrium.sio` and
  `chemistry/acids.sio` each define a private `ln_approx`; equilibrium's has no
  range reduction, acids' does. `acids::ph` silently ran the wrong one, so
  `tests/stdlib/chemistry/test_equilibrium_acids.sio` failed `ph` before this
  change and passes it after.

Doing (a) at the lowering funnels was rejected too: identity is name-keyed in
DCE, preseed, the dep merge and the FO registry, and there is a second pipeline
(the specialized-collapse path merges every module into one item list). Renaming
at the AST, before any of them run, needs no other consumer to learn about modules.

## The change

`self-hosted/compiler/private_fn_identity.sio`, called from the top of
`module_frontend_specialized_prepare` (both `compile` entry paths reach it).

```
private fn N defined (with a body) in >= 2 loaded modules
  -> in each module, in load order, that STILL shares the name with another
     module's definition when its turn comes:
       the definition and every reference to it in that module
       become  N__m<module index>

The check is exact and runs against the programs as they now are, so the rename
is minimal: once every other definition has been renamed away, the last one is
unique and keeps its bare name (2 colliding modules -> 1 rename).
```

* Only the **private** real definitions are renamed. A pub `N` keeps its name
  (importers spell it that way). `main`, `extern` declarations and GPU kernels are
  never touched.
* The mutation is **in place** through raw pointers to heap nodes (boxed `Expr`,
  boxed `FnDef`, `ItemList`/`ExprList` nodes). Nothing is written back into
  `programs[]`: module_frontend documents that nested field-in-array stores are
  dropped by lean_single and that a whole-`Program` writeback SEGV'd at `seed_begin`.
* It runs once per compile, from the two multi-module entry points, before the
  specializer merge, typecheck, DCE, FO preregistration and lowering, so both
  pipelines see unique names. It can **refuse** (see below); both callers stop.
* Function values (`apply(helper, x)`, `let f = helper`), recursion, and `impl` /
  `trait` method bodies that call a private fn are rewritten.
* `SOUNIO_DISABLE_PRIVATE_FN_IDENTITY=1` skips the pass (attribution knob, like
  `SOUNIO_DISABLE_MM_DCE`).

### Identity is exact, and the census is of every emitted symbol

Review of the first version (PR #2598) found four ways the pass could still bind
two functions to one body. All four were reproduced on the first version before
being fixed, and each now has a fixture:

| finding | reproduced as | fix |
|---|---|---|
| names were matched by `(ast_name_hash, len)`; djb2 is not collision-free (`ab` and `bA` collide) | `hashcoll`: compile error E137 -- a reference to `bA` was rewritten to the non-existent `bA__m1` | the per-module table stores and compares the **full name bytes**; the hash is only a prefilter, and a candidate is renamed only after an exact scan finds the same name in another module |
| the census counted only real fns, so a private fn could collide with an extern / kernel / global / `Type_method` symbol, and a generated `name__m<N>` could reuse one | `reserved`: `method=1`, baseline `780` -- the generated name clobbered a method symbol. `symcoll`: garbage result | the census covers **every** symbol lowering puts in its one name-keyed table (all top-level fn items, BSS globals, and `Type_method` for every impl method); it decides both "another module defines this" and "this generated name is free" |
| a generated name could reuse a module **global** (a global is an `ItemFn` with no body) | `reservedglobal`: `global=4198400`, baseline `41` | covered by the same complete census, which counts every `ItemFn` regardless of `fn_def` |
| the per-module table held 64 names and silently dropped the rest | `capacity`: `b=66415`, want `72415` -- exactly the 6 names past 64 were left colliding | table 512, census 65536; exhausting either **aborts the compile** |
| a name skipped as unsafe only produced a warning, so if every colliding module skipped, the executable was wrong | `unresolved`: `7`, correct `26` | after all modules are processed, each skipped name is re-checked (exactly) against the other modules; if another module still defines the symbol the compile is **refused** with `error[private_fn_identity]`, no ELF written |

### What it refuses to guess

A name can be left unrenamed for one of three reasons, recorded per entry so a
refusal states the real one:

| reason | when |
|---|---|
| shadowed | a local, parameter, pattern, `for` or closure binding is spelled like the fn, or the fn is used as a value in a match-arm body / struct-literal field (stored by value in the list node, not behind a `Box`, so it cannot be rewritten in place) |
| generated name | the `name__m<N>` it would get is already an emitted symbol, or would not fit |
| exported | the fn is `pub(crate)` / `pub(super)` / `pub(in ..)`. These are exported exactly like `pub`: importers name them, so renaming would break every importer, and this pass does not rewrite importers |

Skipping is accepted **only if no collision remains**. When every other definition
of the name was renamed away, it is unique again and is merely noted
(`private_fn_identity: left N fn(s) unrenamed; no collision remains`). Otherwise
the compile is refused, with the reason that actually applies:

```
error[private_fn_identity]: fn `helper` (module #1) shares its name with a
  definition in module #2 and cannot be renamed safely:
  it is exported (pub(crate) / pub(super) / pub(in ..)), so other modules
  refer to it by this name and renaming it would break them.
  Both would compile to ONE function. Rename one of them.
```

Restricted-public fns are tracked, not ignored: a `pub(crate)` `helper` next to a
private `helper` is fine (the private one is renamed away and the exported one is
left unique), while two `pub(crate)` `helper`s, or one next to a plain `pub` one,
are refused. In-tree this refuses nothing: the corpus has 247 restricted-public
fns, none sharing a name with another module's fn, across 1718 roots and the
compiler's own closure.

## Evidence

Gate: `scripts/ci/madaros_private_fn_identity_gate.sh`, fixtures in
`tests/multimodule/private_fn_identity/{basic,rich,skip}`, wired into
`.github/workflows/ci.yml` after the scalar-multi-instance step (shared ELF via
`MADAROS_RAW_BIN`).

| check | baseline | fixed |
|---|---|---|
| repro, both import orders | `a=1 b=1` / `a=20 b=20` | `a=1 b=20` both |
| `rich` (3 modules, recursion, fn values, impl method, pub/private mix) | 5 of 11 lines wrong | all 11 exact |
| `skip` (parameter shadows the fn in one module) | `sa=1 sb=1` | `sa=1 sb=20`, noted, no error |
| `unresolved` (shadowed in every colliding module) | compiles, prints `7` (correct `26`) | **refused**, `error[private_fn_identity]`, no ELF |
| `hashcoll`, `symcoll`, `reserved`, `reservedglobal`, `capacity` | wrong / garbage | exact |
| `restricted` (two `pub(crate)` fns of one name) | compiles, one body for both | **refused**, reason "exported" |
| `restrictedmix` (`pub(crate)` met first + a private one) | wrong | `a=1 b=20` |
| capacity boundary, generated by the gate: 512 names / 513 names | (the first version silently dropped every name past 64) | 512 accepted and all renamed; 513 **refused**, no ELF |
| gate | **FAIL** at `basic` with a diff | **PASS** (incl. off-switch control) |
| #854 `duplicate_private_single_main` | exit 12 | `PASS`, exit 0 |
| #854 `duplicate_private_18_main` | exit 23 | `PASS`, exit 0 |

The gate's control compiles the repro with the pass disabled and requires the
answer to be *wrong*, so a green run cannot be a program that never collided.

Corpus comparison (baseline vs fixed, compile + run, 27 roots, one per distinct
collision signature): 24 identical, 3 different -- the two #854 fixtures and
`test_equilibrium_acids` (`ph` fixed, above). Six roots do not compile on either
compiler (unrelated E035 / parse errors) and prove nothing either way.

### Madaros compiling Madaros

`scripts/ci/madaros_fixed_point_gate.sh`, run with CI's own settings
(`SOUNIO_MADAROS_FP_MIN_INTO_ACC_DONE=122`, default expected rung `run`) against a
Madaros built from this tree:

```
[rung check] rc=0 errors=0
[rung gen2]  rc=0 merged_ir_functions=13545   into_acc_done=126 (minimum 122)
[rung run]   gen2 identifies itself: banner=madaros
[rung gen3]  rc=139   (gen2 compiling main.sio: SIGSEGV)
MADAROS_FIXED_POINT_OK: reached rung 'run' as recorded
```

The pass ran on the compiler's own 126-module source and renamed the 4 colliding
private fns the census predicted (`name_is_dynlink_extern` in two modules,
`ir_call_sret` / `ir_return_sret` in `compiler/main.sio`). gen2 built and answers
`--version` as Madaros. The gate is a ratchet -- green only at exactly the recorded
rung -- and `run` is the rung recorded since 2026-09-06; `gen3` is its named next
wall, so the `gen3` SIGSEGV is the recorded state and not a new failure.

What this does **not** show: I did not run the same gate against the baseline
compiler, so "`gen3` crashes the same way before this change" rests on the gate's
recorded ratchet, not on a measurement of mine. Separately,
`spec_dce_mm: REFUSING to filter (marks 8192 of 8192)` appears in the self-compile,
and it appears identically when the *baseline* compiler compiles the *baseline* tree
(measured), so it is pre-existing and not caused by the extra names this pass adds.

## Known cost

Mangled names appear in diagnostics: an E035 on a colliding fn now names
`char_from_i64__m1` where it used to name `char_from_i64`. Stripping the
`__m<N>` suffix when a diagnostic is printed is a small follow-up.

## claims_not_made

* **Parse-time constant folding is still name-keyed and is NOT fixed.** The parser
  folds pure-fn calls in global initialisers through `GLOBAL_VAR_INIT_*`, a table
  keyed by the *bare* name that is deliberately accumulated across modules. A
  global initialised from a colliding private pure fn can therefore bake the wrong
  module's value: with `pfi_a::helper() = 1`, `pfi_b::helper() = 20` and
  `var G: i64 = helper()` in `pfi_b`, `G` reads `1` (want `20`) on the baseline
  **and** with this change. It happens at parse time, before any AST pass runs, and
  the same table holds same-named private *globals*, so it belongs with the
  "same-named private globals" item below and needs the parser's tables to become
  module-aware.

* **Same-named private globals** (`var` / `const`, BSS slots keyed by name) and
  **same-named types** (`struct_layouts`, `enum_variants`) are the same class and
  are **not covered**.
* **Two plain `pub` fns of the same name** in different modules are not renamed and
  not diagnosed here (a `pub` fn is the module's interface; renaming it needs every
  importer rewritten). Restricted-public (`pub(crate)` etc.) collisions **are**
  refused when unresolved, as described above.
* `madaros check` is unchanged: the checker already separated these; only the
  compile path renames.
* The 81-root in-tree set was sampled (27 roots), not run exhaustively; the full
  suite is CI's job, and it was red on main for unrelated reasons before this change.
* Extern declarations and GPU kernels are counted as symbols a private fn can
  collide with, and their names are never reused by a generated name, but no
  fixture exercises an `extern` collision directly (its lowering has builtin-dispatch
  special cases); the method-symbol fixtures cover the same census path.
* Not a fix for the lowering funnels themselves: identity there is still the bare
  name. This removes the *ambiguity* before it reaches them.
