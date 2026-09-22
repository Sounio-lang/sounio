<!-- docs:meta
topic_id: repo.docs.audit.madaros-global-init-module-identity-2026-09-21
authority: repo_only
audience: users
last_validated: 2026-09-21
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-global-init-module-identity-2026-09-21
-->

# Madaros: same-named definitions in different modules vs. the global-initialiser fold table

## Claim

> Under **current-source Madaros**, a global initialiser that calls a pure fn or
> names another global folds with **its own module's** definition. A same-named
> fn or global in another module is never used in its place, in either import
> order; when the name is genuinely ambiguous the compiler says so.

This is the parse-time half of
[MADAROS_PRIVATE_FN_IDENTITY_2026-09-21.md](MADAROS_PRIVATE_FN_IDENTITY_2026-09-21.md),
which renames private fns in the AST *after* parsing and therefore cannot reach it.

## Symptom

```
pfi_a.sio   fn helper() -> i64 { 1 }
            pub fn get_a() -> i64 { helper() }
pfi_b.sio   fn helper() -> i64 { 20 }
            var G: i64 = helper()
            pub fn get_g() -> i64 { G }
main.sio    prints get_a(), get_b(), get_g()
```

`get_g()` printed `1` (expected `20`) on origin/main's compiler **and** on the private-fn-identity
PR's compiler; the PR fixed `get_b()` only.

## Root cause

The parser folds `var G = helper()`, `let H = G + 1` and `[scale(1), 20, 30]` at parse
time through `GLOBAL_VAR_INIT_*` (`parser/ast.sio`). The table is keyed by the **bare
name** (`ast_name_hash`) and deliberately accumulates across modules
(`module_frontend.sio`, `GLOBAL_VAR_INIT_SUPPRESS_RESET`) so imported f64/i64 BSS
constants survive to lower time. Four consequences:

1. `pure_fn_reg_record` / `items_maybe_record_pure_fn_const` skip a fn when "any init
   words already exist under this name" -- so the *second* module's `helper` was never
   recorded, and the first module's body answered every later lookup.
2. A global recorded by two modules under one name had `count == 2`, which the scalar
   fold (`count == 1`) refuses, so `var H = G + 1` stayed zero.
3. At lower time the same two words became one 2-word **element list** for a scalar
   global; the second word was stored past the first global's 8-byte slot, into its
   **neighbour** (`S1 = LIM + 100` read back `4`, LIM's second word, in one import
   order and `3` in the other).
4. Impl methods were recorded under their bare name too, so a `Type::new` could stand
   in for -- or block -- a free fn `new`.

## The change

* `parser/ast.sio`: every table word carries `MODS[i]` (serial of the module that recorded
  it: load order, main = 0) and `PUBS[i]` (may another module name this definition:
  every global, and a `pub` free fn; not a private fn). `ast_global_var_init_begin_module()`
  hands out the serial; `parse_items_preloaded` calls it only while the table accumulates,
  so a single-module compile keeps serial 0 and is unchanged.
* Parse-time lookups go through `ast_gvi_scope(name)`, which picks ONE module:
  1. the module doing the lookup, if it recorded `name`;
  2. while folding the body of a pure fn that belongs to *another* module, that module
     only (its other names live there);
  3. otherwise the single other module with an importable record -- unless the module being
     parsed defines `name` itself (a token pre-scan of its top-level `fn`/`var`/`let`/`const`
     names, so a later or non-foldable own definition also vetoes the fallback);
  4. two or more candidates: **ambiguous** -- not folded, counted, and reported as
     `warning[global_init_ambiguous]` when the module finishes parsing.
* The name-only readers `lower.sio` uses (`ast_global_var_init_count` / `_nth`) read the
  first module that recorded the name. With one defining module that is the whole table,
  as before; with two it no longer concatenates them.
* Impl methods are no longer recorded (a bare `name()` cannot reach one).
* `SOUNIO_DISABLE_GLOBAL_INIT_MODULE_SCOPE=1` keeps every module at serial 0 -- the old
  bare-name table, bit for bit -- as an attribution knob and as the gate's control.

## claims_not_made

* **Same-named private globals still share ONE BSS slot.** The IR keys global slots by
  name, so `var G = 5` in `a` and `var G = 7` in `b` read the same memory. This change makes
  each module's *initialiser folds* right and stops the neighbour clobber; it does not
  give the two `G`s separate storage. That needs a rename pass for globals (the module tag on
  the table words is what such a pass would re-key). No fixture reads a colliding global.
* **Ambiguity is fail-closed, not resolved.** Two other modules defining the same
  importable name, with the importer naming one of them by `use`, is reported and left zero
  rather than resolved by import path.
* **Unchanged limitations:** initialisers still fold only what was recorded earlier in
  load order (`main` is parsed first, so it cannot fold an import; a forward reference to a
  later fn is zero); nested calls inside a pure-fn body do not fold, in one module or many.
* The table cap (8192 words) is unchanged and the compiler's own closure sits close to it
  (7956 words before, 7823 after).
