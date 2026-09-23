# Same-named definitions across modules vs. the global-initialiser side table

Regression fixtures for the module-aware `GLOBAL_VAR_INIT_*` table
(`self-hosted/parser/ast.sio`: `ast_gvi_scope`, `ast_gvi_count_in`, ...; consumers in
`self-hosted/parser/items.sio`). Driven by `scripts/ci/madaros_global_init_identity_gate.sh`;
each case directory holds the modules, one or more `main*.sio` roots and `expected.txt`
(the exact stdout of the compiled program).

The parser folds pure-fn calls and global references in *global initialisers*
(`var G: i64 = helper()`, `let H = G + 1`, `[scale(1), 20, 30]`) at parse time, through a
side table that is keyed by the BARE name and deliberately accumulates across modules so
imported constants survive to lower time. Before the fix a second module's `fn helper` was
never recorded ("already has init words"), so `var G = helper()` in that module folded the
FIRST module's body. That is a different table from the one `private_fn_identity.sio`
renames after parsing, which is why the private-fn identity pass could not reach it.

Every table word now carries the serial of the module that recorded it (load order, main = 0)
and whether another module may name the definition. Parse-time lookups resolve a name in
the module doing the lookup first; otherwise in the single other module that defines it
importably, unless the module doing the lookup defines it itself; two or more candidates is
*ambiguous* and is reported (`warning[global_init_ambiguous]`) and left unfolded.
Lower time (which has no "current module") reads the words of the first module that recorded
the name -- before, a global recorded by two modules became one garbled element list whose
extra words were stored into the NEXT global's slot.

| case        | what it pins |
|-------------|--------------|
| `basic`     | the reported repro: `a=1 b=20 g=20`; `main_swapped.sio` swaps the `use` order |
| `fold`      | same-named private **paramful** fns (binop-of-params and `let`-chain bodies) and an element-list global `[scale(1), one(), 30]` fold with their own module's body |
| `globals`   | same-named private **globals**: `var H = G + 1` folds with its own module's `G` (the two `G` slots themselves still share one BSS slot -- a separate, uncovered defect -- so no fixture reads `G`) |
| `import`    | control for the fallback: a module still folds another module's *importable* fn / global (`base() + 1`, `LIMIT * 2`), evaluates that fn's body in ITS module (`f(1)` sees the lib's `K`, not the caller's own `K`), prefers its own `g` over the lib's, and an imported f64 constant still reaches the BSS slot |
| `shadow`    | a module that defines `helper` itself (later in the file, or effectful) must NOT adopt another module's importable `helper` |
| `ambiguous` | two modules define the same global, a third names it: reported, not folded |

Controls (`SOUNIO_DISABLE_GLOBAL_INIT_MODULE_SCOPE=1`, which restores the previous
bare-name *resolution*) must reproduce the defect on `basic`, `fold`, `globals`, `import`
and `shadow`, and must not print the ambiguity warning -- so a green run cannot be a
program that never collided. The switch does not restore the table's contents to
origin/main: the independent exclusion of impl methods from the pure-fn constant-fold
table remains active under it, and is tracked separately. None of the fixtures above uses
an impl method, so the positive control is unaffected.
