# Same-named private functions across modules

Regression fixtures for `self-hosted/compiler/private_fn_identity.sio` (module-aware
identity for same-named **private** free functions). Driven by
`scripts/ci/madaros_private_fn_identity_gate.sh`; each case directory holds the modules,
one or more `main*.sio` roots, and `expected.txt` (the exact stdout of the compiled program).

Before the fix the merged IR identified functions by unqualified name, so two modules
that each defined a private `helper` shared one body: the first-loaded module won, every
caller ran it, and nothing was reported.

| case    | what it pins |
|---------|--------------|
| `basic` | the reported repro: `a=1 b=20`; `main_swapped.sio` swaps the `use` order and must not change the answer |
| `rich`  | 3 modules: colliding private `inner`/`helper`/`apply`/`fact`/`twice`, recursion, a fn passed as an argument and via `let f = helper`, an `impl` method calling a private fn, and a **pub** `helper` (imported by `main`) that must keep its own name |
| `skip`  | a module whose parameter is spelled like its private fn cannot be renamed safely; the pass must say so (`warning[private_fn_identity]`) rather than guess, while the module that *can* be renamed still is |
