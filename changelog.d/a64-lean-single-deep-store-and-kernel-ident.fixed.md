The aarch64 lean_single seed now parses the deep assignment shapes and the
`kernel` identifier that it previously rejected with a spurious `parse error:
expected expression … found `=``, which — since #3707 made errors in imported
modules fatal — turned the macOS self-host job red on main (#3679).

Two gaps in `compile_stmt_a64_body`, both a missing a64 twin of an x86 handler:

- **Deep field stores.** `(*name.field…)<chain> = v`, `(*name).arr[i].field… = v`
  and the auto-deref `name.arr[i].field… = v` had no a64 handler, so they fell to
  the generic path and were reported as parse errors. Ported
  `compile_deref_indexed_elem_field_store_x86` to `…_a64` (mirror of the x86
  emitter) and dispatched the three x86 predicates
  (`stmt_is_deref_indexed_elem_field_store`, `stmt_is_ident_indexed_elem_field_store`,
  `stmt_is_deref_inner_root_store`) on a64.
- **The identifier `kernel`.** `kernel` lexes as TK=79 (the `kernel fn` GPU
  keyword), not TK=3. The a64 one-hop field store (`name.field = v`) and bare
  variable assignment (`name = v`) guards tested only `k == 3`, so any local
  named `kernel` — e.g. `hlir_to_gpu.sio`'s `kernel.name = …` — was not
  recognized. Both guards now match the x86 twins' `k == 3 || k == 79`.

Verified by cross-building the aarch64 seed from this source and compiling
`self-hosted/compiler/main.sio` with it: 245 `parse error:` lines before, 0
after, ELF produced. `scripts/ci/build_madaros_macos_arm64.sh` now also fails
loudly on any `error:`/`parse error:` line in the seed's build log.
