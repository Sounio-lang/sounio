The committed Madaros prebuilt `bin/madaros-linux-x86_64` is rebuilt from
current source. The one it replaces was built from `0ed977cec` and miscompiled
`self-hosted/compiler/lean_single.sio`: `write_file` on a packed global
`[i8; N]` read every 8th byte, 32 bytes late (#2966, fixed in source by
`f078496f87`), and `0xc3 as i8` was 195 instead of -61 (fixed by `4066c302f5`).
Built by the new prebuilt, lean_single.sio now rebuilds the committed seed
`bin/souc-lean-single-x86_64` byte for byte.
`scripts/ci/madaros_prebuilt_agrees_with_seed_gate.sh` checks exactly that on
every PR (`MADAROS_PREBUILT_AGREES_WITH_SEED_OK`), after two fixtures in
`tests/compiler/madaros_prebuilt_agrees_with_seed/` that discriminate the two
defects.

Shipping current source surfaced what the old prebuilt hid:

- `--check` no longer dies with SIGSEGV after the diagnostic of an epistemic
  builtin that is checked through the by-value bridge (`plan_acquisition`,
  `propose_alternatives`, `transition_reason` and the rest). The seed sized the
  receiver `result.0` in `check_expr`'s `(result.0.exit_expr_depth(), result.1)`
  at 128 slots instead of Checker's 22955, so every by-value `check_expr`
  returned a truncated Checker. `result` is now annotated
  `(Checker, TypeEntry)`. All 39 fixtures pinned in
  `tests/selfhost/madaros_epistemic_compile_fail_crashers.txt` refuse cleanly,
  and the 12 crashes of the TypeKind family B ladder are gone.
- The xorshift generators of the hydrogen demos and the bitset loops of the
  EL+ ontology drivers use `wrapping_add`/`wrapping_sub`/`wrapping_mul`, as
  ADR-031 R1 requires. Under current Madaros the plain operators trapped; the
  bits are unchanged.

Rebuilt again after #2720, so the shipped prebuilt refuses an entry file with
no `main`, or with only a bodyless `extern "C" fn main()`, with
`error[E221]: no main` and writes no ELF. The prebuilt built before #2720
wrote one that died with SIGILL. The seed-agreement gate now checks both
refusals and a positive control.
