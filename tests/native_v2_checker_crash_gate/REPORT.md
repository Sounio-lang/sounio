# Single-Module Checker Crash Campaign Report

Worktree: `/workspace/sounio-checker`
Branch: `claude/checker-singlemodule-crashes`
Base requested by prompt: `2875ba291`

## Corpus

The campaign enumerated 20 valid single-module probes plus 3 invalid controls.
The active gate keeps the probes that are currently accepted by `--check` under
`valid/`; unresolved but valuable repros are retained under `deferred/`.

Active valid probes:

- `01_closure_basic.sio`
- `03_closure_return_block.sio`
- `07_tuple_match.sio`
- `08_option_tuple_match.sio`
- `11_large_struct_return.sio`
- `12_large_struct_nested_return.sio`
- `19_enum_struct_variant_match.sio`
- `20_array_struct_return.sio`
- `21_binary_arith.sio`
- `22_struct_field_sum.sio`

Invalid controls:

- `01_return_mismatch.sio`
- `02_call_too_few_args.sio`
- `03_closure_arg_type.sio`

Deferred valid repros are listed in `deferred/README.md`.

## Fixed Classes

### Binary expression SRET crash

Minimal repros:

- `valid/21_binary_arith.sio`
- `valid/22_struct_field_sum.sio`
- existing coverage: `tests/native_v2_capgate/03_arith.sio`,
  `tests/native_v2_capgate/06_let.sio`,
  `tests/native_v2_capgate/14_struct.sio`

Before the fix, these crashed in `mc --check` after entering the
single-module checker item pass.

Root cause:

- `self-hosted/check/check.sio:2892-3017`
- `checker_check_binary_with_operand_types_inplace` still called by-value
  checker methods for operator typing and unit propagation. Those methods
  return/copy `Checker` on the `--check` hot path even though operands were
  already checked in place.

Fix:

- Added `checker_check_binary_op_types_inplace`.
- Added `checker_check_binary_units_inplace`, with a no-unit fast path that
  returns `result` directly.
- Routed the binary tail through these helpers.

### Impl item by-value checker crash

Minimal repros:

- `deferred/13_method_value_receiver.sio`
- `deferred/14_method_ref_receiver.sio`

Before the fix, these SIGSEGVed during `--check`.

Root cause:

- `self-hosted/check/check.sio:2458-2529`
- `checker_check_item_inplace` reached a by-value `(*c).check_impl_item`
  bridge for `ItemImpl`, reintroducing the large `Checker` return/copy path.

Fix:

- Added `checker_check_impl_item_inplace`.
- Added `checker_check_impl_methods_inplace`.
- Added `checker_check_impl_method_inplace`.
- Routed `ItemKind::ItemImpl` through the in-place implementation.

Post-fix status:

- The method repros no longer SIGSEGV.
- They still fail cleanly with false E011, so they remain deferred rather than
  active green gate cases.

## Deferred Classes

These remain intentionally out of the active green gate:

- closure capture with inferred params: false E004/E009
- higher-order function parameters: rejected by checker
- tuple return types: rejected by checker
- nested `if let`: false E006
- `while let Option`: still SIGSEGV
- method lookup: false E011 after crash removal
- generic functions/structs: false E008/E009/E004
- `Option<Box<T>>` match/deref: false E005

The golden rule was preserved: no deferred case was made accepted merely to
avoid a crash.

## Verification

Built the candidate compiler:

```sh
ulimit -s 1048576
./bin/souc self-hosted/compiler/main.sio /tmp/mc_chk.elf
chmod +x /tmp/mc_chk.elf
```

Required gates:

```text
bash tests/native_v2_checker_crash_gate/run.sh /tmp/mc_chk.elf -> 13/13
bash tests/native_v2_capgate/run.sh /tmp/mc_chk.elf -> 32/32
bash tests/native_v2_soundness_gate/run.sh /tmp/mc_chk.elf -> 7/7
bash tests/native_v2_enum_gate/run.sh /tmp/mc_chk.elf -> 15/15
bash scripts/ci/native_v2_e2e_codegen_suite_gate.sh -> PASS, 37/37-style suite
bash tests/native_v2_backend_soundness_gate/run.sh /tmp/mc_chk.elf -> 40/40,
  UNRESOLVED SILENT MISCOMPILES: 1, rc=1 expected
```

The literal command `./bin/souc self-hosted/compiler/main.sio` returns usage
with rc=1 in this worktree because the bootstrap requires an output path. The
actual self-build form with `/tmp/mc_chk.elf` completed rc=0.
