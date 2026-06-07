# Native-v2 Gate Recovery Audit (2026-06-07)

Branch: `claude/gate-recovery`
Base: `387c9e8ca`

Scope: recover gates that still depend on the retired
`self-hosted/compiler/native_compile_driver.sio` path. This audit did not edit
`self-hosted/**/*.sio` and does not wire anything into `release_gate.sh`.

Compiler used for probes:

```bash
ulimit -s 1048576
./bin/souc self-hosted/compiler/main.sio /tmp/gates_mc.elf
chmod +x /tmp/gates_mc.elf
```

The source-to-ELF path proven here is:

```bash
/tmp/gates_mc.elf --native-v2-compile <source.sio> -o <out.elf>
chmod +x <out.elf>
<out.elf>
```

## Recovered Gates

New honest gate:

```bash
scripts/ci/native_v2_recovered_source_core_gate.sh
```

Verified with:

```bash
SOUNIO_MODULAR_SOUC=/tmp/gates_mc.elf \
  scripts/ci/native_v2_recovered_source_core_gate.sh
```

Result: PASS.

Recovered coverage:

| Retired gate | Feature | Source used | Expected |
| --- | --- | --- | --- |
| `native_v2_serious_track_gate.sh` | hello source -> ELF, `.rodata`/print smoke | `examples/native/hello.sio` | stdout `Hello from self-hosted Sounio!\n42\n`, exit 0 |
| `native_v2_array_gate.sh` | array literal, indexing, sum | `tests/native-v2/recovered/array_min_exit45.sio` | exit 45 |
| `native_v2_logical_gate.sh` | `&&`, `||`, bool branch | `tests/native-v2/recovered/logical_min_exit7.sio` | exit 7 |
| `native_v2_enum_match_gate.sh` | enum construction + match | `tests/native-v2/recovered/enum_match_min_exit2.sio` | exit 2 |
| `native_v2_nested_field_gate.sh` | nested struct field access | `examples/native/nested_field.sio` | stdout `9\n`, exit 0 |
| `native_v2_struct_gate.sh` | struct suite core only, not full orchestrator | covered by the struct rows below | partial recovery |
| `native_v2_struct_mutation_gate.sh` | struct field mutation | `examples/native/struct_mutation.sio` | stdout `14\n`, exit 0 |
| `native_v2_struct_param_gate.sh` | struct parameter passing | `examples/native/struct_param.sio` | stdout `12\n`, exit 0 |
| `native_v2_struct_return_gate.sh` | struct return | `examples/native/struct_return.sio` | stdout `8\n`, exit 0 |

Notes:

- `examples/native/array_basics.sio`, `examples/native/logical_ops.sio`, and
  `examples/native/enum_match.sio` are not used because they fail the current
  source typecheck path. Minimal source fixtures preserve the same feature
  intent without relying on the retired driver.
- `native_v2_struct_gate.sh` was an orchestrator. The new gate recovers its
  core struct/array/logical/enum/nested coverage, but does not claim the full
  old orchestrated suite.

## Not Recovered

| Retired gate | Feature | Probe result / reason |
| --- | --- | --- |
| `native_v2_driver_self_compile_gate.sh` | retired driver self-compile and stage fixed point | Not recoverable by source gate without compiling `native_compile_driver.sio`; the feature itself is the retired path. |
| `native_v2_out_param_boundary_gate.sh` | `&!` out-param field store boundary, same-file and imported | Same-file fixture failed current typecheck under `--native-v2-compile`; imported case also depends on prebundle. |
| `native_v2_prebundle_gate.sh` | `native_prebundle.sio` bundle -> native-v2 compile -> run | `./bin/souc run self-hosted/compiler/native_prebundle.sio ...` failed with `error: no main`; no bundle was produced. |
| `native_v2_gum_primitives_gate.sh` | GUM primitive corpus | All four manifest cases segfaulted in `--native-v2-compile` after typecheck (`rc=139`, last line `Type check complete for module 0`). |
| `native_v2_epistemic_science_spine_gate.sh` | full epistemic/science manifest corpus | Mixed results: `hello`, `struct_return`, and `octonion_fano_168` passed, but several cases had stdout mismatch and GUM/PBPK cases segfaulted. Full gate not recovered. |
| `native_v2_dissertation_rapamycin_gate.sh` | rapamycin DES/GUM dissertation case | Not recovered; it depends on GUM primitives and the `rapamycin_des_gum` source path, which is in the failing science/GUM class. |

## Probe Summary

- Built `/tmp/gates_mc.elf` successfully from `self-hosted/compiler/main.sio`.
- `tests/native_v2_capgate/run.sh /tmp/gates_mc.elf` passed 32/32.
- `scripts/ci/native_v2_recovered_source_core_gate.sh` passed with
  `SOUNIO_MODULAR_SOUC=/tmp/gates_mc.elf`.
- No release-gate wiring was changed.
