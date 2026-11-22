# Scalar f64 expression-classification receipt

## Discriminating native evidence

The repair was tested as an isolated source overlay on
`c21b0e139830ca4949f80cba62887a8922a972d7`, not against a stale prebuilt compiler.
Slurm job **17265** executed on `gpuorangefs-r770-proxmox`.

| Artifact | SHA-256 |
|---|---|
| Rebuilt baseline Madaros | `a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3` |
| Rebuilt repaired Madaros | `b30def13149e1d95d07dde2cf107a82c7bc99e8ec8c910d8b7f6098f5ef8d2c9` |
| Repaired `self-hosted/ir/lower.sio` | `973a4230bc849d161cf501e2c4fb16ca6430cfddd6b26141d9de442623b6874c` |
| `tests/run-pass/f64_expression_class_audit.sio` | `11a484258bad9c19a44587185ff4eb6a36f87f8a2cef8722d2b6af768e4ef392` |
| `tests/reference/f64_expression_class_reference.cpp` | `31280b4542437854a1ce0c9f905e9acae003aca1779a5ce126afd2423468fc09` |

The regression passed check and compilation on the rebuilt baseline, then its
native executable exited **1** at the first match-ratio assertion. The repaired
compiler produced an executable that exited **0** and printed
`F64_EXPRESSION_CLASS_OK`. The canonical harness selected `engine=madaros` and
reported **Pass: 1, Fail: 0, Skip: 0**. The repaired lowering module also passed
`bin/souc check`.

The same repaired ELF passed these neighboring native regressions:

- `ard_ratio_large_main.sio`
- `f64_if_expr_merge_typing.sio`
- `f64_unary_neg_signed_zero_bits.sio`
- `regalloc_pressure_store.sio`
- `madaros_tuple_f64_array_from_if_and_match.sio`

## Independent C++23 reference

`f64_expression_class_reference.cpp` checks the declared binary-fraction cases,
integer controls and explicit casts independently of the Sounio compiler. Its
bytes match the source used by job 17265. It was compiled with
`g++ -std=c++23 -O2 -ffp-contract=off`, exited **0**, and printed
`CXX23_F64_REFERENCE_OK`.

## Reproduction

Run inside a disposable allocated Linux worker, not the shared development pod.
The private `artifacts/self-hosted/madaros` below must not replace another
agent's compiler artifact.

```sh
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
bash scripts/ci/build_modular_madaros.sh /tmp/madaros-f64-expression-current
mkdir -p artifacts/self-hosted
cp /tmp/madaros-f64-expression-current artifacts/self-hosted/madaros
export MADAROS_RAW_BIN="$PWD/artifacts/self-hosted/madaros"
bin/souc check self-hosted/ir/lower.sio
bin/souc check tests/run-pass/f64_expression_class_audit.sio
SOUNIO_MADAROS_AVAILABLE=1 SOUNIO_TEST_JOBS=1 \
  bash scripts/run_sio_test_suite.sh f64_expression_class_audit --verbose
g++ -std=c++23 -O2 -ffp-contract=off \
  tests/reference/f64_expression_class_reference.cpp -o /tmp/f64-expression-reference
/tmp/f64-expression-reference
```

For the failing side, build the baseline commit in a separate clean worker and
compile this same regression with that baseline ELF; do not overwrite the repair
or substitute the shipped prebuilt. Preserved full build/check/compile logs and
executables are under
`/orangefs/training/sounio-foundry/runs/f64-publication-20261025` on the worker
storage, with a custody copy in the isolated publication worktree.

## Claim boundary

This receipt covers the declared scalar-f64 branch, block-local alias, method,
cast, comparison, integer and shadowing cases. It is not a proof of whole-compiler
correctness, a complete expression-kind audit, full-corpus qualification, a
self-compilation fixed point, or wide-float parity. The temporary classifier's
4096-entry capacity is guarded fail-closed; its extreme boundary was not stress
qualified by this receipt. No scientific model, input data, IR-field meaning or
legacy compiler path was changed.
