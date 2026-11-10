# Block-local f64 tuple/reference classification receipt

## Defect and repair

An `if` whose two block tails read a local tuple's f64 slot or dereference a
local `&f64` produced the correct raw value while downstream arithmetic treated
its IEEE-754 payload as an integer. The reduced program printed value
`2.000000` and quotient `288230376151711744.000000` for division by `16.0`;
the independent reference yields `0.125`. Explicit result annotation `: f64`
made the controls pass.

`block_tail_is_float_ref` now captures tuple slot masks and reference pointee
identity before inserting its temporary binding, including aliases and
same-name shadowing. The tuple mask is written into that new slot directly:
the normal tuple binder creates a new locals Box, which would detach restoration
of count from the caller's shared temporary stack. This classifier emits no IR;
its capacity guard and count restoration remain.

## Reproducible native evidence

Base: `fa9d3be860aea57757776395d14fce44a9857483`.
Slurm **17694**, **cpuops-t560-proxmox**, built through
`bash scripts/ci/build_modular_madaros.sh` in a disposable worker tree.

| Artifact | SHA-256 |
|---|---|
| Rebuilt baseline Madaros | `8596dd2f2eeaf16042095e7b118cfbc6b39059a274c93fac5c473309a0c432f0` |
| Rebuilt repaired Madaros | `b3fed3f2cd5e6d5a003de7da576a8e7461e29ccd7a11e9d39f9ff6270be66a1e` |
| Repaired lower.sio | `33a17610531dd09430823e7b29761b10abb92a71e10a7211e97c15a3ce0d4170` |
| f64_block_metadata_audit.sio | `5e8a3758ba01f7e199fb7dd6134215f74197a032b11c77449b1131347233be43` |
| f64_block_metadata_reference.cpp | `6c0042b6281abd2ccdfe174c7aab6a7b5249bf73412a2762412fb5914f29bceb` |

The identical expanded regression passes check and compilation on the baseline,
then its native executable exits **1**. On the repaired compiler it passes
check/compile, exits **0**, and prints `F64_BLOCK_METADATA_AUDIT_OK`. Both
condition values exercise local tuples, references, aliases, same-name
shadowing, inferred/mutable references, match arms, integer controls and outer
binding preservation. The lowering module itself passed check.

The canonical harness selected `engine=madaros` and executed one nonempty
selection for each of `f64_block_metadata_audit` and the earlier
`f64_expression_class_audit` (#3133): each **Pass 1 / Fail 0 / Skip 0**.
The initial corpus enumeration is not a claim that the corpus was executed.

The same repaired ELF compiled and ran these neighboring tests, all exit 0:
`ard_ratio_large_main`, `f64_if_expr_merge_typing`,
`f64_unary_neg_signed_zero_bits`, `regalloc_pressure_store`,
`madaros_tuple_f64_array_from_if_and_match`, and `madaros_f64_deref_operand`.

The checked-in C++23 source is byte-identical to the independent source used
for the reference. It was compiled using
`g++ -std=c++23 -O2 -ffp-contract=off -static` and executed on the worker:
exit **0**, `CXX23_BLOCK_METADATA_REFERENCE_OK expected_ratio=0.125`.
It verifies the declared arithmetic values; the Sounio native tests additionally
exercise compiler metadata behavior. No Python/Rust numerical oracle was used.

Worker logs and artifacts are preserved at
`/orangefs/training/sounio-foundry/runs/f64-classification-repair-20261026`,
with a custody copy under `artifacts/f64-classification-repair-20261026/final/`
in the isolated publication worktree. `F64_METADATA_REPAIR_PASS` and artifact
readback are the accepted receipts; Slurm accounting was unavailable, so no
accounting result is claimed.

## Running the focused gate

Run inside a disposable allocated Linux worker. Install the rebuilt artifact
only in that private worker tree and select it explicitly:

```sh
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
bash scripts/ci/build_modular_madaros.sh /tmp/madaros-block-metadata
mkdir -p artifacts/self-hosted
cp /tmp/madaros-block-metadata artifacts/self-hosted/madaros
export MADAROS_RAW_BIN="$PWD/artifacts/self-hosted/madaros"
bin/souc check self-hosted/ir/lower.sio
SOUNIO_MADAROS_AVAILABLE=1 SOUNIO_TEST_JOBS=1 \
  bash scripts/run_sio_test_suite.sh f64_block_metadata_audit --verbose
SOUNIO_MADAROS_AVAILABLE=1 SOUNIO_TEST_JOBS=1 \
  bash scripts/run_sio_test_suite.sh f64_expression_class_audit --verbose
g++ -std=c++23 -O2 -ffp-contract=off \
  tests/reference/f64_block_metadata_reference.cpp -o /tmp/f64-block-reference
/tmp/f64-block-reference
```

For the failing side, use the unchanged base in a separate worker and compile
the same regression with its rebuilt ELF. Preserve the original fixture.

## Separate pre-existing refusal

**BLK-20261026-F64-NESTED-TUPLE-ALIAS**, severity **P2**, class
**compiler/lowering-name-resolution**, owner **codex-f64-repair**.
Slurm **17700** established executed check/compile evidence on both baseline
and repaired compilers: check 0, compile 1 with unresolved `t`. This shape
creates a tuple and aliases it inside the same arm:

```sounio
let result = if flag {
    let t = (7, 2.0)
    let u = t
    u.1
} else {
    let t = (9, 2.0)
    let u = t
    u.1
}
```

It is distinct from this repair's silent arithmetic misclassification. The
passing regression covers aliases from already-bound tuples and same-name
shadowing; it does not conceal this refusal. Worktree:
`/workspace/sounio-worktrees/f64-classification-20261026`, branch
`audit/f64-classification-20261026`. Acceptance gate: the minimal source checks,
compiles and executes correctly while the scalar-f64 gates remain passing.
Next action: trace the metadata/shape reader that reports the block-local alias
RHS unresolved and repair it in a separate focused change.

## Limits and review

This evidence is bounded to the declared scalar-f64 cases, not full-corpus,
fixed-point, wide-float, capacity-stress, closure/function-pointer or aggregate
lifetime qualification. No scientific model, data, IR field or legacy compiler
path was changed.

Darwin `sounio_verificado` was attempted for new test code (four candidates did
not compile; repository syntax supplied the correction). `coder-next` reviewed
the initial diff but supplied no accepted concrete finding; a `qwen38-27b`
request for the amended diff ended without a result. Independent approval is
not claimed. Direct source inspection found the locals-Box identity concern;
the corrected form is the one qualified by Slurm 17694.
