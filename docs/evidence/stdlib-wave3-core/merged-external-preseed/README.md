# Merged external function preseed: exact-source component

The production change suppresses external free-function and impl-method
signature registration when the lowerer receives merged items. Those merged
items already carry the live function bodies. External struct and enum metadata
still registers, and ordinary nonmerged registration keeps function return and
receiver metadata.

The lane tested the complete root `lower.sio` and `ir.sio` files in isolated
snapshots. The before variant removes exactly the two signature guards from the
after file. Both variants retain the common compatibility corrections:
`Alloc` on `user_enum_call_index_ref` and `payload_fields: None` in
`empty_enum_variant_entry`. Production files are pinned by
[QUALIFIED_PRODUCTION.sha256](QUALIFIED_PRODUCTION.sha256); they are not copied
into this evidence directory.

The fixture uses the real `load_module_file` parser. It contains one struct
with an f64 field, a two-variant enum, a public free function returning f64, and
an impl method taking a shared receiver and returning f64. The temporary probe
calls the actual private `lowerer_preseed_external_struct_items_mut`; the probe
is appended only to the snapshot. Prefix comparisons verify the complete
production file, and source guards cover all compiler modules and stdlib inputs.

## Executed results

| Slurm job | Result | Scope |
| --- | --- | --- |
| 18899 | Wrapper exit 1, raw timeout 124 | Preserved original source; E035 missing Alloc and E046 missing initializer field were printed before the 300-second wrapper deadline. |
| 18902 | Wrapper exit 1, raw timeout 124 | Corrected exact source, still limited by the wrapper's 300-second native deadline. |
| 18904 | COMPLETED, exit 0, 2m00 | Legacy native component, paired before/after with the same fixture and probe. |
| 18905 | COMPLETED, exit 0, 6m41 | Raw Madaros e9 checked the exact 28-module closure: verdict 0, check OK; all source/compiler guards passed. |
| 18911 | COMPLETED, exit 0, 2m01 | Executed the versioned portable gate; repeated the paired native result and all guards. |

The timeout attempts are not classified as conclusive semantic refusals.
The source diagnostics from 18899 were repaired before the successful pair.

[Native receipts](native18904/RESULTS.tsv) demonstrate:

| Property | Before | After |
| --- | --- | --- |
| Compilation | 0 | 0 |
| Native execution | 1, required assertion failure | 0 |
| Merged function count | 2 | 0 |
| Merged struct layout and f64 field | Preserved | Preserved |
| Merged enum variants | 2 | 2 |
| Nonmerged functions | 2 | 2 |
| Nonmerged f64 return signatures | 2 | 2 |
| Nonmerged free function / impl method | 1 / 1 | 1 / 1 |
| Nonmerged method shared-receiver metadata | 1 | 1 |

All metadata assertions precede the final merged-function assertion.
The native after run prints `PRESEED_COMPONENT_OK`.
The complete source manifests of the native and checked compiler closures are
identical, SHA256
`addd713aa6bf5adc910269d06f9a96364163cbd5fe1795cae750273fe6fcecee`.
The comparison covers self-hosted sources, stdlib, fixture and driver; the
wrapper's derived verification stamp is not a compiled input.

The native compiler is the legacy positional seed used by job 18820:
`9cfb6a970d73718b09091331dc514d015e3d4e6f402ee756cdc7f7b0cc55110a`.
The independent type/effect check uses raw Madaros:
`e9ad6cde0328e1dbf249da726fb44a798b0dd575236d5ec4040cd309f558b03e`.
No type/effect check was disabled. Its full log is retained at
[types18905/check.stdout](types18905/check.stdout).

The portable gate rerun uses the identical checked source-manifest hash shown
above. Its receipts and script/fixture identities are retained in
[portable18911/RESULTS.tsv](portable18911/RESULTS.tsv) and
[QUALIFIED_GATE.sha256](QUALIFIED_GATE.sha256).

The original component ELF hashes are recorded in
[native18904/EXECUTABLES.sha256](native18904/EXECUTABLES.sha256). Source/compiler
guards and production-prefix comparisons passed for both variants.

## Reproduction and integration boundary

The reusable gate is
`scripts/stdlib/merged_external_preseed_component_gate.sh`.
Run it under Slurm with four arguments: a complete source snapshot, the after
`lower.sio`, the common `ir.sio`, and a fresh absolute output directory outside
the input snapshot. Pin `PRESEED_SEED_BIN`, `PRESEED_SEED_SHA256`,
`PRESEED_AFTER_SHA256` and `PRESEED_IR_SHA256`.
It constructs the before variant, appends the same probe to each isolated
snapshot, and requires the original failure, successful after run, nonmerged
metadata controls, compiler/source guards and unchanged production inputs.
Its success marker identifies native component acceptance. Type checking is a
separate receipt, as executed by job 18905.

Templates have the `.sio.in` suffix because they are test inputs assembled by
the gate, rather than standalone entry points.

Durable custody is
`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/merged-preseed01`:

| Artifact | SHA256 |
| --- | --- |
| component-source-closure.tar.gz | 4da2ec00d81707e7976fe8beb875b6fcd107cdd773f00285e082ed19c50e47e6 |
| native-component-and-failures.tar.gz | 65ed254e4fc647f67d112b5911d1b3702e89cee3028062b9fec6e4911890f5be |
| typecheck18905.tar.gz | fa6e7d3db72d9a138a3a2bfcc64629e1c92354a032f46214588d56a11a76deb0 |
| portable-gate18911.tar.gz | d0082c0d9988e4c5406f7d314eea2ae039fd48398317d70399b98e642021ff9f |

The source closure includes the pinned legacy seed. Original failures, full
source manifests, production captures, both ELF files, test wrapper and guards
are retained. The original uncorrected ir.sio is retained separately with SHA256
`065a8c7623a5fd068cd8f5ee137f8574d309cfbc99470b3ffd2d1727a8ff194a`.

These jobs qualify the source-level compiler component and its type/effect
closure. A rebuilt current raw CLI, the unchanged owner_views and fully
qualified consumer witnesses, and collection ownership/storage gates remain
separate integration work.

