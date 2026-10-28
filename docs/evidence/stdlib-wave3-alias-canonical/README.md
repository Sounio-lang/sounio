# Checked concrete aliases: component qualification

The original `aliases_fields.sio` failure is closed **in the actual native
compiler component**. The normalizer resolves checked aliases to a concrete
nominal declaration before lowering consumes field/local/parameter metadata.
It does not relax the empty-stub guard.

The original fixture bytes remain
`0781ca5cfed395df4ff5b13df0314abbc4575c22088dec349e42bdaa4f19a990`.
Raw CLI `11332ed8a6cf7ccb85f0a31bf63bed1fcc10659054d68639d6dba233a6014ff4`
checks that original path with status 0, then refuses compile with status 1,
`name=i64_get` and no ELF. The direct native emitter returns its own refusal
code 12 before the fix. After canonicalization, the same unchanged program
emits and exits 0, checking its original result of 18.

## Exact implementation boundary

The qualified production files for this commit are:

- `self-hosted/check/specializer.sio`:
  `2d2687366ac5ee24dc2534812cb673a93778ce36febd3ecf9ef2781c55db37ff`.
- `self-hosted/compiler/module_frontend.sio`:
  `84f5faa7bf16d3df7cc1a39b233ab33f8c16342eb781b4e5e311ce37c73bcd97`.

The source baseline is `792a602a04715bb43b47b541e670f26998f43603`.
The frontend invokes the new pass only after the authoritative checker verdict
is zero. It updates the existing metadata-bearing AST boxes and the fresh
specialized list, without writing a large Program back through an array.
After a change it discards and recollects FO/use/tuple-array preregistration
by the original module owner before preseed/body lowering.

Resolution follows the existing physical owner/import/declaration map, checks
the alias chain and requires a unique emitted concrete nongeneric struct.
Unknown, cyclic, ambiguous, generic/applied, enum or reference-alias targets
do not acquire a guessed identity. Outer references, type arguments, source
spans and Item identities remain intact. This pass does not choose a receiver
from field count, representation size or a fixture name.

The faithful substitution visitor has a distinct post-check context that
does not instantiate generics or rewrite call targets. Generic struct
templates are skipped. Single uppercase parameter spellings in function and
impl scopes are conservatively protected, so not every one-character alias is
claimed supported in an ambiguous generic scope.

A draft had failed this last requirement: a global `type T=Word` could capture
the `T` in an impl target. Slurm 19441 checks the actual AST and checker:
the predecessor changes the placeholder count from 1 to 0 and returns 1;
the final implementation retains 1 to 1 and returns 0. The ordinary runtime
shadow test alone had not observed that pruned impl metadata. Both executions
and source identities are preserved in [impl-pair](impl-pair/).

## Actual native matrix

The final plain component has **no diagnostic AST probe or test-only wrapper**.
Its ELF SHA-256 is
`a884a7a00f008b7eee372949dd9f2654eb1b4b74a6b2ebf27f4a2feb9837b415`.

| Witness/boundary | Before | After |
| --- | --- | --- |
| Unchanged original alias field/reference | emitter 12, no ELF | compile 0, native 0 |
| Alias chain in field/reference | emitter 12, no ELF | compile 0, native 0 |
| Distinct same-shaped f64 record | emitter 12, no ELF | compile 0, native 0 |
| Explicitly annotated alias parameter/returned reference | emitter 12, no ELF | compile 0, native 0 |
| Ordinary concrete record, generic shadow, concrete constructor | compile/native 0 | compile/native 0 |
| Inferred reference-return receiver, with and without aliases | emitter 12, bare `get` | **OPEN**, same refusal, no ELF |
| Constructor written using alias spelling | checker E015 | checker E015 before normalization |
| Unknown/cyclic used alias | checker status 1 | checker status 1 before normalization |

The inferred-reference-return pair is a real existing lowerer limitation.
The matrix labels it `OPEN_inferred_ref_return_receiver`; it is not an accepted
language behavior or a generic API success. The alias-constructor refusal is
also explicit: this post-verdict pass does not bypass the checker to add that
surface. All original witnesses and assertions are retained.

[scripts/compiler/alias_canonical_component_gate.sh](../../../scripts/compiler/alias_canonical_component_gate.sh)
runs the listed before/after modes under Slurm and records every actual status.
The matrix marker is deliberately
`ALIAS_EXPLICIT_NOMINAL_COMPONENT_ONLY_OK`, with
`INTEGRATED_RAW=NOT_VALIDATED`. The per-mode results and fixture bindings are
in [native](native/). The direct emitter's 12 is distinct from the public CLI's
status 1.

## Source and job custody

The plain native phase of 19447 uses the exact production self-hosted tree,
not the earlier bootstrap snapshot's expanded-capacity profile. Before any
check/build it verifies every self-hosted file and compares the complete file
manifest for equality. Slurm 19476 verifies the same frozen tree again after
the native phase and verifies its archived copy. See
[native/PRODUCTION_SELF_HOSTED.sha256](native/PRODUCTION_SELF_HOSTED.sha256)
and [durable/GUARD_FILES.sha256](durable/GUARD_FILES.sha256).

The 19447 job also runs a whole frontend source check. That independent check
reported E038 in three unchanged functions:
`ir_module_promote_canonical_into_stub_slots`,
`ir_module_ensure_builtin_call_targets` and
`module_frontend_lower_source_file_summary`. It is **not a whole-file PASS**.
The observed log at receipt capture is kept separately. A subsequent small
source cleanup evaluates getters and scalar indices before taking `&!module`;
it is a separate commit and is not silently included in the alias hashes above.

Additional sealed evidence:

- 19434: annotated/inferred/no-alias boundary pairs and original raw CLI refusal.
- 19441: failing/passing impl-parameter AST control and production specializer
  library check, verdict 0.
- 19442: the same plain native matrix on the older expanded-capacity source
  profile, completed 0; this is recorded separately from the actual tree.
- 19476: actual native-phase source guards, durable archive and readback,
  completed 0.

Earlier failures remain recorded: the initial positional seed lacked its
required cwd; an obsolete monolithic lower entry correctly refused; a source
check exhausted an 8 GiB job; another reached its 600-second source-check
watchdog; an isolated replay lacked cwd; and an exact-tree guard caught five
remaining capacity-profile differences before checking. Those are not
reported as semantic passes or discarded favorable retries.

The durable bundle contains the plain actual-tree source/ELF/matrix, compiler
and seed binaries, generic before/after AST artifacts, raw predecessor refusal,
prior-attempt logs/inputs and the independent cleanup diff:

`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/alias-canonical01/alias-component-source-receipts-v1.tar.gz`

SHA-256:
`c63a5095923b3b81b271c94ef8b5a5980af6787edcc4ea21cb7ca28ba247109e`.

[SACCT_AT_CAPTURE.tsv](SACCT_AT_CAPTURE.tsv) retains whole-job states at capture;
a completed native component phase is not recast as a successful whole source
check. Compiler/seed identities are the immutable raw 11332 above and legacy
seed `9cfb6a970d73718b09091331dc514d015e3d4e6f402ee756cdc7f7b0cc55110a`.

## Remaining integration gate

A converged root raw rebuild must replay the unchanged original alias witness
and the relevant imported consumers with Worker/origin/arena fixes integrated.
No raw CLI containing this change has been bootstrapped by this lane. General
alias inference, the open inferred-return receiver, all imported alias forms,
generic storage/HeapVec/HeapDeque, and global compiler/whole-stdlib acceptance
are not claimed by this component result.

## Converged CLI successor, 2026-10-28

Slurm **19528 completed 0** replays the unchanged original
`aliases_fields.sio` through the converged public CLI: **check 0, compile 0,
native 0**. The raw is
`/scratch/sounio-stdlib-wave3-20261027/root-integrated02/results/madaros`,
SHA-256
`8abe891a66c71bdbeb784a9200d1ea395d4d5a55f0036a0daa51ad3cf0614932`.
It includes the root's alias, cleanup, wildcard and Worker lifetime integration.
The original fixture remains at
`compiler-resumed/receiver-raw-array01/source/tests/fixtures/wave3_receiver_dispatch/aliases_fields.sio`
under the sprint scratch root, SHA-256
`0781ca5cfed395df4ff5b13df0314abbc4575c22088dec349e42bdaa4f19a990`.
The same original stdlib path and 512 MiB stack are preserved. A separate
wrong-value copy (expected 18 changed to 19) checks and compiles 0 but exits 1
natively, showing that the original assertion executes. Raw and original
before/final guards pass. See
[supplement/alias-converged-cli01](supplement/alias-converged-cli01).

This closes the unchanged `aliases_fields` CLI replay gate for this particular
converged raw. It does not extend the accepted alias subset or close the
inferred-return, general imported-alias, generic-storage or global acceptance
limits described above.

Slurm **19489 completed 0** also qualifies the separate three-function borrow
cleanup controls: the original nested shared-getter expression is refused with
E037, and the original owner-field index expression with E038 (both
check/compile 1, no ELF); the temporary-value version checks, compiles and runs
0 and preserves the count and bit-value assertions. See
[supplement/frontend-source-borrow03](supplement/frontend-source-borrow03).
The source cleanup remains commit
`15bce9ecfa3c9c2687dd40cad8d3e56b6ca362a0`, distinct from the alias change.

The independent whole frontend source check **19474 ended OUT_OF_MEMORY**,
scheduler exit `0:125`, elapsed 00:19:51, maximum RSS 25163728 KiB. It did not
produce a whole-file verdict 0 and is not treated as a pass. This final
scheduler result supersedes only its previously running status; it does not
alter the completed component or CLI evidence.
