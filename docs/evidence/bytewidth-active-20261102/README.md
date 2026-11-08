<!-- docs:meta
topic_id: repo.docs.evidence.bytewidth-active-20261102.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.bytewidth-active-20261102.readme
-->

# Active backend byte-width witness

Source snapshot: c1e89588a730, worktree /workspace/upstream-lab/sounio-memory-contract-20261102. This is a follow-up to MEMWIDTH-NATIVEV2-20261102.

## Correction to the initial suspicion

compiler/main.sio imports native::codegen_x86_linux at line45. The earlier source-only suspicion inspected native/codegen.sio instead. The active x86 backend explicitly consumes width flags4/12 in its MIR unary dispatch at9319, and its streaming IR OpDeref dispatch at12180. Both choose signed-byte MOVSX or unsigned-byte MOVZX. A word read remains the default for word pointers and appropriate word-backed borrows.

The compatibility-file observation does not establish a defect in the active backend. No emission or lowering implementation has been changed in this follow-up. The prior report was corrected with this distinction, preserving the original hypothesis as history. Coordination correction message: msg-1793614470-1191104-12696.

## Witness

A real heap allocation contains a word with distinct, nonzero neighboring bytes. Typed helper functions read the same address through *mut i8, *mut u8 and *mut i64. Low-byte fixtures: FF,80,7F,00. Expected signed values: -1,-128,127,0. Expected unsigned values:255,128,127,0. Full-word checks ensure a byte load is not accidentally used for a word access.

A C++23 oracle independently constructs the expected stdout for the FF fixture and requires the all-cases success marker. An intentionally changed program casts the unsigned-byte pointer to *mut i64 before dereference: it must return exit2 and be rejected by the oracle. This is a semantic negative control of the test; it does not mutate the compiler.

## Build and execution

All compiler builds and tests execute on Slurm cpu-ops, cpuops-t560-proxmox. Source snapshot is archived into /scratch/sounio-bytewidth-20261102/source. The canonical scripts/ci/build_modular_madaros.sh derives a source-tracking seed, then builds compiler/main.sio. Actual resolved bootstrap bin/souc-linux-x86_64 SHA256 a63ca2c960183aafcdca56e57a0c2da88b5a2005db9df5c4f2dc6a6434b8a694. The separately recorded lean_single seed is not the bootstrap selected by the build script.

The witness uses bin/souc compile with SOUNIO_MADAROS_BIN explicitly naming the freshly built artifact. No Rust/Python; no replacement by an older Madaros ELF. This tests the source-to-native default route for this scalar program. It does not separately execute every internal MIR path or qualify all narrow types.

Commands:

    sbatch /scratch/sounio-bytewidth-20261102/active.sbatch
    sbatch --dependency=afterok:21221 /scratch/sounio-bytewidth-20261102/check.sbatch

Results: Slurm21221 COMPLETED0:0 in7m14s. Fresh Madaros SHA2566deefb97c00e62e90300c654813cd71fb0d2a3bdda9b95e8f2e86d54f50e4f59. Public resolver compilation log confirms native_v2_compile,17 merged IR functions. Native witness exits0 and prints -1,255,305420031,BYTEWIDTH_ACTIVE_PASS. All10 explicit value checks pass. Slurm21222 COMPLETED0:0 in2s. C++23 oracle accepts witness; deliberately widened unsigned dereference returns exit2 and is rejected by the oracle. MEMWIDTH-NATIVEV2-20261102 is closed for the tested active source-to-native route. Both runtime success and semantic negative control are recorded; no backend repair was needed.

## Reviews and state

Auxiliary darwin-llm sounio_verificado was called for the small witness and returned an invalid candidate. Corrected before compilation. No mathematical/clinical claims or architecture change. Final review relies on import tracing, explicit artifact identity, native result and C++23 oracle.

Only a test fixture and evidence documentation are added. No shared dirty files, compiler source changes, push, merge or deployment. The AVX512 repair from the preceding step remains separate.

## Historical receipt scope

The .sbatch files preserve the original job invocations and their fixed scratch/toolchain paths. They are historical receipts, not portable CI entrypoints. For integration-time validation, use docs/evidence/memory-integration-20261102 and its new job receipts. The source snapshot and artifact identity in each report delimit its claims.
