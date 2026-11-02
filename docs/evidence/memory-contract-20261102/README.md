<!-- docs:meta
topic_id: repo.docs.evidence.memory-contract-20261102.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.memory-contract-20261102.readme
-->

# Sounio memory contract audit and AVX-512 repair

Source baseline: 4e3cc61c690755adb391f7391c79c564f19b3909, isolated worktree /workspace/upstream-lab/sounio-memory-contract-20261102. The shared /workspace/sounio source tree was not edited.

## Findings and changes

self-hosted/native/avx512.sio:

1. Current source did not compile: avx_encode_evex_b1 used undefined idx_id. Use its existing x_bit parameter. Register-register callers already supply bit4(rm); no-index memory callers supply zero. Update the direct T71 test to use the same register-register contract.
2. avx_emit_evex_rm_disp32 emitted ModRM followed immediately by displacement for RSP/R12. Those bases require a SIB byte. Emit 0x24, denoting scale1/no-index/base RSP or R12, before disp32.
3. Existing suite ran 72 cases but retained total70. Correct the total and add T73, containing four full LLVM MC literal references for RSP/R12 loads/stores, positive, negative and unaligned displacements.

The EVEX functions here emit disp32, or zero displacement in the separate no-displacement path. No nonzero compressed-disp8 mechanism was introduced, and the ASMC tuple-scale defect is not being asserted for Sounio.

## Validation and provenance

All Sounio builds/tests and native execution ran on Slurm cpu-ops, node cpuops-t560-proxmox. The local machine generated the independent LLVM MC references; it did not run Sounio builds or numerical tests.

Explicit bootstrap seed path (fallback/unit-test route): bin/souc-lean-single-x86_64. SHA256 81937ae1e193af363baa05fb9d868fadf114bf86a72ccab65eae5d424ef2f069. Seed receipt staged with the job. This compiles the actual emitter source plus probe-main.sio into a fresh executable; it is not a rebuild or qualification of the full modular Madaros compiler. No Rust or Python used.

- Job21197: FAILED1:0. Original source rejected undefined idx_id. Preserve this diagnostic; no runtime comparison from the unmodified source is claimed.
- Job21201: COMPLETED0:0. Common prerequisite repair applied to both sides: x_bit resolution, direct T71 argument, suite total72. Without the SIB fix: existing72/72 pass, but independent LLVM comparison rejects12/36 memory cases. With SIB fix:72/72,36/36 byte cases,36/36 native loads/stores pass.
- Job21203: COMPLETED0:0. Final source with embedded golden regression:73/73; independent36/36 bytes and36/36 native execution. Baseline still rejected12 cases.
- Job21206: COMPLETED0:0. Remove only SIB repair from final source: T73 fails, suite72/73, executable exit1. The job succeeds only when this deliberate mutation is rejected.

Matrix: base RDX,RBX,RSP,RBP,R12,R13; displacements -256,1,256; vmovupd load/store zmm0. LLVM references force disp32. The C++23 checker compares every emitted byte before executing the generated instruction. Native routines preserve callee-saved registers and restore RSP before returning; full destination buffers are compared to detect address errors and unintended writes. CPU AVX512F required. This is bounded emitter validation, not a claim about all AVX512 instructions or source-to-native lowering.

Reproduction on the CPU worker:

    sbatch /scratch/sounio-memory-contract-20261102/probe.sbatch
    sbatch /scratch/sounio-memory-contract-20261102/mutation.sbatch

Artifacts: /scratch/sounio-memory-contract-20261102/run-21203 and mutation-21206. Controller logs: probe-21203.log and mutation-21206.log. Local copy: /Users/demetriosagourakis/dev/upstream-lab/receipts/sounio-memory-contract-20261102.

## Width contract audit: initial suspicion superseded by active-route inspection

Existing machinery already preserves narrow byte widths; no new IR field was needed for the confirmed emitter repair:

- lower.sio deref_narrow_width_ref (~25992): signed/unsigned byte pointer provenance versus word-backed local borrows.
- lower.sio (~37980): IR_DEREF_NARROW_SIGNED_FLAG / UNSIGNED_FLAG attached to unary instruction.
- machine_ir.sio (~506,1321,1544): width bits4/12 transported to MIR and through both legalization paths.

Initial suspicion in the compatibility file (not the active compiler import): codegen.sio MIR_OP_UNARY (~3799) turns aux into a float boolean, then native_v2_emit_unary_code (~3737) emits a word load for OpDeref. The observed data flow merits a runtime witness; runtime failure on the default/current compiler is not established. codegen.sio/lower.sio were not edited. Finding broadcast through sounio-coord msg-1793613900-973999-10633.

Correction after tracing compiler/main.sio line45: the active import is native::codegen_x86_linux, not native::codegen. codegen_x86_linux.sio at9319 explicitly masks aux width bits and emits movsx/movzx; its streaming IR path at12180 also preserves the signed/unsigned byte flags. The suspected omission in the compatibility file is not evidence of a defect in the active backend. Runtime source-to-native witness is now qualified: Slurm21221 rebuilt Madaros from this snapshot and passed the signed/unsigned byte and word controls through bin/souc. Slurm21222 passed the independent C++23 oracle and rejected a deliberately widened load. See ../bytewidth-active-20261102/README.md.

Blocker-ID: MEMWIDTH-NATIVEV2-20261102
Status: closed for the tested active source-to-native route; initial suspicion superseded, runtime witness and negative control passed
Severity: not a confirmed regression
Class: codegen/width propagation
Evidence: source data flow only
Owner: codex-memory-contract, next isolated validation lane
Acceptance result: active source-to-native signed/unsigned byte and word controls passed with a freshly built compiler and semantic negative control. A separate direct-MIR runtime test was not performed; both MIR and streaming flag handling were inspected in the active file.
Next action: retain this regression; do not repair the active backend based on the inactive-file observation.

## Review and operational limits

Auxiliary LLM: darwin-llm sounio_verificado, one small harness candidate; returned invalid code. Harness corrected locally before cluster compilation. Orthogonal review: darwin-llm coder-next alias; actual backend not attested. Reviewer alleged x_bit regression despite the callers explicitly supplying the correct register-register bit; alleged other required base-only SIB cases without identifying one; misread expected baseline failures as candidate failures. Those claims were rejected against call-site inspection, independent bytes, and hardware execution. No architecture decision delegated.

Required startup helpers ONBOARDING.md, CLAUDE_HANDOFF.md, sounio-whereami and .claude/PARALLEL_BLOCKER_CONTRACT.md were absent at this branch. This is documented as missing tooling, not an empty coordination bus. bin/sounio-coord was used for an exact write claim, heartbeat, and findings message. No shared dirty file overwritten. No legacy compiler path removed. No push, merge or deployment performed.

## Historical receipt scope

The .sbatch files preserve the original job invocations and their fixed scratch/toolchain paths. They are historical receipts, not portable CI entrypoints. For integration-time validation, use docs/evidence/memory-integration-20261102 and its new job receipts. The source snapshot and artifact identity in each report delimit its claims.
