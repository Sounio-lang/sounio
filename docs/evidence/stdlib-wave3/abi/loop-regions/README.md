# Opt-in loop-body regions

Source base: f861ef481137 plus the lowering/native changes committed with this receipt. The bootstrap snapshot additionally applies the separately qualified capacity overlay eaa61d90e8d2d1897f545636c06496f8b0791fbbf206dca5bd4fa4151926ac60. That overlay is not part of this change. Bootstrap seed: 9cfb6a970d73718b09091331dc514d015e3d4e6f402ee756cdc7f7b0cc55110a.

The feature requires SOUNIO_LOOP_REGIONS=1 and the existing region runtime. It is disabled by default and excluded from kernel loops. Each iteration enters after its condition; break and continue close that body. Returns leave records without reclaiming and use the existing function return handling. Written outer registers without numeric proofs preserve the current record. The private native emitter1101 never marks ancestors; ordinary stores retain their transitive barrier.

Closures get independent loop labels. Numeric binding proofs are saved before lowering the closure and restored afterwards; the first prototype reused parent state and its closure stress tests failed with139/181. Those failures remain in /scratch/sounio-stdlib-wave3-20261027/loop-regions-closure. No loop count, runtime arena size or per-case deadline was reduced to obtain the corrected result.

Slurm evidence:

- 19096: rebuild and returned-loan gates in both option modes, exit0. Each mode checks51, compiles51 and runs21 positive cases;30 negative cases and wrong-output controls remain rejected. Rebuild stdout contains no explicit error[E...] or field-initializer mismatch diagnostic.
- 19110: modern compiler checks the complete lower and native module closures, both exit0. This checks source types/effects independently of the legacy bootstrap.
- 19119: seven original stress inputs, all native0 after the closure isolation fix.
- 19128: the committed portable gate replays the same rebuilt raw compiler with option0 and1. while/continue_break/for each exhaust the fixed arena with181 in mode0; all seven mode1 cases finish0 with exact markers. Inner carry runs20million outer iterations; the other large allocation loops retain their original4million iteration count.

Raw compiler SHA256: bd4ba2409a0097fdc9e9b4fe32e5858e396d4d71dd3515648a6f01d451b8ace1. Portable gate retains input, compiler and generated ELF hash checks. Reproduce only on a scheduled Linux x86-64 worker:

    AFTER_RAW_BIN=/absolute/rebuilt/madaros bash scripts/ci/wave4_loop_regions_gate.sh /absolute/fresh/output

The original BigFrame test is still blocked at borrow checking by the128-binding saturation and has no corrected end-to-end runtime receipt. Full stdlib acceptance remains open. This change does not certify worker callbacks, alternative native backends, serialized provenance flags, or general allocator reclamation.
