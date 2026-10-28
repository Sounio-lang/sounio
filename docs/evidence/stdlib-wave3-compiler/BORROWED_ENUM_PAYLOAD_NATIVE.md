# Borrowed enum payloads: bounded native qualification

This change admits references to the native scalar cells and registered ordinary struct handles in user enum payloads. Named struct and enum fields now participate in the persistent owner-loan walk. A borrowed carrier cannot move its affine mutable-reference payload. Lowering stamps reference/pointee metadata after aggregate inference so the latter cannot erase the pointee.

Arrays/tuples/Box containing references, nested references, unit references, nonexistent i16/u16 cell kinds and other unproved representations remain refused. This is not a general opening of generic borrowed storage, closure storage, or HeapVec.

## Exact native run

Slurm 18849 built a real raw Madaros from git archive b1876ea0c2 plus only the sealed borrowed-ref feature overlay. It excludes capacity WIP, the later checker shadowed-memory fix, and the later root cast/exit fix. Durable directory:

`/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/borrowed-ref-raw-b187`

- Baseline archive SHA256: 939db99ae32fbf5158f0cc7d732ecead3ff559930c69c27d0857bf0e63507127.
- Feature overlay SHA256: 9d3012890aabd0043672e55abc6d53a928938744c0d2107a8c5f28f0d1dfc584.
- Raw compiler SHA256: 0b6a36d3f95513b0cfa06c4de2e7a34345ad538ae67dbf071c1787e21133ed43.
- Strict check matrix: 46 cases, 16 positive and 30 negative, PASS.
- Strict compile matrix: the same 46 cases and diagnostics, PASS; all negative cases produce no ELF.
- Native: 15 of the 16 existing positives pass. All five new borrowed-enum positives pass with their expected markers: scalar, f64, nominal owner, mutable-reference move, and the combined supported scalar/string cell witness.
- Additional original string-only and combined witnesses compile and execute successfully.
- Final source, feature, seed, raw compiler, and archive guards pass.

The outer job is FAILED1 because the original `returned_loan_pattern_forward` compiles but executes with status139. It destructures a tuple, and the existing lowerer skips PatTuple arm bodies. The gate deliberately retains that failure. An earlier tuple-scope witness was therefore insufficient evidence of native tuple execution. The separate typed-tuple correction is not part of this commit.

## Component controls and preserved failures

The actual checker component in Slurm18806 passes the same46cases. Its SHA256 is4c530ed1efd0e43ad2083e46b63c7f04b1f92dd25e055a9b9649ec236452a189. Slurm18824 runs the C++23 strict verifier and seven wrong-output controls; the compile-format control is only a verifier control, not a compiler execution claim.

Slurm18807's direct actual lowerer/native component executes scalar/f64/owner/mutable-reference witnesses. Its combined-string witness failed an unresolved-string lowering path also observed without enums; real raw18849 subsequently executes the original combined witness unchanged. No source witness was weakened to make that component pass.

Slurm18893 recompiles the actual checker after adding the separate typed-match draft: all46 verdicts and seven wrong-output controls still pass. Its first tuple attempt refuses all new tuple programs because the auxiliary ontology pass cleared the checked-type cache. This is preserved as a refusal, not reported as acceptance. Subsequent tuple qualification belongs in a separate receipt.

Before relocation of the reference stamp, Slurm18805 showed a reference-to-f64 binding losing its declared pointee after aggregate metadata inference. The stamp-after-inference correction is exercised by18807 and raw18849.

## Acceptance boundary

These receipts qualify borrowed enum payload admission, loan diagnostics, and the five new native payload witnesses. They do not close the full returned-loan native gate, owner-view import/body lowering, tuple patterns, raw root integration, storage hooks, HeapVec, WorkerFn, or whole-stdlib acceptance.
