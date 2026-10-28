# Typed storage native value-cell ABI — qualified components

Slurm 18971 completed exit 0. The portable gate checks the entire 34-module
emitter dependency closure and the registration component with modern raw
c02a0d212761d1954959f53832689d44887da96a048749dbff4a3976c268c3dd.
It then compiles both components with legacy seed
9cfb6a970d73718b09091331dc514d015e3d4e6f402ee756cdc7f7b0cc55110a.
Both generated components execute successfully. This separates modern source
type admission from legacy bootstrap code generation.

The protected native symbols use IDs 1000+op, outside platform and worker IDs.
Registration deduplicates only matching empty stubs, checks arity, returns -1 on
invalid operations or capacity/storage failure, and leaves the shared out stub
integer-shaped. Concrete float call instructions carry their own float marker.
Invalid protected native operations latch ir_validation_failed, which the existing
writer checks before publishing an ELF.

The C++23 oracle executes actual bytes emitted by the Sounio backend, with W^X
mappings. It covers all ten word/unit opcodes, exact word bit patterns (including
negative zero and a NaN payload), zero-byte unit behavior, both directions of
overlapping movement, disjoint movement, same-pointer/zero-count behavior, and
negative count, multiplication/address overflow, null and unaligned endpoints.
Failing operations run in children and must exit 1 before returning. With native
barriers enabled, relocations point to an alignment-checking recorder: it checks
the actual value/owner pairs, root ownership on move-out and borrow, reverse-order
range barriers, and no value access/barrier for byte-free unit moves. Register
preservation and the barrier call's SysV stack alignment are exercised directly.

The full closure check exposed an existing witness call that evaluated
nc.entry_offset after acquiring &!nc. Hoisting that scalar before the call closes
E038 without changing emitted behavior. The earlier 8 GiB typecheck 18949 both
reported that error and was OOM-killed. The corrected closure passed in 18967 with
24 GiB and again in the complete portable gate. 18961 was canceled after identifying
an omitted injected probe in that harness snapshot; it is not a pass.

Other preserved boundaries: 18928 used the wrong nonexistent compile_ok field in
the new emitter and its native component failed 41; the implementation now uses
the existing ir_validation_failed flag. 18938/18942 passed the first transport
component and C++ oracle; 18948 passed enabled-barrier transport. Registration's
modern check passed in 18951, but that raw compiler crashed during dependency
lowering; the legacy native registration component passed in 18962. The portable
gate deliberately documents the two compiler roles rather than claiming that
crash resolved.

Receipts and immutable source/executable/compiler hashes are beside this file.
Full artifacts: /scratch/sounio-stdlib-wave3-20261027/storage-portable/result.
This qualifies the internal value-cell backend and factory, not source-level
typed storage hooks, safe generic HeapVec/HeapDeque admission, a rebuilt final
compiler, full region reclamation, or whole-stdlib acceptance. Those remain
required integration gates.
