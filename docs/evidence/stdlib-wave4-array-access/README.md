# Typed reads of local BigFrame accumulator arrays

This change replaces42 reads through raw-pointer casts of lexically declared local arrays:17 in bigframe_ops,18 in bigframe_stats and7 in bigframe_ml. Each target is a unique local declaration res:[f64;1024]. The ledger records the owning function, declaration/access line, complete before/after statement, source hashes and byte-exact reversal. Only read_f64(res as *mut f64,k) becomes res[k]. The existing k>=0 && k<1024 bound, formulas, loop order, API, effects, allocation sizes and original test assertions remain unchanged. Heap-pointer casts, including the sc scratch-cell reinterpretations, are preserved.

TYPE-017 does not promise an arbitrary representation conversion through as. EXEC-014 distinguishes an aggregate handle from the address of its slot. The library therefore reads its typed local arrays directly. This is a correction of library use, not an ABI argument-packing workaround or an implementation of array decay.

## Qualification19460

Slurm19460 COMPLETED0 in9m30s,32GiB/2CPU; batch MaxRSS12194684KiB. Raw compiler is the immutable integrated11332ed8a6cf7ccb85f0a31bf63bed1fcc10659054d68639d6dba233a6014ff4. Before source is byte-identical to root-integrated01/source/stdlib; after differs only in the42 recorded expressions.

All three libraries check successfully before and after. The unchanged C++23 oracle verifies nine ABI controls and the scientific median fixture in both loop-region modes, on both snapshots. The18 ABI controls pass before and after. The scaled-median fixture changes from native SIGSEGV139/oracle refusal to native0/oracle0 in both modes, exact3.75 bits4615626668101337088.

| Original, unchanged scientific program | Before mode0 | After mode0 | Before mode1 | After mode1 |
|---|---:|---:|---:|---:|
| test_bigframe_ops_stdlib.sio | arena181 | arena181 | SIGSEGV139 | 0, BIGFRAME_OPS_GATE_OK |
| test_bigframe_stats.sio | SIGSEGV139 | 0, BIGFRAME_STATS_GATE_OK | SIGSEGV139 | 0, BIGFRAME_STATS_GATE_OK |
| test_bigframe_ml.sio | SIGSEGV139 | 0, BIGFRAME_ML_GATE_OK | SIGSEGV139 | 0, BIGFRAME_ML_GATE_OK |

Every original program checks and compiles0 in all comparisons. Native deadlines remain120seconds. The full BigFrame witness stays SHA2566f95285b3f9c239382a7f44616644b7cdeeca1740daa51e177edf838c3b48a23. No marker instrumentation or assertion/tolerance change is used in this qualification.

All final input, library, compiler, original-witness/tool, matrix-input and reference guards pass. The collector retains actual failures in PHASES.tsv; its0 is custody completion, not a claim that default mode0 passes. The mode0 arena181 frontier is independent and remains open.

The C++ oracle establishes exact transport/median results and ABI control values. The broader numerical extension formulas are checked by their unchanged original assertions; this receipt does not claim a new independent C++ rederivation of every statistical/ML formula.

## Representation evidence and separate work

19448 collects the original16 array controls plus heap/read_i64/raw-index and same-array reference-pointer controls. Raw heap reads and same-array pointer reads/writes pass their C++ bit oracle. Array-value to element-pointer casts still fault; array-reference to element-pointer casts read bits1. Same-array slot casts retain valid behavior. The additional array-element-borrow alias control exposes a separate issue and is not repaired here.

19480 reuses root's qualified C++ ptrace tool without edits. A value-array cast faults with RDI=1, RSI=0 and address1 at the raw read. Heap and same-array controls exit0. The compiler cast-guard proposal remains a separate artifact; this change edits neither lower.sio, checker, native codegen nor read_f64.

Source/evidence roots:
- /scratch/sounio-stdlib-wave3-20261027/services/bigframe-local-arrays11/input
- /scratch/sounio-stdlib-wave3-20261027/services/bigframe-local-arrays11/results
- /scratch/sounio-stdlib-wave3-20261027/services/array-semantics10

run-19460.sbatch is the exact executed orchestration. The already qualified C++ source is tests/fixtures/wave4_bigframe_array/reference.cpp; the immutable matrix inputs and compiled reference are retained with their hashes. The parent worktree was3e71ac08; only the three library files and this ledger/receipt are changed here. Durable custody follows separately. No push or main merge was performed.
