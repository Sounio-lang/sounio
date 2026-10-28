# Frontend exclusive-call source cleanup

This independent change moves shared reads out of three old `ir_fn_set`
argument lists before constructing the exclusive call. It does not change the
checker, lowerer, builtin admission, source alias rules or Worker resets.

- Canonical stub promotion computes the existing deep-copied function first,
  then passes that value to `ir_fn_set(&!module,...)`.
- Builtin stub creation captures the existing scalar function-slot index first.
- Source summary construction captures its existing scalar function-slot index
  first.

The setter itself writes the addressed function chunk entry and does not change
`fn_count`; the original count updates remain in place. The selected getters,
deep copy, destinations and values are unchanged. No full IrModule copy is
introduced.

The original frontend file SHA-256 is
`84f5faa7bf16d3df7cc1a39b233ab33f8c16342eb781b4e5e311ce37c73bcd97`;
the file after this cleanup is
`6323340644e75295850900a6d017d18f385696f7cd7c9a2df4050c32afdce06f`.
The actual-tree source check in 19447 reported E038 in precisely these unchanged
functions before this cleanup. Its observed log and exact-tree guards are in
the preceding alias evidence. That whole frontend check is not represented as
a successful gate.

Slurm **19489 COMPLETED 0:0** qualifies the explicit argument-borrow pattern on
the immutable raw compiler
`11332ed8a6cf7ccb85f0a31bf63bed1fcc10659054d68639d6dba233a6014ff4`:

| Control | Check | Compile | Native |
| --- | --- | --- | --- |
| Exclusive argument plus nested shared getter/deep copy | 1, E037 | 1, no ELF | not emitted |
| Exclusive argument plus source object's field as index | 1, E038 | 1, no ELF | not emitted |
| Getter/deep copy and index captured before exclusive call | 0 | 0 | 0 |

The positive retains both count 7 and payload 19; neither assertion was removed.
Its ELF and all fixture/compiler before/final digests are recorded in
[results](results/). [worker.sbatch](worker.sbatch) records exact commands and
the original per-command deadlines. All execution was under Slurm.

19483 had incorrectly required E038 for both negative shapes; the nested shared
borrow actually reports E037. A subsequent harness regex was incorrectly
escaped. Neither attempt changed the fixtures, compiler or implementation.
Both failed JobIDs remain in [results/SACCT.tsv](results/SACCT.tsv); the corrected
fresh-directory gate checks the borrow-conflict diagnostic family and retains
the actual E037/E038 outputs. These operational failures are not semantic passes.

The actual corrected-tree whole frontend check is running separately as 19474,
with all self-hosted files bound to its manifest. Its completion is not claimed
here and does not gate this limited source-component result. The converged root
raw/preflight must still validate the combined frontend/Worker/canonicalization
changes; this commit does not claim an integrated CLI rebuild or whole-stdlib
acceptance.
