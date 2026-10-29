# Durable argv custody and exact-ELF controls

Slurm19609 completed0 and sealed the final sources and receipts, including the orchestration failure19597 and original current-document admission failure19599. It verified the source/compiler/executable manifests of every included execution before and after sealing. Slurm19613 completed0 and copied the archive to OrangeFS with SHA-256 and length readback.

Archive:
`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/argv-export08/stdlib-argv.tar.gz`

SHA-256 `3d7125422c24b7ed9709e7debf8078a6675ca247149c3c0ad91ec534d67366d3`, 944737421 bytes. Custody STATUS0, COPY_STATUS0, READBACK.verify0.

Contents include complete staged input trees, raw phase captures, original and corrected sidecars, historical preimages and their Git/blob/SHA ledger, final material executable, every before/after failure and success receipt, raw8abe with its actual compiler source/ready/results, the Node binary used by the orchestration and immutable patch03e2. The source snapshots are the actual staged trees used by these focused runs; this archive does not claim full-stdlib execution or acceptance.

A supplementary direct control used one retained material ELF for all three executions, avoiding any inference from equivalent recompilations. Slurm19615 completed0/2s with ELF SHA-256 `355ee404756da8249c2b630eae4c2c132b5af838fce1752f2dcd64f367840e72`:
- The twelve frozen files returned0 and printed the original passed marker.
- Empty argv returned2.
- The twelve original current-document paths returned1 and printed failed9.
- The executable and complete input manifests verified unchanged afterward.

The control recipe, stdout, exact statuses and complete guard readbacks are sealed separately:
`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/argv-export08/same-elf-control.tar.gz`

SHA-256 `831de26eeaed92771c040d61e5f10bb3ac7bc71b3a031e6bf6ab9a61631210fc`, 167972 bytes. Readback verified. Its executable and input paths resolve to the final42 tree already contained in the main archive.
