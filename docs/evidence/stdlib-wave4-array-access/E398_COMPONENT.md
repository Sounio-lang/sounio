# E398 candidate checker component: failed qualification retained

Slurm 19501 is FAILED 1 after 1 minute 53 seconds, with batch MaxRSS 394032 KiB. It builds two executables from the real parser, specializer and checker source using seed 9cfb6a970d73718b09091331dc514d015e3d4e6f402ee756cdc7f7b0cc55110a. The before/after source differs only by the artifact proposal. This is check_items_verdict_boot4 on one input file; it is neither a BFS/import-loader gate nor an integrated compiler/native-emission gate.

Both checker builds exit 0 with no error diagnostics, unresolved symbols or dropped writes. Seed warnings about assignments and large stack frames are preserved. All input, executable and tool guards pass after execution. The job deliberately fails its original expectations; they were not revised after observing failures.

Twenty-two source producers are selected. Their sources and EXPECTED.tsv are immutable within the run. The expectations require all before checks to pass and twelve after refusals with E398.

Observed results:

- The six value/reference-to-element cast fixtures change from accepted to E398.
- Both local raw-pointer ABI witnesses change from accepted to E398.
- Wrong extent, wrong element type and const-reference-to-mutable-pointer controls change from accepted to E398.
- The global/BSS GBUF:[i64;2] to Pair witness remains accepted, despite its expected E398 refusal. This gap prevents adoption.
- Typed arrays, heap read_f64/read_i64, same-complete-array reads/writes, the explicit element-borrow diagnostic and the BSS same-array pointer remain admitted.
- heap_index fails with E013 before and after: "indexing requires an array type". This is a pre-existing component difference; the pinned raw11332 integrated compiler checks/compiles/executes that source successfully in 19448. It is not normalized into a passing positive.

The BSS result is an observed boundary failure. This receipt does not attribute it to cast lowering or claim that the driver has complete global-variable type registration. Further source investigation and the existing ref-to-raw owner must resolve that distinction before a guard is adopted. Nested array extents, unresolved/generic leaves and complete import resolution remain unqualified.

The preceding attempt, 19498, failed in one second before checker construction with E224 unreadable import self-hosted/crypto/sha256.sio. Its input package omitted stdlib. The fresh 19501 package includes stdlib and preserves the earlier failed package unchanged.

Source/results paths on DL380p:

- /scratch/sounio-stdlib-wave3-20261027/services/array-cast-checker14
- /scratch/sounio-stdlib-wave3-20261027/services/array-cast-checker15

run-19501.sbatch is the exact executed script. No tracked checker, lower.sio, native codegen or read_f64 file is edited. E398 remains an unadopted artifact. The library42 correction and its original scientific tests remain independently qualified.

## Failed-component durable custody

Slurm 19503 completed 0 in 13 seconds (512 MiB, two CPUs; MaxRSS 115844 KiB) while sealing the failed 19498/19501 inputs, executables, diagnostics and complete source snapshots. It does not change either failed verdict. The archive is /orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/array-cast-checker-export03/array-cast-checker.tar.gz, size 110,207,209 bytes, SHA256 363720895cdeb7ed3a1bef9cde7a9540d4acd953ca8099710e124092f63bc32e. The OrangeFS readback matches; adjacent metadata and DURABLE_READBACK.verify retain the checks. The exact custody script is run-19503.sbatch.
