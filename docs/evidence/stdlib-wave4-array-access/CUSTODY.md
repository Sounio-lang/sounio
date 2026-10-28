# Durable custody for the local-array repair

The library repair is commit c3c4345e180557c9e659671258807b0a8d987b75, independently adopted by root as 9f8340e0f9. The source, before/after ledger and original scientific assertions are unchanged by this custody-only follow-up.

Slurm 19499 completed with status 0 in 12 seconds, 512 MiB requested, two CPUs and batch MaxRSS 78552 KiB. The job checks the original manifests, creates and tests the gzip archive, then checks every archived input again. The archive was relayed from the DL380p local scratch to the mounted OrangeFS on DL380 and its complete bytes were read back for SHA256 verification.

Durable archive:

`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/array-access-export02/array-access.tar.gz`

SHA256: `e8f109d198ab4a6a659d8a93faee15e66bc70854522a1b1193456a212ccae59e`

Size: 73,947,031 bytes. The adjacent ARCHIVE.sha256 and DURABLE_READBACK.verify pin the remote copy. metadata contains Slurm accounting, original manifest checks and post-archive checks; export.sbatch is the exact executed custody script.

The package contains the complete before/after library snapshots, all 19460 results and ELFs, original immutable BigFrame/Stats/ML programs, raw11332 plus RAW_READY, the ABI inputs, the unchanged compiled C++ reference, 19448 IR/native/oracle results, 19480 ptrace results and tracer source, the c3c4345 patch and both versions of the static impact inventory. The candidate checker patch is retained as a clearly unadopted artifact. The contents manifest is inside the package and beside it.

The older integrated-frontier archive remains available separately with SHA256 ea122de0ca5fbc2d6d674fc6bbacc80e0a74a16eb73fae8210784ef0d93dee2d. It preserves the earlier failures and private GCC recovery but predates the three qualifications above.

## Remaining limits

- BigFrame ops in SOUNIO_LOOP_REGIONS=0 still reaches arena status 181 before and after the repair. It is open.
- The element-borrow alias control still fails its C++ oracle.
- E398 is not adopted, and no integrated checker/native acceptance is claimed for it.
- The first E398 component attempt, 19498, stopped before checker construction with unreadable import crypto/sha256 because the staged source omitted stdlib. Its failure is preserved in array-cast-checker14. The complete-source attempt 19501 failed its original expectations and is documented separately in E398_COMPONENT.md; this archive predates that component and does not contain its later results.
- No push or merge to main occurred.
