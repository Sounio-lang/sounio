<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-184.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-184.readme
-->

# Romberg exhausted-budget error estimate and analysis consumer

Romberg used to return an error estimate of zero when its iteration budget ended without convergence: the final subtraction used prev_best after assigning best into it. The library now retains the last measured diagonal-table change. The integral, convergence threshold, iteration cap and successful-convergence branch remain unchanged. This is a heuristic last-change estimate, not a rigorous error enclosure.

The original analysis consumer imports analysis::lib and preserves its15 assertions. A separate run-pass checks positive decreasing error estimates at budgets2/3 and polynomial convergence. The independent C++23 generator uses incremental long-double trapezoids and an explicit Romberg table, while Sounio recomputes full trapezoids. Four builtin functions at budgets2/3/6/8 produce16 generated cases with1e-10 value/error concordance and exact iteration/convergence status; standard analytic integrals backstop the reference. No Python, Rust or Julia is used.

## Actual job17651

One source-bound Slurm job ran for15 seconds, requesting1CPU/4GiB and receiving2logical CPUs/4GiB. Both engines reproduce the original exhausted-error bug: the direct regression exits2 and the generated concordance program exits201. After the correction, the original consumer, direct regression and generated16-case program all check, compile and execute with exit0 on Madaros and lean_single. Their expected sentinels were observed.

The aggregate job remains FAILED1:0. Its41 phase rows include four strict seed contract mismatches: before consumer check/compile returned0 instead of expected1, and direct standalone library checking returned1/E221 no main before and after instead of0. Those expectations were preserved. Neither acceptance of the original import nor E221 is relabelled as numerical success or claimed to be a newly introduced compiler defect.

The raw compiler is the external historical c21 ELF SHAa292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3. The canonical seed is SHA06c90e5b408e3f42f7dbf5c0ef34817333381a770a0734dae4493e36a496c939 with paired SeedReceipt cd9b24558998696183243ebaabb01eb0bfef80a6c519e1504912ced1caa7ebec. This does not certify a current-main compiler rebuild or fixed point.

Root independently extracted the444-file compact text receipt, verified every file hash, the41 phase rows and exact four mismatches, all current production sources against the staged after catalog, and51 verification manifests containing21720 OK/PASS lines. Both before/after catalogs contain2398 entries. The source/mode/link census, raw/seed, generated source, reference/native ELFs, toolchain and dynamic runtime closures closed. Raw and seed are explicitly classified as static by program headers. Controlled Python refusal probes are separate from the zero-byte runtime log.

The retained compact archive SHA256 is fe978151463bb3aab84547cebf3394bc94b193cef077618897ab95495277bf69. It contains command/output/resource/ledger/controller/source-catalog metadata without executable or full-source bulk. Original source and ELFs remain at /scratch/sounio-stdlib-inference-scale-20261024/SIO184-analysis-wave01; original terminal archive SHA4f17d39b04da777841ef4808e9b5619380cad6c334c9022093e82ea7f9ad9859 is preserved separately.

Reproduce only the metadata review with Node:
node docs/evidence/stdlib-SIO-184/verify_receipt.mjs docs/evidence/stdlib-SIO-184/terminal17651-text.tar.gz "$PWD" /tmp/romberg-root-review.json

The other19 analysis/mathematics cases from the earlier SIO184 triage remain outside this focused increment. There is no full corpus, generic-callback integrator, arbitrary precision, formal error-bound, compiler build, or seed fixed-point claim. Notion SIO184 remains In progress.
