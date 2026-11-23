<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-170.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-170.readme
-->

# SIO-170: dynamic named counts, exact identities and validated CSR

The implementation extends the real incremental `count_tsv` parser and the real `dynamic_normalization` estimator. It adds dynamic named-table framing, explicit byte-exact duplicate/join policies, checked integer coalescence and canonical CSR ownership. API contracts, statuses, allocation boundaries and compiler scope are in `stdlib/genomics/COUNT_IDENTITIES_CSR.md`.

## Source and executable authority

Base: remote main `7ce5eddeef78c76c7e744150ed9c169c53a3d3b0`. Its numerical/compiler input trees equal c21; the exact c21-to-7ce diff contains only the stdlib reliability workflow and its internal changelog fragment. This feature changes the parser and normalization source as recorded in the source manifests. It does not claim the other open sprint PRs are integrated.

The source-built modular compiler is CI run40144792765 / Witness121024346990 / artifact12580970584, built from `c21b0e139830ca4949f80cba62887a8922a972d7`, ELF SHA256 `a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3`. Its 5295 input hashes and tracked path catalog were verified against the exact Git source. The separate committed bootstrap seed has SHA256 `8956946def36004d11ec4eca0d20c75c1e7d3d0b17ee0f0a1fdda9cd6ec0e4cc`; it is not called a fresh rebuilt artifact or the hosted CI Stage2.

All numerical qualification ran as UID1000 on Slurm cpu-ops / cpuops-t560-proxmox with one CPU, 2GiB reservation and 512MiB stack. The token microbenchmark and scale campaigns retain their exact launchers, affinity and per-process observations. The storage quota and unrelated jobs were unchanged. Compiler ELFs were reused from identified custody locations; they are not copied into Git evidence bundles.

## Focused qualification

Final portable gate: **Slurm17349, source19, STATUS0**, 1m35. Actual modular and committed-seed ELFs execute; checking alone is not counted as execution. Source/compiler/reference/caller identities are verified. The gate was launched with the deliberately hostile inherited `SOUNIO_SOUC_BIN=/bin/false` and correctly used its explicit compiler pins.

- Named framing: independent C++23 verifies **15 valid/invalid tables ×256 chunk sizes =3840 native records per engine**. Cases include 512 samples, CRLF split boundaries, 255/256-byte IDs, byte-exact UTF-8 variants/spaces, a final row without newline, i64 maximum, overflow and precise refusal positions. The input and verifier log paths differ.
- Original parser: its existing independent `std::from_chars` reference supplies **24 cases ×5 chunk sizes**. Both snapshot and reusable-result behavior remain checked; the added slot fixture checks all six result fields, stale-result reset, sticky errors, pauses and EOF snapshots. No decimal validation implementation is copied into the named adapter.
- IDs: independent C++ sorting checks **4358 IDs /33074 byte values**, including 4097 IDs, every allowed byte value, duplicates, missing/ambiguous joins, empty owners and atomic output on refusal.
- Count coalescence: independent exact-i64 controls cover **13 cells**, the maximum representable sum, overflow refusal, zero rows and actual two-pass work counts. Required nnz stays unavailable until pass1 completes; a late budget refusal reports only work actually executed.
- CSR: scalar access, explicit zero removal, column duplicate coalescence/refusal and normalization are compared with dense results bit for bit. Independent decimal100 medians/logs check both modes. Nine malformed/precision/zero controls verify unavailable owners and atomic factor/log outputs. The original dense normalization reference still passes, including strided/repeated views and counts near 2^53.
- Scale: the portable gate executes **20000×512** count matrices at 1% row density, dense and CSR, in both selected engines. Independent decimal100 analytic factors check all1024 mode/sample records per process. This is a stated row-sparse scenario, not random sparsity or calibration of a biological model.
- Four corrupted numerical/metadata streams are rejected with exact exits8/9/8/8 and specific numerical diagnostics for changed named counts, missing completion, changed factors and corrupted refusal status.

**Slurm17364** executed the central `record` function and genomic registration block extracted verbatim from `deep_science_gate.sh`, with a fresh child output. Parent and child STATUS0, source/compiler hashes unchanged. The central registration must be retained additively when integrating the other sprint branches.

Earlier scientific replays **17326/17334/17335** pass the parser/CSR/shared dense estimator checks; final scoped gate17349 supersedes their missing registration/structural coverage. IDs17327 and coalescence17329 retain their own frozen source manifests and exact corrupted-input controls.

## Actual compiler limitation, kept separate

The original full gate17337 failed after all positive numerical phases and four numerical mutants passed: it incorrectly expected the committed bootstrap to reject a private-field read. Reachable original-owner controls in **17343/source17** show the actual distinction: modular rejects `IntegerBuffer.size` and `CountCSR.nnz` reads with E259; raw seed895 compiles both. Both compilers reject repeated linear consumption. The exact original IntegerBuffer module is unchanged. A separate explicit-module-declaration control17344/source18 did not repair the bootstrap read behavior.

The final gate requires exact E259/private-field semantics from the modular compiler and exact linearity diagnostics from both engines. It records the accepted bootstrap reads as `PRIVATE_READ_NOT_ENFORCED_BY_BOOTSTRAP` in SEED_LIMITATIONS.tsv, with reachable source, compilation output and emitted ELF identity. These are not seed privacy passes. The negative double-consumption programs are never executed. Numerical compatibility on the seed is distinct from private-owner enforcement on the canonical compiler. No compiler fix is claimed here, and this local seed result does not identify or predict the hosted CI Stage2.

## Measured process time and memory

All entries below are whole-program observations, including input construction, validation, both normalization modes and output; they are not isolated normalization timings. Campaign **17336**, source12, completed STATUS0 in19m26. It executes three pairs in AB/BA/AB order at each density, pins the same actual allowed CPU, compares every dense/CSR factor/log bit and independently checks all factors. The matrix has20000 genes and512 samples: the first stated fraction of genes is positive in every sample, remaining rows are zero. Positive counts are `(gene+1)*(sample%7+1)`. Input counts/dimensions/nnz/checksum are independently verified.

The table uses median elapsed time and maximum observed RSS over the three runs, calculated by the checked C++23 `resource_report.cpp` in **Slurm17352**. Raw pairs are in REPORT.tsv.

| Nonzero row density | Dense seconds | CSR seconds | Dense max RSS KiB | CSR max RSS KiB |
|---|---:|---:|---:|---:|
| 1% | 4.550 | 1.920 | 79872 | 2048 |
| 10% | 11.350 | 11.400 | 79872 | 16384 |
| 100% | 80.270 | 111.300 | 80384 | 160768 |

At1%, the C++ report's dense/CSR median time ratio is2.370. At100%, CSR is slower and uses more memory; it is not an automatic replacement for dense storage. Caller payload queries permit an explicit representation decision. The 1% input payload is81920000bytes dense versus1798408bytes CSR; process RSS and payload are distinct quantities.

The same campaign also recompiles the exact original main normalization core with all other source inputs/caller/compiler identical, at100% density. Three before/after process pairs produce bit-identical complete output. Before median78.520s, after80.310s, same80384KiB peak RSS; the C++ ratio is0.978. This small dense regression is reported, not hidden behind the sparse gains. No statistical significance or universal speedup is claimed. The first direct binary-lookup CSR candidate is also retained in campaign17332, STATUS0/24m18; it established the slow-path baseline before sparse validation/row means were changed to walk stored entries.

The canonical token slot has a separate **ABBA** process benchmark, Slurm17340, with identical rows/positions/checksum before/after:

| Tokens / one-byte feed calls | Original RSS KiB | Reusable-result RSS KiB |
|---|---|---|
| 100000 /200000 | 24576,24576 | 1584,1604 |
| 1000000 /2000000 | 249856,249856 | 1584,1604 |

All old and new processes completed successfully. At1M tokens, original times were0.68/0.71s and reusable-result times0.64/0.63s; no significance claim is made. This measures the token operations. Named framing still constructs its own outer result per feed and resets scalar state per feature row, so it is not a universal constant-memory stream guarantee.

## Retained failures and replay

Immutable packets live under `/scratch/sounio-stdlib-inference-scale-20261024/SIO170` and the sibling IDs/coalescence/token run directories. No failed packet was overwritten. Initial CSR constructor early owner returns and repeated field moves were refused by linear checking; validation moved into a borrowed helper and close destructures once, preserving linearity. A print-argument scalar-kind failure was repaired by typed fixture locals. A generated large literal exceeded the per-function IR instruction limit; the generator now emits bounded helper chunks containing the same bytes. A C++ warning-as-error was corrected without relaxing flags. The IDs fixture originally widened UTF-8 `str_char_at` differently between engines; the byte adapter now maps signed byte values into0..255, retaining all original expected bytes. These are classified individually, not pooled into scientific passes.

To replay the portable gate on an allocated Linux worker:

```sh
MADAROS_RAW_BIN=/absolute/source-built/madaros.elf \
SEED_RAW_BIN=/absolute/committed/souc-lean-single-x86_64 \
bash scripts/stdlib/genomics_count_identity_gate.sh /absolute/fresh/output
```

This establishes focused numerical/contract qualification with the engine-specific limitations above. It does not establish integration of all open PRs, clinical validity, biological calibration or the D34 hardware/formal contract.

## Reviewable custody

- `QUALIFIED_SOURCE_CODE.sha256` is the source-code subset of the final gate's original INPUTS.sha256. It verifies against the feature worktree; this is a byte-identity inventory, not a claim that every listed stdlib module was executed.
- `native-reference-text.tar.gz`, SHA256 `d44326342d867e745fb20b8802225674a5dfd1cb79fc3d60e50805d3a82bcec0`, contains final native/reference/status/negative-control receipts, exact central execution and selected retained failures/sources. ELFs and C++ executables are excluded.
- `paired-scale-text.tar.gz`, SHA256 `a519991d423eb86c8a7ddc8955c6153088faa7c7dd32aa6b20975a0e4ff2cd50`, contains both scale campaigns, raw process observations, comparisons, reference results and the C++ report's source/toolchain/hash/status.
- `ids-coalesce-text.tar.gz`, SHA256 `bc409524aac7255ad55aa8c9518769f4161485f334837d68b331402b8cb89afd`, preserves the independent focused IDs/coalescence runs and their earlier failures.
- `tests/stdlib/genomics_counts/token_receipts.tar.gz`, SHA256 `47af6a42cf3bef1d5fd2ab0542dbf1b5daddfb0dc319bf874d13579ba59a1c81`, preserves canonical parser before/after protocol, reusable-slot checks and paired memory observations.

The independent source review covered sparse traversal order, anchor/zero/precision semantics, staged factor/log commit, identity sorting/join/coalescence and named result reuse. Review is distinct from the native/reference executions above. It also prompted exact diagnostic checks for structural negatives, avoiding accidental matches on probe filenames. Documentation registry/consistency, shell syntax and maintained diff whitespace checks run before publication. No merge, queue enrollment or automatic merge is part of this increment.
