<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-169.log-slot-experiment.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-169.log-slot-experiment.readme
-->

# Retained-log-slot memory experiment: hypothesis not confirmed

The final review requested a paired memory witness for replacing the per-point `ln_checked`/`log1p_checked` result with one retained canonical `CheckedLog` slot, borrowed as a whole struct by the risk-update and curve-emission helpers. The proposed patch preserved equations and operation order. The original qualified code at `d7277d28315ba4fea5bd023ccae4d2959fdc898a` was preserved in immutable source packet 13; the experiment is immutable source packet 14.

The experiment **did not demonstrate a representative execution-time or resident-memory improvement**. The proposed production refactor was therefore not retained. The shipping source and scientific gate were restored exactly to the source qualified by Slurm 17367. `final_restored_source.verify.stdout` confirms all seven tested source hashes. The patch, benchmark generator/controller, traces and measurements are retained here so this negative result is reviewable. This is not an allocation fix or a performance improvement claim.

## Paired standalone measurement

Slurm **17369** completed with exit **0:0** in **00:10:15**, under a **2048 MiB** allocation and a 524288 KiB stack. Both engines execute each version independently in a fresh process. The source-built compiler is SHA256 `a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3`; the committed seed is `8956946def36004d11ec4eca0d20c75c1e7d3d0b17ee0f0a1fdda9cd6ec0e4cc`.

Before **both** check and compilation the controller explicitly changes cwd to source packet 13 (baseline) or 14 (experiment), invoking that directory's `bin/souc`. This matters because the committed raw seed loader does not use `SOUNIO_STDLIB_PATH` as source authority. The seed ELF hashes differ between variants at both sizes, as recorded in `memory/results/*.elf.sha256`; the generated input program is shared and separately hashed.

The fixture contains 100,000 or 1,000,000 **distinct** times, all observed deaths and alternating groups. The input permutation is descending time, requiring the real sort. All three cohort records are written at every pooled time; this is distinct from the earlier million-record, six-time ambiguity controls. The requested worst-case curve/sort payload is 296*n bytes, with another 24*n input bytes. Each run reads every numerical output field's raw binary64 bits and every integer output, accumulating two bounded arithmetic digests for each buffer. Public result fields and the digests are printed; the exact native stdout SHA256 and byte comparison are recorded. These are useful change detectors, not a cryptographic proof that every buffer value is identical.

| Distinct times | Engine | Before elapsed / max RSS (KiB) | Retained-slot experiment elapsed / max RSS (KiB) |
| --- | --- | --- | --- |
| 100,000 | Committed seed | 0:19.02 / 34316 | 0:18.93 / 33808 |
| 100,000 | Source-built Madaros | 0:07.58 / 31248 | 0:07.62 / 31252 |
| 1,000,000 | Committed seed | 3:11.57 / 314892 | 3:11.26 / 315408 |
| 1,000,000 | Source-built Madaros | 1:18.20 / 312336 | 1:18.73 / 312340 |

All eight processes completed successfully. Each before/after pair matched its all-field digests and public stdout exactly. There was no baseline failure or managed-handle-cap failure, and no failed-run performance ratio is reported. These single paired observations do not establish a universal absence of an allocation or any general timing bound. They do show that this specific refactor did not provide the intended measurable benefit on the qualified workloads.

## Scientific and IR evidence

The experimental version also passed the **unchanged** complete numerical gate in Slurm **17368**, exit **0:0**, **00:02:31**: 29 ordinary cohorts, five scale cohorts, 34 tail cases, both engines, six precise corrupted-output rejections and the fresh-output guard. `after_science/` retains its independent C++ results, input/compiler/ELF hashes and control receipts. Every selected native stdout file compared byte-identical with the original Slurm 17367 output. Redundant identical numerical rows are referenced by their hashes and comparisons rather than committed a second time.

Slurm **17370**, exit **0:0**, captured the compiler's opt-in `SOUNIO_NV2_IR_TRACE` for the same benchmark. The traced ELF hashes match the corresponding executables used in the memory comparison: before `e7e15899a9ac0f967f2c230a0ef9d0166a7059072639ee45d6c58fcc85b86dde`, after `6d9bf932ebb7af69c6ba1a7dd5f1f5e78c95f128a81552c5ce1498b182212455`. The C++23 allocation census is included with its source.

The trace still contains one `IrAlloc` in each `ln_checked` and `log1p_checked` wrapper, marked `unboxed=1` for its 16-byte value. It does **not** demonstrate that those constructors were eliminated or hoisted. The measurements alone do not establish the ABI/region/lifetime mechanism behind the observed RSS. No such explanation is claimed, and no compiler or reclamation change was made.

The benchmark sources under `benchmark_source/` retain their original `scripts/stdlib` and `tests/reference` layout. To replay the controller, place those two files at the corresponding paths in a full source checkout and supply the explicit preserved baseline checkout and a fresh output directory. The controller hashes both sources, both seed artifacts, the raw compiler, generated programs and its C++ reader; it requires both 100k variants and the experimental 1m variant to complete. An unavailable 1m baseline would be recorded separately and could not support a ratio claim. All heavy execution belongs in an allocated Slurm job, as shown by the retained job script.

No historical source packet was changed or deleted. No quota was increased. Only the uncommitted experimental production edit was restored after the evidence failed to support retaining it.
