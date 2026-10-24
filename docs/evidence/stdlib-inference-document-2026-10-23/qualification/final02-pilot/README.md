# Final02 pilot: executed qualification harness, not general calibration

Slurm job16488 used the final frozen source of the successful full02 functional
gate and compiler SHA256
`ee0f341e6e710d35c162084023cbb10ce72d2b7ca29726ea6e3a5994f080eded`.
The job reserves22h for the new pilot and subsequent qualification together;
this bundle records only the completed pilot. The B199 campaign was still in
progress when the bundle was prepared. It has no completed-result claim here.

## Observed pilot result

The pilot used n16, supplied generating alpha0.25, null mean8, B31 and16 fixed
observed datasets. Each dataset ran in its own process using one ELF compiled
for the batch. All16 processes completed and all16 bootstrap aggregates were
available. The independent C++ summary recorded0 rejections at the declared
0.05 threshold and an approximate Wilson95 interval
[0,0.193607680534436522]. This small pilot does not establish nominal type-I
error, general calibration, exact finite-sample validity or model adequacy.
The endpoint empirical MC SE of0 is not a statement of certainty.

The metadata/prefix C++ report verified all22 scientific lines retained from
the earlier failed serialized pilot byte-for-byte. That earlier run terminated
with exit181 after11 complete datasets; it was not classified as a Slurm OOM.
The new pilot completed all16. The maximum observed single-process RSS was
16000KiB. Individual elapsed times and RSS appear in
`PILOT_RESOURCE_AND_PREFIX.tsv` and the16 GNUtime receipts. These are process
measurements, not isolated-kernel benchmarks or a memory bound for B199/n256.
Both private hot-allocation changes and process isolation were present; this
bundle does not attribute the whole memory change to either one in isolation.

## Strict functional prerequisite and normalization

The control rejects full01 STATUS1 with exit65 before emitting a receipt.
The successful full02 gate is retained in full as RESULTS.full.tsv. Every row
and exit status survives normalization; LABEL_MAPPING.tsv records old/new labels.
Reconstructing the original table from that ledger produced byte-identical
RESULTS.reconstructed.tsv. No failed row is recategorized as success, and no
subset of full01 was accepted.

The seven-field FUNCTIONAL.receipt binds the selected compiler, a canonical
frozen-source manifest and the normalized complete result table. The main
harness additionally checked required native/marker/concordance rows and the
fixed-alpha mutant rejection. This is a consistency record controlled by the
coordinator, not a signed attestation or a substitute for native outputs.

## Custody and omissions

COPIED_FILES.sha256 covers every file copied in this compact bundle, including
this README, and was verified after transfer to the workspace. Gzip members
preserve complete uncompressed manifest bytes; their decompressed contents were
compared to the original files before transfer. The strict receipt keeps the
original absolute runtime paths intentionally; these are not rewritten into a
new execution receipt.

The original pilot INPUTS manifest, compressed here, includes the selected
compiler and receipt association. ORIGINAL_BATCH_FILES.sha256 identifies the
full original batch, including omitted files; it is not the manifest of this
compact bundle. ELF-before.sha256 and PILOT_PROOF.sha256 retain omitted binary
identities. No driver source, ELF, compiler, cache or temporary source tree is
copied here. Reproduction uses the tracked generator and its fixed schedules;
datasets.tsv records parameters and seeds.

Complete original artifacts remain under
`/scratch/sounio-stdlib-advanced-20261023/calibration_final02/`.
The original baseline transcript remains under
`/scratch/sounio-stdlib-advanced-20261023/qualification01/pilot_B31/n16_alpha.25/batch0/`.
Native record/schema failure leaves a batch partial and unpublished. Numeric
bootstrap unavailability remains a counted result rather than a discarded
observation. Source preparation, this B31 pilot, B199 execution, full-corpus
acceptance, CI and merge are separate claims.
