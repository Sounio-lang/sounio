<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-inference-scale-2026-10-24.combined-cache-preguard.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-inference-scale-2026-10-24.combined-cache-preguard.readme
-->

# Historical combined gate16994 — before aggregate availability guard

This is a compact historical receipt for Slurm16994 only, not the final source
acceptance. Its immutable source is
`/scratch/sounio-stdlib-inference-scale-20261024/combined_final_cache01/source`,
with original short SOURCE_HEAD `257fb0d0a2`. The original SUBMISSION.txt copied
here is that run's SUBMISSION_GUARD_RETRY.txt, naming job16994. Earlier superseded
submissions are deliberately not copied or reclassified.

The run completed with STATUS0 and Slurm COMPLETED0:0. RESULTS.tsv preserves all
normalized gate outcomes exactly; status0 in a negative/mutant row means that the
expected rejection was observed, not that invalid source compiled successfully.
The final input-hashes and compiler-hash rows are unchanged0, with their original
verification receipts retained. The compiler was the guarded current-source ELF
`d74fbebe0ff442e8c77bec024d6d92325b83a98d3f1487caa1db59ef171a41e6`;
its separate build/guards/fresh-selfcheck custody is in ../compiler-effectbridge.

This source predates the conservative aggregate-cancellation guard. A subsequent
adversarial audit found an available rounded-zero aggregate despite a nonzero
exact rational contrast; see ../aggregate-cancellation. Therefore16994 passing
its selected corpus does NOT close that defect or establish final-source
correctness. The later source includes that guard and its explicit availability
policy. Final CENTRAL17009 receipts are maintained separately by the coordinator;
they are not duplicated, anticipated or accepted by this historical bundle.

## Retained evidence and limits

- Exact RESULTS/STATUS, compiler identity and verification, input manifest and
  verification, original job script, source-head metadata and actual submission.
- Existing per-program resource receipts, without recomputing summaries or
  attributing their figures to isolated kernels. SLURM_ACCOUNTING.tsv is an
  observed scheduler record; its AllocCPUS is retained literally, independently
  of the job script's requested CPU count.
- Existing per-program identity files bind source, native ELF and stdout hashes.
  Complete stdout, stderr, source trees, ELFs and caches remain in scratch.
  No full scientific stdout is repeated here.

No numerical recalculation, native execution or new performance/calibration claim
was made while preparing this folder. SOURCE_MAP.tsv records original receipt
paths, ORIGINAL_FILES.sha256 their original byte hashes. Large text manifests are
losslessly gzip-compressed, so source-map paths precede optional `.gz` suffixes.
FILES.sha256 is the authoritative copied-file manifest for this folder only.
