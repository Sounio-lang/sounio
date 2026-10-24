<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-inference-scale-2026-10-24.inference.pilot-b31.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-inference-scale-2026-10-24.inference.pilot-b31.readme
-->

# Single-pair B31 pilot, n256, case0

Native job16836 completes both immutable snapshots; final analyzer job16837 recompiles the final C++ analyzer and reuses the original native ELF/stdout/resource receipts without rerunning Sounio. Its output agrees byte-for-byte with the preliminary analyzer report. The first dispatch16834 failed only its guard-receipt path before native execution; that attempt is preserved separately and is not a timing sample.

Both variants have the same4096-count literal, options, seed schedule, n256 and selected case0. B31 and max_profile_attempts64 are an explicitly smaller pilot budget than the intended B199/max400 benchmark. The original B199 data are not replaced. Source/ELF/compiler/manifest identities bind every row. Complete Sounio snapshots and ELF binaries remain on the scratch paths in RUNS.tsv; hashes alone are not a portable full-source replay package.

The final C++ report verifies identical native CAL_RESULT/CAL_P lines: status0,30hits,31completed, no failure, and equal final128-bit RNG. The drivers do not emit uniform/proposal counts; no equality of those unavailable counters is claimed. Before wall154.51seconds/RSS62272KiB; after138.28seconds/RSS60448KiB. The ratio before/after is1.11737055250216951111 for this one pair.

This is operational and functional pilot evidence. It neither establishes a2x improvement nor evaluates the full B199 three-pair performance target. No confidence interval, causal speedup, generic scaling, assessed model assumptions or calibrated inferential validity follows from this pilot. Future source changes require new executable receipts.
