<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-inference-scale-2026-10-24.performance-cache.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-inference-scale-2026-10-24.performance-cache.readme
-->

# Count-profile optimization evidence custody — 2026-10-24

This compact bundle retains text receipts, source/ELF/compiler identities,
workload drivers and their transformations. It deliberately excludes native
ELFs, C++ executables, full stdlib/source snapshots and source archives. Their
recorded hashes and original paths remain in the receipts; hashes here do not
substitute for replaying the excluded binaries. Synthetic numerical inputs are
retained in the driver sources, with no credentials or environment dumps.

## Selected whole-program pilots

Each is one B31/n256/alpha0/case0 pair, compiled with the same compiler92d within
that pair. The data/options/seed checks, native completion, hits/tail/final RNG,
source/input/compiler/ELF hashes and raw GNU time resources are preserved.

- pilot04: original full02 -> strict + replica-null warming + local ranking.
- pilot05: isolated trace-factory removal; both sides already include local ranking.
  It is not an original-full02 comparison and showed no useful resource gain.
- pilot06: original full02 -> combined candidate with objective normalizer cache.
- pilot07: original full02 -> combined candidate with objective and ranking caches.

Do not pool these four distinct candidates into one median. A single pilot is
not the B199 three-pair target or a calibration result. Untraced drivers do not
report uniform/proposal/work counters; those totals remain unavailable.
`RUNS.tsv` and original reports keep their original absolute paths. The copied
`drivers/`, `runs/` and manifests provide a portable receipt view.

`tools/custody_summary.cpp` rechecks the copied resource/status/scientific rows,
source-driver and manifest hash bindings and recalculates ratios in C++23. It
uses the unchanged canonical analyzer09044 parsing helpers. It does not execute
or authenticate the excluded ELFs. Run numerical summaries only under Slurm:

```
g++ -std=c++23 -O2 tools/custody_summary.cpp -o /scratch/UNIQUE/custody_summary
/scratch/UNIQUE/custody_summary BUNDLE_ROOT
```

The original per-pilot C++ analyzer source and logs are also retained. Full
original analyzer replay requires the raw ELFs/sources at their recorded paths.
Normalization-phase summaries belong to a different earlier experiment and are
not mixed into this count-profile evidence.

## Functional and discriminating controls

- 16912 local-ranking checks/metadata, 61+3 independent comparisons, B2/RNG.
- 16929 additional extreme finite/subnormal/count-boundary checks.
- 16931 checked trace-control route and canonical CountGLM/increment comparisons.
- 16936 objective-cache metadata bit equality, 216-case likelihood, P0, paired/B2.
- 16941 wrong alpha-key and missing reset invalidation: compiled0/native1 controls.
- 16954 shared-core/cached fallback: 13 native fixtures, 61+3/216/P0/B2 C++ and
  byte-identical legacy protocols against the previous frozen candidate.
- 16965 independent eight-case far PMFs and compiled0/native1 controls for wrong
  second alpha and incorrectly zero PMF calls on normalizer hits.

The named cache-counter scope changed between versions: objective only in16936,
then objective plus cached ranking fallback in16954. Never compare those totals
as the same scope. Original ten trace metrics and public snapshots stayed stable.
Final uncached bootstrap comparisons and sampler PMFs are outside named cache counts.

## Preserved failures and compiler distinction

- Original trace_focused02/16872 compiled the traced fixture but native assertions
  failed. That packet also contains an unrelated wrongly routed C++ invocation
  against an observed/preflight fixture; its failure is operational, not a
  numerical verdict on the full B2 route.
- Corrected-original16899 is the accepted unannotated program with compiler92d,
  native/C++ B2/off-on checks0.
- The annotation-only control16889 compiled then failed native1; it was not a fix.
- Borrow diagnostic16885 showed stable correct alpha through a typed borrow while
  direct field projection was wrong. It diagnoses a compiler projection defect,
  not numerical memory corruption. No materialized incorrect-alpha workaround is
  accepted as a scientific fix.
- B199 launcher16977 built both ELFs but srun could not exec relative `bash` and
  never ran a numerical preflight. It is not a failed numerical B199 case.

B199 final02 runs are separate pending/future evidence, never reused from the
preflight. The control92d benchmark is separate from current-compiler CI work.

`ORIGINS.tsv` maps copied receipts to original paths. `FILES.sha256` closes the
bundle when finalized. Original raw packets remain intact on scratch. Repository
source, focused gates, resource pilots, B199 comparison and CI/merge acceptance
are distinct claims.

## Compact payload and restore protocol

Raw collected directories are now inside `RAW_RECEIPTS.tar.gz`, a deterministic
GNU tar (`--sort=name --format=gnu --mtime=@0 --owner=0 --group=0 --numeric-owner`)
compressed with `gzip -n -9`. The archive contains original bytes, including the
opaque `.md.raw` documentation. Loose receipts were removed only from this new
repository copy after every archived regular file passed a restored SHA check.
Original scratch receipts and the original collection manifest were untouched.
`RAW_RESTORE.verify` records that check; `COLLECTED_RESTORE.verify` records the
original collection check with the two Markdown-to-`.md.raw` path mappings.

Restore to a new empty directory, check `ARCHIVE.sha256`, extract, and verify:

```sh
sha256sum -c ARCHIVE.sha256
mkdir /scratch/UNIQUE_RESTORE
tar -xzf RAW_RECEIPTS.tar.gz -C /scratch/UNIQUE_RESTORE
(cd /scratch/UNIQUE_RESTORE; sha256sum -c /ABSOLUTE/BUNDLE/RAW_RECEIPTS.sha256)
```

Run the portable C++ receipt summary under Slurm against that restored directory,
not against this compact top-level directory. The two needed C++ sources remain
human-readable under `tools/`; compilation output belongs outside this bundle.
`COLLECTED_INPUTS.sha256` is the unchanged historical collection manifest;
`RAW_RECEIPTS.sha256` closes the extracted payload; `FILES.sha256` closes this
compact repository view. None of these substitutes for authenticating or
replaying the excluded original numerical executables.
