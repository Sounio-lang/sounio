<!-- docs:meta
topic_id: repo.docs.plans.scientific-main-integration-execution-2026-10-23
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.scientific-main-integration-execution-2026-10-23
-->

# Scientific stdlib reconciliation with main — 2026-10-23

## Identity and scope

Branch: `sprint/scientific-integration-main-3ef3db-20261023`.
Remote main incorporated: `3ef3dbaa6e` (only public website translations changed
since the prior a59930 checkpoint). Frozen integration source:
`19c684a99bb5728c15b424a13fd190742efad718`.

The merge brings the later scientific branch5c84f28288 into the current main-based
continuation, preserving the observer/analyzer and all historical receipts. It
resolves real overlapping source additions rather than replacing the whole stdlib
with either side. The compiler source and committed bin tree are byte-identical
to incorporated main; the selected stdlib source differs and needs fresh execution.

## Reconciliation decisions

Three assigned editing fronts covered statistics, multidomain kernels, and
rendering/packaging. Root owned documentation and Git. Independent cross-review
then compared the other domains. Source reviews did not run numerical oracles.

- Retain main's explicit `beta_new` import in linear regression and `sw_exp`
  wrapper/callers in Shapiro; those two files exactly equal incorporated main.
- Retain main's Boltzmann entropy rewrite from block-tail expression to direct
  declarations/loop; thermo exactly equals incorporated main.
- Retain later shared logarithmic kernels and the migrated probability/channels,
  genomics normalization, entropy and Poisson callers. Genomics preserves integer
  subtraction before conversion. Entropy preserves equal/near/far dispatch.
- Retain NB moment availability under aggregate rather than per-row underflow,
  exact zero-count Pearson contribution, NB2 residual/deviance primitives and
  their separate status contracts.
- Retain signed bars, shared rectangle/viewport geometry, sampled uncertainty,
  direct scalar forest access and Panic propagation through rendering callers.
- Union the packager selection: main's NB dispersion reference/status artifacts
  plus the later bar references/fixtures. Packaging preserves these existing
  artifacts; no fixture or reference executable is run in this checkpoint.
- Keep governance metadata on historical docs and restore later scientific
  continuation entries. Preserve the prior Main14 attribution checkpoint too.

`SCIENTIFIC_RESOLUTION_MAP.tsv` records exact byte comparisons with both parents.
The complete historical branch remains an ancestor of the reconciled source.

## Execution protocol

Compiler and examples execute on the Slurm cpu-ops worker as uid/gid1000, using
the canonical builder and compiler wrappers. The packet contains the current
selected bin/scripts/self-hosted/stdlib trees plus five real examples. Its sole
script replacement is the explicitly identified packet-only dispatch at
`scripts/run_sio_test_suite.sh`. Forge's corpus-smoke label is transport only;
there is no test-suite/corpus acceptance claim. Requested23 typechecks include
libraries and example consumers; five existing examples are compiled and run.

Compiler/cache/temp/native outputs use private `/scratch` ext4 storage. The
canonical broad key, no-cache/read-only guards, environment identity checks and
3TiB shared quota remain in force. No historical removal was performed.

### Attempt1 — integrity guard correctly stopped incomplete transport

Slurm15990 failed1:0 after6seconds before starting the compiler build. The packet
manifest exposed missing `stdlib/data` members. The Forge create_snapshot path
uses `--exclude='*/data'`, which also excludes source directories of that name;
the packet itself and its declared manifest were complete. This is a transport
omission, not a source/compiler typecheck result. Preserve its logs and status.

### Exact snapshot correction

Build a gzip tar of the already selected packet with no directory-basename
exclusions. Extract it into a new private directory and check every declared
regular-file hash before accepting it. Keep this archive and its identities in
private custody; place one immutable copy on the shared runs filesystem and use
Forge `--reuse-snapshot` for attempt2. No global Forge filter was silently changed.
The attempt2 packet dispatch uses a fresh run2 scratch directory; source commit
and selected scientific/compiler files remain19c684a99b.

Detailed results, resource receipts, artifact identities and figure inspection
are recorded in the completed execution addendum below.

## Completed execution — Slurm15992

State COMPLETED0:0; elapsed00:07:37, TotalCPU07:31.519, batch MaxRSS3588712K.
Canonical fresh builder exit0; resource receipt: elapsed402.87s, user397.15s,
system1.53s, maxRSS736380KiB. These describe this run; no comparative speed or
memory improvement is claimed. Compiler SHA-256:
`238c7e210d82cdd8474f68c488a0aed2b22d3157f64536703f03e1d995dce734`.

All23 declared library/example typechecks exit0, including linear, Shapiro,
thermo, shared logs, NB moment/Poisson/NB2, both rendering backends and consumers.
All five native example compilations/executions completed successfully. The
complete selected regular-file manifest matched before build and after examples;
this is endpoint equality, not a complete input-consumption or immutability proof.

| Actual example | Native exit | Max RSS KiB | Wall seconds (reported resolution) |
|---|---:|---:|---:|
| NB2 deviance | 0 | 1052 | 0.00 |
| NB dispersion diagnostic (includes Canvas path) | 0 | 8768 | 0.01 |
| Logarithmic primitives | 0 | 1064 | 0.00 |
| Genomics/entropy | 0 | 1060 | 0.00 |
| Sampled uncertainty | 0 | 572 | 0.00 |

0.00 is the time tool's centisecond reporting resolution, not zero runtime.
The compiler-build RSS, whole-job batch RSS and native-example RSS are distinct
measurements and are not interchangeable.

### Actual figure artifacts

The five raw native stdout streams are preserved and hashed. The extraction
recipe copies the report, HTML and SVG byte ranges without XML reserialization.
All five actual SVG files passed `xmllint --noout` with exit0. Full-aspect `sips`
PNG rendering and root inspection completed; independent visual inspection also
reported no blocking composition/legibility defect. Dimensions: sampled
uncertainty640x400, NB2 deviance1040x620, other three1380x620.

Two NB2 curves and zero reference are distinguishable; both NB diagnostic panels,
legends, negative ticks and units are visible. Genomic/entropy panels and their
scientific caveats remain legible. The logarithmic normalized quadratic curves
largely overlap, with separate legends. Sampled uncertainty shows patterned
strips at sample positions and a central line with an explicit +/-1sigma caption.
The observed zero-rate scatter glyph is partially clipped at the closed lower
viewport edge according to the existing scissor contract; the sample is visible.

This is actual render inspection, not pixel equivalence, numerical concordance,
inferential calibration or a bar-native acceptance result. No new test suite or
reference/fixture executable was run. The later bar implementation is integrated
and typechecked through bar/IR/Canvas/HTML; its historical native receipts remain
at their original source/compiler identities.

## Evidence custody and storage

[Compact public evidence](../evidence/scientific-main-integration-2026-10-23/README.md)
retains results, both attempt receipts, compressed logs and packet manifest,
selected compiler-input hashes, source resolution map, extraction/submission
recipes and five actual SVG/HTML/report artifacts. Raw stdout is compressed;
compiler/program binaries and the full source snapshot are not copied into Git.

Private t560 custody:
`/home/devsounio/sounio-artifact-archive/scientific-integration-20261023/final`.
It includes complete source-bound execution artifacts and file-hash readback.
The exact complete snapshot has24993425bytes and SHA-256
`6fcf0ac60178b34aa01fc4f72df1a2896eda649aecae83c183247cc6e105a92c`.
Its immutable shared copy is referenced by reuse-snapshot; no third source copy
was generated for the retry run. The original filtered snapshot is retained with
its failed-attempt evidence. Private custody remains host-local, not an independent
physical backup. Shared free space observed after submission was12582912bytes;
this is a point observation, not guaranteed capacity for the next sprint batch.

## Remaining acceptance boundaries

This closes the previously queued scientific-source reconciliation with main,
fresh modular build, selected23 typechecks, five native demonstrations and their
SVG inspection. It does not certify all stdlib APIs, complete corpus/CI, fixed
point, numerical edge-case concordance, inferential calibration, or merge to main.
Canonical broad cache authority and the3TiB quota are unchanged. Further work
continues with independent numerical concordance and broader scientific coverage,
without treating these descriptive examples as full inferential models.
