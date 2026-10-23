<!-- docs:meta
topic_id: repo.docs.plans.nb2-diagnostics-execution-2026-10-21
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.nb2-diagnostics-execution-2026-10-21
-->

# NB2 supplied-parameter diagnostics — execution, 2026-10-21

## Delivered implementation

`stats::regression::nb2_diagnostics::nb2_count_diagnostic` now supplies scalar
Poisson/NB2 Pearson residuals and independently reports staged variance availability.
It takes a count, expected count mean and supplied dispersion. Accepted integer
counts are0..2^53 inclusive; larger values are refused before conversion.

For positive alpha it evaluates the NB2 residual through scaled branches,
avoiding the need to construct the raw variance first. Alpha zero copies the
already computed Poisson residual. Invalid inputs and unavailable arithmetic have
explicit statuses; a failed variance evaluation does not discard valid residuals.
The kernel has scalar storage and no explicit heap allocation. See the
[contract](../../stdlib/stats/regression/NB2_DIAGNOSTICS.md) for the staged
binary64 rounding policy and precise status meanings.

The existing count/offset example now plots Poisson and plug-in NB2 residuals on
a shared axis, retaining observed/fitted rates in its left panel. It uses the
projected moment alpha from the same observations and labels that construction.
It reuses the existing two1024-element scratch arrays. Unavailable paths return1;
zero follows Canvas rendering, HTML emission and allocation release.

## Executed scope

**Slurm15783 COMPLETED0:0**, elapsed00:07:08. A source-tracking bootstrap seed
and modular Madaros were rebuilt, with isolated worker scratch/cache and no
fallback to a stale workspace ELF. Four explicit source targets returned
TYPECHECK_OK:

- `stdlib/stats/regression/nb2_diagnostics.sio`.
- `stdlib/stats/regression/nb_dispersion.sio`.
- `stdlib/stats/regression/poisson.sio`.
- `examples/viz_nb_dispersion_diagnostic.sio`.

The example was compiled to a native ELF and executed with exit0. Its output
reports successful Poisson fit/moment statuses and contains complete HTML/SVG.
The source path calls the Canvas renderer in memory before SVG emission; no
Canvas pixel capture or cross-backend numerical/pixel comparison was performed.

Forge used the corpus-smoke transport with a package-only replacement of
scripts/run_sio_test_suite.sh. The canonical worktree harness was unchanged.
The package builds/checks/compiles/runs this demonstration; generic check_corpus
or corpus_smoke result labels do not establish a corpus run. No tests were
added or run for this increment.

## Source and executable identity

| Item | Identity / SHA256 |
|---|---|
| Frozen source commit | `4a65e767af7ce955cfde85f25484ea7f8a5f44f2` |
| Incorporated and final observed remote main | `085da1da15f3c05946b66e4ddf7b60a0e29d5d40` |
| Source snapshot | `59f77117ae3c456f4952d489bf55ca67cd9dc91dd591801c95d509bc76c57f51` |
| Package input manifest | `e7e58734c1db2cb11c937947e01c5c75bae0ffbe2fafba3ffe5a11b76e2f5d1e` |
| Rebuilt Madaros | `0c896942018ff4be9a86d69f30e3a8aadac955f32451c445a07fbf502ae37fc3` |
| New kernel source | `091b5ee0e741b6c2c7e61d252f65b448745dc28c34bc926afc6b4d7e3b3dc1e6` |
| Example source | `be8a3123522b9dec6a693fc1bd77af313ad3e781adc692c835d5dc4cbd683cd7` |
| Native example ELF | `694a9bab24cde523bced34c36a67d5070635b601c04da575cc22ee1458fc8a13` |
| Native stdout, report plus HTML | `c4638d32173f59987360db4d33fae97f903ca71e1fe884ba44e8ac9a6709f9bc` |
| Extracted clean HTML | `4d27181571ef332e2982983db64b426f337bdb4ea1801b8e78e8eb70177c6da3` |
| Extracted SVG | `0eddee70af81a88fb09701a8983086cb24cbc9ee651f16b9780fd0e8347a37ae` |
| Original build/check/run log | `aef6ec5cb2a63feb318532cc42df173cc91a548a32356749ca4efdb1c1d0748d` |

The new stdlib tree received a fresh compiler build. Its resulting ELF happens
to match the previously preserved compiler byte identity. The live t560 ext4
custody hash was read back and matched it; another large compiler copy was not
added to Git. A compressed output of the new build was preserved before checking
consumers. Documentation/evidence commits follow the frozen executable source.

## Actual figure inspection

Stdout contains a textual report followed by the HTML document. The report prefix
and clean HTML were separated; the complete SVG element was extracted with its
interior unchanged and one trailing newline. The SVG passed
xmllint --nonet --noout with exit0.

Quick Look's square thumbnail clipped the wide figure and was not used to accept
its complete layout. macOS sips converted the unmodified SVG into a1380x620 PNG;
that complete image was opened and inspected. Both panels, the shared residual
axis, the two residual colors/legends, rate/residual units, negative labels and
the plug-in/assumption caveats were visible. This is artifact inspection, not
independent numerical agreement or model adequacy assessment.

[SVG](../evidence/nb2-diagnostics-2026-10-21/figure/nb2-diagnostic.svg),
[HTML](../evidence/nb2-diagnostics-2026-10-21/figure/nb2-diagnostic.html),
[PNG preview](../evidence/nb2-diagnostics-2026-10-21/figure/nb2-preview.png),
[actual report](../evidence/nb2-diagnostics-2026-10-21/figure/nb2-diagnostic.report.txt).

## Native process resource observation

**Slurm15785 COMPLETED0:0** executed the exact same ELF after decompression and
SHA256 verification. The dispatcher performs no program compilation.
GNU time measured the native child process separately:

| Metric | Observed value |
|---|---|
| Elapsed seconds | 0.01 |
| User seconds | 0.01 |
| System seconds | 0.00 |
| Maximum RSS, KiB | 8748 |
| Program exit | 0 |

This is one execution of the fixed eight-observation program, including mean
fitting, moment dispersion, residuals, Canvas rendering and SVG emission. It is
not isolated kernel timing, a scaling study, a comparative benchmark or a claim
of improved performance. The timing has centisecond resolution.

Slurm whole-job accounting is separate: the build/check/native job records
TotalCPU07:04.673 and batch MaxRSS3476392K; the profile transport job records
elapsed00:00:03, TotalCPU00:02.285 and batch MaxRSS1482220K. Those groups include
compiler/build or orientation/transport work. They do not replace the timed
native child's8748KiB observation.

## Artifact size and retention

The source snapshots are24156168bytes (build) and24477589bytes (profile).
The generated compressed compiler is12751794bytes and stays in remote custody.
Prepared source packets and the evidence staging directory live on t560 ext4.
The retained Git evidence uses losslessly compressed raw logs/input manifests;
original uncompressed SHA256 values remain alongside them. Program ELF/output,
metadata and full-preview derivatives are retained with relative file hashes.

The final recorded training availability is629145600bytes within the unchanged
3TiB quota. Other writers may change it. No further snapshots were removed and
no quota change was made. [Retained evidence index](../evidence/nb2-diagnostics-2026-10-21/README.md).

## Open scientific and integration states

Independent C++23/Sounio agreement and explicit extreme/boundary runtime coverage
remain pending. Staged variance is not certified correctly rounded, and no formal
float-error theorem is supplied. This implementation supplies no deviance, NB
likelihood fit, shrinkage, CI, test or differential-expression inference.
Same-data plug-in scaling does not assess independence, exposure adequacy, the
mean model, dispersion calibration or predictive coverage.

No full corpus, broader compiler CI, PR/merge, Canvas pixel agreement or human
scientific acceptance is implied. Earlier114/114 rendering acceptance is unchanged.
The mass sprint retains statistics, scientific rendering, mathematics, genomics,
physics and measured execution/memory/artifact work. Stable deviance primitives,
genomic count consumers and domain calibration remain subsequent work.
