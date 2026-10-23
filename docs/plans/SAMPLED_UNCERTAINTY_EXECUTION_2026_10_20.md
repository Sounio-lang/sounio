<!-- docs:meta
topic_id: repo.docs.plans.sampled-uncertainty-execution-2026-10-20
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.sampled-uncertainty-execution-2026-10-20
-->

# Sampled uncertainty — native SVG execution, 2026-10-20 UTC

## Result and exact scope

**Slurm 15776 COMPLETED, exit 0:0.** Eighteen selected source targets returned
`TYPECHECK_OK`. The example compiled to a native ELF and ran with exit 0,
producing complete HTML containing the SVG. The compiler hash was verified before
and after typechecking. No tests were added or run for this increment.

The 18 targets comprise nine library files (`uncertainty`, `epiviz`, `ir`,
`viz_canvas`, `viz_html`, `workbench`, `viz_app`, `viz_replay`, `viz_window`)
and nine example entry points (`viz_physchem_demo`, `viz_lab`, `viz_lab_window`,
`pbpk_dashboard`, `viz_hello`, `viz_html_export`, `viz_workbench`,
`viz_nb_dispersion_diagnostic`, `viz_sampled_uncertainty`). Only the last example
was compiled and executed to produce the figure in this receipt.

Forge's `corpus-smoke` profile served as transport. A package-only replacement of
`scripts/run_sio_test_suite.sh` performs the explicit check/compile/figure steps.
Generic `check_corpus` / `corpus_smoke` result labels do not establish a corpus run.
The canonical worktree test harness was unchanged.

## Source, compiler and output identities

| Item | SHA / identity |
|---|---|
| Frozen source commit | `52ba99b87d49da04e97524fa0d8911b1d1d4d46b` |
| Incorporated remote main | `085da1da15f3c05946b66e4ddf7b60a0e29d5d40` |
| Compiler build origin commit | `ea8c720823797638ec15887b4e0bafd29987cb0a` |
| Source snapshot SHA256 | `bb02cff9d76f43e2adf9f480f989ec5c5b57883cd7c80a6e8dbe20f0f6836110` |
| Package input manifest SHA256 | `b0186cc2032e6238834e45c3c2e0100ed16c82320918144403080f187b293e88` |
| Current-source Madaros SHA256 | `0c896942018ff4be9a86d69f30e3a8aadac955f32451c445a07fbf502ae37fc3` |
| Example source SHA256 | `a94ce68325443744db8cd500f4f9f5550e8fffe13279d910814b03c5ae8023a7` |
| Native program ELF SHA256 | `19a14654a09c652764a23b40785634234292f7a903f7ec793ea5382d443ef59a` |
| Native HTML SHA256 | `82b72b8c718a8ab2743c038d289f4c704f4bb541f6e49a3a5cbf2ad4015f614e` |
| Extracted SVG SHA256 | `08929f84ec1551960a2c2f8ecb92cbb74771af6b963564f3265c9019a1368eb5` |
| Final raw check/compile/run log SHA256 | `1dfcfebc6267d4bef243c58f04f30b608d65a54099d03e016aef7cc8fae65a12` |

Remote main was read back with `git ls-remote` after execution and still matched
085da. Final documentation/evidence commits follow the frozen executable source.

### Attested compiler reuse

Job15772 rebuilt the source-tracking bootstrap seed and modular Madaros from the
incorporated current source. Before worker cleanup, a watcher read the live ELF
through the job's private mount namespace using its UID, compared its hash with
the build log, streamed it to t560 ext4 custody, and verified the copied hash.
The compiler was subsequently supplied as an explicit compressed input; each
reuse job decompressed and verified that exact executable, without a fallback.

Submission rejects any diff between origin ea8c720823 and the selected commit
under `bin`, `scripts`, `self-hosted` or `stdlib`. These Git tree identities are
recorded in SOURCE.txt. Only example signatures and the demo axis interval count
changed. Job15776 **reused this attested current-source compiler**; it did not
rebuild the compiler itself. The origin log, custody identities, exact dispatchers
and submission scripts are retained in the evidence directory.

## Attempts retained, including failures

| Job | Outcome | Evidence boundary |
|---|---|---|
| 15771 | FAILED 1:0 | Compiler build + six library checks passed; first example lacked Panic |
| 15772 | FAILED 1:0 | Compiler rebuilt + fourteen checks passed; demo returned 1 before HTML |
| 15775 | FAILED 1:0 | Attested compiler reused + eighteen checks passed; same demo rejection |
| 15776 | COMPLETED 0:0 | Eighteen checks, native compilation and figure execution passed |

The first failure exposed missing effect propagation. All known non-test
production/demo consumers now declare required Panic effects. The native demo
failure was its X-axis request: four intervals need at least 448px by the existing
112px-per-interval label contract, but its axis width was 400px. The final source
uses three intervals. No compiler or axis contract was weakened.

The failed v2/v3 program had the same ELF hash in both attempts:
`bc0fc272e9221b60aeea6270445185305f8e7c0148ad9071cb8bfd2ab74b86a0`.
Their exact source/compiled identities and diagnostics remain retained. Their
empty output is not presented as a figure.

## Actual figure inspection

The native HTML was downloaded and its complete SVG element extracted without
changing its interior; one trailing newline was added to the standalone SVG.
`xmllint --nonet --noout` accepted that SVG with exit 0. macOS Quick Look rendered
it and the resulting preview was opened for visual inspection.

Inspection found visible sampled uncertainty marks and center line, legible
caption/axis labels, preserved negative Y-axis labeling and no clipping of the
observed labels. The actual markup contains uncertainty input/span/sampling
metadata, plot clip geometry and the parity pattern. The caption identifies
unitless synthetic data. This is structural and visual artifact inspection;
it is not an independent numerical oracle, pixel-parity comparison, scientific
coverage assessment or a general rendering regression suite.

Canonical outputs:
[HTML](../evidence/sampled-uncertainty-2026-10-20/figure/sampled-uncertainty.html),
[SVG](../evidence/sampled-uncertainty-2026-10-20/figure/sampled-uncertainty.svg).
The compressed native ELF and program identities are retained beside them.
[Raw receipts and scripts](../evidence/sampled-uncertainty-2026-10-20/README.md).

## Resources and storage

Final Slurm elapsed 00:00:26, TotalCPU 00:25.413, batch MaxRSS 2490704K.
These describe the combined checks, compilation and example execution with a
reused compiler. They are not renderer/kernel latency or RSS measurements.

The preserved compiler is 129712243 bytes; its gzip input is 12751795 bytes.
The final source snapshot is 28866279 bytes. Compiler custody and prepared source
packets reside on t560 ext4, outside the shared training quota. The last recorded
training availability was 692060160 bytes within the unchanged 3TiB quota;
concurrent writers may change it. No further artifacts were deleted.

Source changes bound Canvas fill/line traversal to plot/raster intersections,
avoid an explicit whole-series copy in the band dispatch and add no lower/upper
IR buffers. SVG reuses a small parity pattern with per-sample rectangles.
Measured renderer speed/RSS comparisons remain pending.

## Remaining acceptance and next implementation

Independent C++23/Sounio numerical concordance, edge-case runtime coverage,
Canvas execution, cross-backend comparison, inferential calibration, full corpus,
broader CI, PR/merge and human scientific acceptance remain separate open states.
Historical114/114 renderer acceptance has not been extended by these typechecks.

The mass sprint retains statistics, scientific rendering, mathematics, genomics,
physics and execution/memory/artifact scope. The next concrete candidate is an
allocation-free NB2 count diagnostic conditional on supplied means/dispersion,
with explicit Poisson limit and representability status, connected to the existing
count/offset consumer. It is proposed, not implemented or accepted by this receipt.
