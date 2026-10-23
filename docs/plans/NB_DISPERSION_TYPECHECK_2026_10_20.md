<!-- docs:meta
topic_id: repo.docs.plans.nb-dispersion-typecheck-2026-10-20
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.nb-dispersion-typecheck-2026-10-20
-->

# NB2 count diagnostic — current-source typecheck, 2026-10-20

## Result and exact scope

**Slurm 15747 COMPLETED, exit 0:0.** The package rebuilt the source-tracking
bootstrap seed and modular Madaros, then checked exactly these two files:

- `stdlib/stats/regression/nb_dispersion.sio`: `TYPECHECK_OK`.
- `examples/viz_nb_dispersion_diagnostic.sio`: `TYPECHECK_OK`.

The final marker is `CURRENT_SOURCE_TYPECHECK_READY`. The compiler hash was
verified unchanged after both checks. The example was not compiled to a program
ELF or executed. No numerical concordance, SVG/Canvas output, tests or corpus
were run. Historical 114/114 bar acceptance remains a separate checkpoint.

Forge used the `corpus-smoke` transport with a package-only replacement of
`scripts/run_sio_test_suite.sh`; its body builds the compiler and invokes only
`bin/souc check`. The generic results.tsv phase/layer names `corpus_smoke` and
`check_corpus` do not establish corpus acceptance. The canonical worktree harness
was not modified. The exact dispatcher and submission script are retained.

## Source and artifact identity

| Item | Identity |
|---|---|
| Source commit | `9285bb2f118983282256035ef1a9df3a13dfb12a` |
| Incorporated remote main | `b10c046e7b003c4346d5f2d97f47f380081cf06a` |
| Source snapshot SHA256 | `44c3fa4b07ac435bd6519dbfaacceceb60441a3dae6316ae9ebbe05a8e79213b` |
| Package input manifest SHA256 | `a04fb29842d05cefadff1c1175cbbad2f3d34fd561ed71b499a4d4fbc870ed2f` |
| Current-source Madaros SHA256 | `bbd910f0e21312922a28947d7a1e8d8d387fb85f7c4612e4f78cb98659b6d7c0` |
| Kernel SHA256 | `54d46840a14684ca36a9c01972b59dc11b02269cfcb3648c0e9831ac7d993381` |
| Consumer SHA256 | `13323042719164299b376b70c4747568e19c91912969045f5a645bb2c63cb2e4` |
| Raw build/check log SHA256 | `8b23a553cee1f02da1e690b22e6f1cfd7729e9038a3d39d0f2484db2e568aa05` |

Run: `nb-current-source-typecheck-20261020`.
Hot receipt root:
`/var/lib/cephfs-slurm/training/sounio-foundry/runs/nb-current-source-typecheck-20261020`.
Source packet retained outside the shared training quota:
`/home/devsounio/sounio-artifact-archive/prepared/nb-current-source-typecheck-20261020`.
Compiler ELFs, temporary files and the isolated cache were generated under worker
scratch, without copying their large binaries into Git or the training receipts.

## Recorded resources

Slurm elapsed: `00:06:44`; TotalCPU: `06:41.671`; batch MaxRSS: `1516432K`.
The package phase records 401 seconds. These describe the complete compiler
build/check job, not the statistical kernel's or renderer's runtime/memory.

The last `df -B1` observation after this job showed 5,754,585,088 available bytes
within the unchanged 3 TiB training quota. Other writers remained active; the
previous offload's 10,733,223,936-byte observation is historical, not a reservation.

## Remaining work

Independent C++23/Sounio numerical concordance for mixed tiny/ordinary means,
zero-count subnormal Pearson contributions, all-rounded-zero weights, declared
parameter limits and conversion boundaries; discriminating legacy controls;
actual example execution and SVG/Canvas inspection; empirical calibration and
resource comparisons remain open. Source/static review and typecheck do not
supply those results. Broader compiler CI, PR/merge and scientific acceptance
remain separate states.
