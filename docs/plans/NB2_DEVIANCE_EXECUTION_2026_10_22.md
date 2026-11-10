<!-- docs:meta
topic_id: repo.docs.plans.nb2-deviance-execution-2026-10-22
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.nb2-deviance-execution-2026-10-22
-->

# NB2 deviance and signed residual: execution receipt, 2026-10-22

## Remote source and ownership

Remote main advanced from085da to `06f927dbca5753f5d7b6f64c29eb745eff77f351`.
It was fetched and merged into the sprint, retaining all previous increments.
Branch: `sprint/nb2-deviance-main-06f927-20261022`.
Initial implementation: `70723f4f786e491374833ccee6819f46060ee396`.
Final executable source: `76f30851adaace93d0dfc5ff4c0d3b8df3741973`.
Subsequent documentation/evidence commits are not relabeled as executed source.

The new scalar `stats::regression::nb2_deviance::nb2_deviance` returns a signed
residual and independently staged squared-deviance availability. Existing Pearson
and variance diagnostics remain independently callable. It follows the existing
validation order and exact count bound2^53. No module export layer exists or was
invented; imports use the established direct module path.

## Algorithm and scientific contract

Mean and dispersion are supplied. No regression/dispersion fitting or inference
is performed. [Module contract](../../stdlib/stats/regression/NB2_DEVIANCE.md)
describes the saturated-deviance definition, positive decomposition, compensated
32-term local series, scaled roots, validation and unavailable regions. The
[official statsmodels family source](https://www.statsmodels.org/stable/_modules/statsmodels/genmod/families/family.html#NegativeBinomial)
was used only to identify the NB2 parameterization/statistical quantity; no Python
or third-party numerical implementation was executed.

Independent static reviews checked the algebra and status/figure integration.
They identified and the source addressed: near-mean cancellation, rounded-zero
scaled contrasts, subnormal-product amplification in the distant branch, and
canonical sqrt Panic effects. These are source/algebra reviews, not a Lean proof
or floating-point concordance. No sign epsilon clamp or silent approximation to
Poisson is used when distant arithmetic is rejected.

Residuals are formed before squaring. Squared deviance may be unavailable while
the signed residual remains available. The scalar kernel has no arrays or explicit
heap allocation. Global precision, universal representability coverage and
comparative performance are not claimed.

## Native execution

Initial Slurm15853 was deliberately cancelled during the build, before typechecks,
after static review found missing canonical sqrt Panic effects. Its source70723,
partial build log, source hash and accounting are retained. Cancellation is not a
native compiler failure or successful acceptance. Source76f308 propagates Panic
in both the public entrypoint and Poisson-root helper.

Final Slurm **15854 COMPLETED 0:0**, elapsed **00:06:59**, TotalCPU **06:56.593**,
batch peak RSS **2742612K**. A fresh source-tracking seed and modular Madaros were
rebuilt from the final frozen snapshot. The raw compiler matches the earlier
compiler hash; it was still freshly rebuilt, not substituted from that custody.

Four explicit typechecks passed:

- stdlib/math/logarithmic.sio
- stdlib/stats/regression/nb2_diagnostics.sio
- stdlib/stats/regression/nb2_deviance.sio
- examples/viz_nb2_deviance.sio

The real example compiled and executed with exit0. Completion marker:
CURRENT_SOURCE_NB2_DEVIANCE_DEMONSTRATIONS_READY.
GNU time recorded elapsed/user/system **0.00/0.00/0.00s**, peak native-process RSS
**1028KiB**, exit0. Displayed zero centiseconds is reporting resolution, not zero
execution time. This is one fixed whole-example observation. The compiler/transport
RSS above is separate; no isolated-kernel or comparative performance follows.

The supplemental report observes Poisson status0, large-mean residual status0 with
deviance status1, local large-alpha status0 and far subnormal-product status5.
These are actual descriptive outputs, not assertions or independent reference
agreement, and do not establish complete boundary runtime coverage.

| Artifact | SHA-256 |
|---|---|
| Raw compiler | `0c896942018ff4be9a86d69f30e3a8aadac955f32451c445a07fbf502ae37fc3` |
| Source snapshot | `35c0bb9163ff4103dd3886da377dcad072c75bb24a4ea1199a4a4a12949034a2` |
| Raw input manifest | `3054af9eb194d2aa341208db9bb081487360b1d59ab91165f6c983663d51c7f9` |
| Raw check/run log | `4800b602eaa747651947b2bd08ce47b1beb2f4cf3780fa8eea0b8ac82afff5de` |
| Native ELF | `582be14ef1a5340bbd962abdfcbc5e833320efded1b238898050dff76315045c` |
| Native stdout | `f322fdbf73cd0c72084e9f599cf2957309fd2cf83ba12f65173dc2ca133c3ea7` |
| SVG | `2d4fe7c275cd8dec23e791d85334e9b2ac2049914b0f8b44e548b4e5e82e87f0` |

The actual SVG passed xmllint --nonet --noout, exit0. Its complete1040x620 sips
rendering was visually inspected: both curves, signed axis, observed-count axis,
zero reference, legends and caveats are visible.
[SVG](../evidence/nb2-deviance-2026-10-22/viz_nb2_deviance/figure.svg),
[full preview](../evidence/nb2-deviance-2026-10-22/viz_nb2_deviance/preview.png),
[report](../evidence/nb2-deviance-2026-10-22/viz_nb2_deviance/report.txt),
[evidence inventory](../evidence/nb2-deviance-2026-10-22/README.md),
[relative hashes](../evidence/nb2-deviance-2026-10-22/FILES.sha256).
Last storage observation:461373440bytes available within the unchanged quota.

## Figure and resource interpretation

The real example compares Pearson NB2 and signed deviance at nine synthetic
counts, fixed supplied mean4 and alpha0.25. Axes show count and dimensionless
signed residual; both curves share a zero reference. Lines join discrete inputs.
Supplemental Poisson, large-mean, local-large-alpha and subnormal-product scenarios
print availability states without expected-result assertions. Decimal formatting
is not a tiny-contrast accuracy demonstration. No p-values, confidence intervals,
model calibration or inferential acceptance are implied.

Two1024-element plotting buffers and one64-byte label buffer are reused. Scene
storage is separate. SVG-only output avoids Canvas allocation and does not imply
that the whole program is allocation-free. No Canvas pixel comparison occurs.

## Custody and remaining acceptance

Raw logs/input manifests are losslessly compressed with original hashes. Native
ELF is preserved compressed with its raw identity. Compiler custody is also copied
to t560 ext4 and verified after decompression. Large compiler binaries and source
snapshots remain remote. Shared quota remains3TiB, with no new deletion or quota
change. File/hash/decompression checks are artifact-integrity operations.

No tests were added or run. Selected typechecks, native demonstration, and figure
inspection do not establish independent C++23/Sounio concordance, global ULP
accuracy, boundary coverage, calibration, scaling studies, full corpus, CI or
merge. Generic Forge corpus_smoke labeling is transport, not corpus acceptance.
Historical114/114 remains historical. The broad statistics/scientific-rendering/
math/genomics/physics sprint continues; this is one conditional diagnostic slice.
