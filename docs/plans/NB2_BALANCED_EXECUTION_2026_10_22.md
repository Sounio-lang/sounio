<!-- docs:meta
topic_id: repo.docs.plans.nb2-balanced-execution-2026-10-22
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.nb2-balanced-execution-2026-10-22
-->

# Balanced NB2 tiny-product continuation: execution receipt, 2026-10-22

## Source and scope

Initial remote-main basis was `06f927dbca5753f5d7b6f64c29eb745eff77f351`.
Final readback found `7d67f3a944cb8597cb4359461e7666fbc46aa308`, a README-only
change, incorporated after native execution. Its merge is checked to leave
bin/scripts/self-hosted/stdlib/examples unchanged relative to the closure commit.
Branch `sprint/nb2-balanced-main-06f927-20261022` retains it and all prior
sprint increments. Frozen executable source:
`eeeab3025c68bebfdbde284116e41176b9976770`. Subsequent documentation/evidence commits
are not relabeled as executed source.

The preceding NB2 implementation deliberately rejected the far region where
alpha*mean rounded to zero or subnormal. That restriction prevented amplification
of product rounding by the count. The new bounded branch evaluates

m1 = mean*(1+alpha*count)/(1+alpha*mean)

directly in the balanced order, computing alpha*count first. Its denominator rounds
to one in this region; finite/positive guards remain. Scaled difference delta1 is
retained independently. If m1 is subnormal, the far Poisson contribution uses
ln(mean)+log1p(alpha*count)-log1p(alpha*mean), carried independently of rounded m1.
This avoids the relative rounding of subnormal m1 being magnified by its logarithm.
The rounded positive m1 still supplies the additive term.

Independent static review found no blocker in that bounded formulation or its
actual source/effects/control flow. This is algebraic/source review, not a Lean
proof, numerical concordance, correctly rounded implementation or global error
bound. Existing local, zero-count, Poisson-limit, normal-product and squared-payload
availability paths retain their formulas. Remaining arithmetic guards are not
removed to force availability.

## Scientific report

The existing io::sci_print::println_f64_sci formatter now prints available residual
and deviance payloads with15significant digits. Each supplemental scenario prints
both statuses first; unavailable fields print unavailable rather than an error
payload zero. Fifteen digits are not a binary64 roundtrip or independent accuracy
reference. Signed zero is normalized by that formatter's existing text policy.

The nine displayed counts, supplied mean4/alpha0.25, curves, axes, zero reference
and scene geometry remain the same as the previous demonstration. Supplemental
scenarios are reports, without expected-result assertions. No extra Canvas or
plot buffers were added. The scalar kernel remains free of arrays/explicit heap.

## Native evidence

Slurm **15874 COMPLETED 0:0**, elapsed **00:07:03**, TotalCPU **06:57.243**,
batch peak RSS **2820204K**. A fresh source-tracking seed and modular Madaros were
rebuilt from the frozen snapshot. Cache was explicitly disabled for this run.
The rebuilt raw compiler has the same identity as the prior build; it was freshly
built rather than substituted from prior custody.

Five explicit typechecks passed:

- stdlib/io/sci_print.sio
- stdlib/math/logarithmic.sio
- stdlib/stats/regression/nb2_diagnostics.sio
- stdlib/stats/regression/nb2_deviance.sio
- examples/viz_nb2_deviance.sio

The example compiled and executed with exit0. Completion marker:
CURRENT_SOURCE_NB2_BALANCED_DEMONSTRATIONS_READY.
GNU time recorded elapsed/user/system **0.00/0.00/0.00s**, peak native-process RSS
**1052KiB**, exit0. Zero displayed centiseconds is reporting resolution, not zero
execution time. This is one fixed whole-example observation; compiler/transport
RSS above is separate. No isolated-kernel or comparative-performance claim follows.

Actual supplemental report observations:

| Scenario | Residual status | Deviance status | Reported result |
|---|---|---|---|
| Poisson limit | 0 | 0 | Scientific payloads available |
| Large mean | 0 | 1 | Residual -1.41421356237310e+154; deviance unavailable |
| Local large alpha | 0 | 0 | Residual -9.99999333367058e-157; deviance 9.99998666732676e-313 |
| Far subnormal product | 0 | 0 | Residual 1.06603988682139e+02; deviance 1.13644104029415e+04 |

These are descriptive native outputs, not independently verified expected values.
The previous source76f308/run15854 reported status5 for the same far-product
scenario; the current execution reports both payloads available. No old source
was rerun or numerical reference generated for this increment.

| Artifact | SHA-256 |
|---|---|
| Raw compiler | `0c896942018ff4be9a86d69f30e3a8aadac955f32451c445a07fbf502ae37fc3` |
| Source snapshot | `4a619763d7b6bcfd4d39381e8c54a3293933be9aef0be63c050c71d11e50d55c` |
| Raw input manifest | `60484496cec7baf5c67c992c4f2858fbe748298e8c11cc00272a99a005b2ae4e` |
| Raw check/run log | `ebc10456c2e6d6e58697d1515f3e280d6a6adea48f7d99047351cb2f0c3fce08` |
| Native ELF | `de6826dbe5bf74037a9e66c8ca92051af4d0da4ce40c6ac374890f09e9809686` |
| Native stdout | `a189379a7d1a7e0e21d1b6e6cf90b0e98130cd040d645df49e01133628912788` |
| SVG | `2d4fe7c275cd8dec23e791d85334e9b2ac2049914b0f8b44e548b4e5e82e87f0` |

The SVG hash is identical to run15854, establishing serialized figure identity,
not independent numerical concordance. The current SVG passed XML parsing with
xmllint --nonet --noout, exit0, and its complete1040x620 sips rendering was visually
inspected: curves, signed axis, observed count, zero line, legends and caveats are
visible. No Canvas execution or pixel comparison was performed.
[SVG](../evidence/nb2-balanced-2026-10-22/viz_nb2_deviance/figure.svg),
[full preview](../evidence/nb2-balanced-2026-10-22/viz_nb2_deviance/preview.png),
[report](../evidence/nb2-balanced-2026-10-22/viz_nb2_deviance/report.txt),
[inventory](../evidence/nb2-balanced-2026-10-22/README.md),
[relative hashes](../evidence/nb2-balanced-2026-10-22/FILES.sha256).
Last shared-storage observation:314572800bytes available under the unchanged quota.

## Custody and acceptance boundaries

Raw logs/manifests are compressed losslessly with original hashes. Compressed
native ELF is preserved and its raw identity checked. Compiler custody is also
copied to t560 ext4 and verified after decompression. Large compiler/source packets
remain remote; quota remains3TiB, with no new removal or quota change.

No tests or independent numerical oracle were added or run. Availability observed
in a real example is distinct from independent C++23/Sounio concordance, boundary
coverage, global accuracy, calibration, scaling/comparative resource evidence,
full corpus, CI or merge. Generic Forge corpus_smoke naming is transport only.
Historical114/114 remains historical. See [module contract](../../stdlib/stats/regression/NB2_DEVIANCE.md)
and [example contract](../../examples/viz_nb2_deviance.md).
