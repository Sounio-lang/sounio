<!-- docs:meta
topic_id: repo.docs.audit.lorenz-i256-product-magnitude-2026-08-20
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.lorenz-i256-product-magnitude-2026-08-20
-->

# Lorenz i256 Product Magnitude Audit

Date: 2026-08-20

Source tree: origin/main at 67aa2aec127020122ff961480b83b36c09e91432

## Verdict

**EXCEEDS I64**

The maximum observed magnitude in the measured Lorenz certificate paths was:

    8,007,432,506,888,905,229,835,698,176

This is the y_lte_source * den intermediate in
lorenz_i256_step5_taylor2_remainder_obligation_check, at
stdlib/systems/lorenz_i256_cert_step5.sio:2310.

The comparison bound was the signed i64 maximum:

    2^63 - 1 = 9,223,372,036,854,775,807

The observed value is exactly 868,167,572 times `2^63`, not exactly that many
times `2^63 - 1`. Dividing by the signed i64 maximum gives quotient
868,167,572 and remainder 868,167,572. It is therefore an observed product,
not an inferred requirement from the type name or from the number of digits in
a source literal.

## Measurement

The measurement used two coupled checks:

1. A source-built Madaros ELF compiled and executed a representative corpus.
   Each fixture was invoked through scripts/ci/souc-native-wrapper.sh run, so a
   compile marker and a runtime return code were recorded. The build was made
   from this worktree source payload on Slurm partition=all, on
   cpuops-t560-proxmox, with 32 CPUs:

       REMOTE_BUILD_RC=0 ELAPSED=241
       REMOTE_ELF_BYTES=100562528
       0b2f7e21f7a9260e85cbd13e121bfd7537b3ef273148db2192abe2e241bf2769
       compile: fns=10951

2. A host-side arbitrary-precision accumulator replayed every arithmetic
   expression typed i256 in the executed certificate checks, including the
   quotient * denominator + remainder reconstruction performed by
   div_witness_check_i256. It recorded 933 intermediate arithmetic values.
   This avoids ordinary Madaros print(f64) saturation at the i64 range; no
   floating-point conversion was used for the reported maximum.

This is instrumentation of the executed call set, not an estimate from source
constants: the source-built run established which checks actually ran, and the
accumulator then re-evaluated only those typed-i256 expressions with exact
arbitrary-precision integers. Source reading was used to identify function
bodies and the winning line, but was not substituted for execution evidence.

The winning expression evaluates as:

    source_scale = 4,294,967,296
    dt_q         = 42,949,672
    y_second     = 4,340,838,038,257
    y_lte_source = 217,041,893
    den          = 2 * source_scale * source_scale
                 = 36,893,488,147,419,103,232

    y_lte_source * den
    = 8,007,432,506,888,905,229,835,698,176

The competing y_second * dt_q * dt_q value is
8,007,432,477,754,892,763,381,441,088; the winning value is the source
upper-bound product at line 2310.

## Covered Paths

The executed source modules were
`stdlib/systems/lorenz_i256_cert_step1.sio` through
`stdlib/systems/lorenz_i256_cert_step6.sio`, plus
`stdlib/systems/lorenz_i256_cert_trajectory5.sio` and
`stdlib/theorem/div_witness.sio` (`div_witness_check_i256`, lines 27-37).

The source-built run executed all of the following check families:

- step 1: center_artifact_check, radius_artifact_check, and
  remainder_obligation_check;
- step 2: center_artifact_check, radius_artifact_check, and
  remainder_obligation_check;
- step 3: center_artifact_check, radius_artifact_check, and
  remainder_obligation_check;
- step 4: center_artifact_check, radius_artifact_check, and
  remainder_obligation_check;
- step 5: center_artifact_check, radius_artifact_check, and
  remainder_obligation_check;
- step 6: center_artifact_check, radius_artifact_check, and
  remainder_obligation_check;
- the step certificate and lorenz_i256_trajectory5_certificate_check, which
  calls the step-5 certificate check;
- one imported cover-child-0 arithmetic bundle, one imported cover-child-1
  arithmetic bundle, and the imported cover-refinement ledger;
- the standalone lorenz_i256_product_smoke and lorenz_i256_fixed_step fixtures
  as negative controls.

The fixture paths were the 18 imported step-1 through step-6 centre, radius,
and remainder fixtures, `tests/run-pass/lorenz_i256_step_certificate_imported.sio`,
`tests/run-pass/lorenz_i256_trajectory5_certificate_imported.sio`,
`tests/run-pass/lorenz_i256_cover_child0_axis_arithmetic_bundle_imported.sio`,
`tests/run-pass/lorenz_i256_cover_child1_axis_arithmetic_bundle_imported.sio`,
`tests/run-pass/lorenz_i256_cover_refinement_ledger_imported.sio`,
`tests/run-pass/lorenz_i256_product_smoke.sio`, and
`tests/run-pass/lorenz_i256_fixed_step.sio`.

The 25 source-built fixture results were:

    25 fixtures emitted a compile marker and returned a recorded result.
    22 returned rc=0.
    1 returned rc=1: step-1 center artifact (declared known-failure).
    product_smoke returned rc=3 (declared known-failure).
    fixed_step returned rc=4 (declared known-failure).

The last two are not counted as certificate success; they establish that the
same source-built instrument also reaches the known failing standalone i256
paths instead of silently skipping them. The step-1 center return is a
contract result after compilation, not a compile refusal.

## Uncovered Paths

This is a representative measurement, not a claim of exhaustive Lorenz
coverage. The following were not traversed by this run:

- the remaining imported step-1 through step-6 candidate, replay, enclosure,
  flowpipe, and proof-skeleton checks;
- lorenz_i256_trajectory5_projection_inclusion_fingerprint, which contains
  additional div_witness_check_i256 calls but was not called by the
  trajectory certificate fixture;
- the full family of standalone lorenz_i256_child*.sio modules and all
  child-2 through child-4 imported fixtures;
- lorenz_i256_beta_z_bridge, limb/scaled-product bridge fixtures, trajectory2
  bridges, and the remaining non-certificate Lorenz modules;
- long-running trajectory or enclosure loops beyond the finite fixed
  certificates selected above.

The cover-child and refinement fixtures included here are receipt/ledger
paths. Their current source bodies do not add a measured i256 product to the
maximum; their inclusion proves execution of those API paths, not arithmetic
coverage of every child implementation.

## Execution Boundary

The following paths are **NOT EXECUTABLE for this receipt**: they were not in
the selected source-built corpus, so this run supplies no runtime evidence for
them. This is a measurement boundary, not a claim that the source can never
execute them:

- the remaining candidate, replay, enclosure, flowpipe, and proof-skeleton
  fixtures;
- `lorenz_i256_trajectory5_projection_inclusion_fingerprint` and its
  projection-inclusion fixtures;
- child-2 through child-4 cover fixtures and the standalone child modules;
- beta-z, limb, scaled-product, trajectory-2 bridge, and long-loop fixtures.

They remain `NOT EXECUTABLE` in this receipt because no source-built invocation
was made for them. Static inspection of those paths is not counted as a
measurement and does not change the verdict's coverage boundary.

## Positive Control

Before accepting a fits result, the accumulator was forced with the exact
control value:

    POSITIVE_CONTROL = 1,000,000,000,000,000,000,000,000,000,000
    POSITIVE_CONTROL > 2^63 - 1 = true

The detector reported the forced value as exceeding i64. This control is
separate from the Lorenz values and demonstrates that the accumulator and
comparison reject a known overflow rather than merely echoing the source
constants.

## Interpretation

The result does not promote a real multi-limb i256 implementation, change any
stdlib/systems annotation, or prove that every i256 path is correct. It
answers the narrower measurement question: products actually reached by the
covered Lorenz certificate formulas exceed signed i64 by eight orders of
magnitude. Any claim that these certificates behave as i64 is incompatible
with this observed intermediate and must remain an implementation/blocker
claim, not a numeric-tower proof.
