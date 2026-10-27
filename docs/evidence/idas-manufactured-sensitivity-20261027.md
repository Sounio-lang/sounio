<!-- docs:meta
topic_id: repo.docs.evidence.idas-manufactured-sensitivity-20261027
authority: repo_only
audience: users
last_validated: 2026-10-27
validated_by: Codex-publication-audit
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.idas-manufactured-sensitivity-20261027
-->

# IDAS-inspired analytic and parameter-sensitivity witnesses

[SUNDIALS #1065](https://github.com/llnl/sundials/pull/1065) corrects an omission
in simultaneous sensitivity error control: an accurate state can conceal an
inaccurate parameter derivative. These three self-contained Sounio fixtures
preserve that distinction. They exercise existing scalar arithmetic and the
existing first-order `sensitivity_of` API; no compiler, standard-library or
solver implementation is added.

The metadata header records documentation validation and registry checks on
the publication date. It is not a summary of numerical validation or a claim
that every scientific statement has been independently replicated.

## Scope and public reproducibility

The three `.sio` files in this PR are sufficient to check, compile and execute
their own asserted invariants. Historical C++23 cross-verifiers, IDAS producer
programs, negative-control variants and full job archives are retained in
private author-managed custody. They are **not distributed in this PR**.
Consequently the public reproduction commands below re-execute the Sounio
fixtures, not the external IDAS concordance, original artifact hashes or all
negative controls. Job IDs identify historical observations; they are not
release-readiness evidence or publicly accessible CI results.

## Fixtures and data contracts

### Analytic grid

`tests/run-pass/idas_manufactured_sensitivity.sio` evaluates
`y(t,p) = t + p*a*t^d`, its parameter sensitivity `a*t^d`, and the time
derivative of that sensitivity. Degrees are 1, 3 and 5; factors are 1, -2 and
0.5; times are 0, 0.25, 0.5 and 1.

Output starts with `IDAS_SOUNIO_GRID_V1`, contains 36 records in degree,
factor and time order, and ends with `IDAS_SOUNIO_WITNESS_PASS`. Each record
has five integral fields: degree, factor index, time index, scaled sensitivity
and scaled sensitivity time derivative. The scale is 2^30; whitespace need
not follow physical record boundaries. Factor indices 0, 1 and 2 map to
1, -2 and 0.5; time indices 0, 1, 2 and 3 map to 0, 0.25, 0.5 and 1.
The header and footer are exact case-sensitive tokens; the intervening payload
is exactly 36 groups of five signed decimal integers, with no extra tokens.

The central parameter differences use h = 1/1024. Exactness is specific to
this solution's affine dependence on p and the selected dyadic values, which
remain exactly representable in binary64. It is not a claim that finite
differences or compiler calculations are exact generally. A one-step Euler
toy illustrates exact state evolution with an incorrect sensitivity integral;
it does not reproduce IDAS stepping internals or test automatic differentiation.

### One-channel compiler API

`tests/run-pass/idas_parameter_sensitivity_ad.sio` seeds one zero-valued
parameter once using `measure(0.0, uncertainty: 0.0)`. Six straight-line
expressions query literal channel 0: cubic with factor 1 and quintic with
factor -2, each at t = 0.25, 0.5 and 1. Output uses `IDAS_SOUNIO_AD_V1`, six
sensitivities scaled by 2^30, and `IDAS_SOUNIO_AD_PASS`. The six signed
decimal integers are cubic times 0.25, 0.5, 1, then quintic times 0.25, 0.5, 1.
Markers are exact case-sensitive tokens and no extra tokens are permitted.

Zero uncertainty is a numerical derivative seed here, not a physical
measurement model. A zero primal parameter must not erase a nonzero derivative.
State values and sensitivities are asserted before the final marker.

### Four channels and state cancellation

`tests/run-pass/idas_multichannel_sensitivity_ad.sio` seeds four inputs once
and queries literal channels 0 through 3. Two profiles use unrolled expressions:

- `y0 = t + p0*t - 2*p1*t^3 + 0.5*p2*t^5 - 0.25*p3*t^2`.
- `y1 = t + (p0+1)*t - 2*(p1+1)*t^3 + 0.5*p2*t^5 - 0.25*p3*t^2`.

At zero parameters their states are t and `2*t - 2*t^3`. Both have gradient
`[t, -2*t^3, 0.5*t^5, -0.25*t^2]`. At t = 1 the second state is zero while
all four gradient components are nonzero: `[1, -2, 0.5, -0.25]`.

Output uses `IDAS_MULTI_AD_V1`, eight records in profile/time order, and
`IDAS_MULTI_AD_PASS`. Times are 0, 0.25, 0.5 and 1. Seven integral fields
encode profile, time index, state and four partials, scaled by 2^30. Profiles
0 and 1 map to the two expressions above; time indices 0 through 3 map to
0, 0.25, 0.5 and 1. Markers are exact case-sensitive tokens and the payload
is exactly eight groups of seven signed decimal integers, with no extra tokens.

## Historical execution profile

Initial source baseline was `1a92dfbb48b96e86ccae9b05eae5bfd4bdaafe66`.
The publication branch was prepared from `27a39d1f1dcde0ce1e94efbccbac633dce5f2049`;
the three fixture files are unchanged from their validated private versions.
The canonical `bin/souc` launcher selected the committed Madaros prebuilt,
with `SOUNIO_REQUIRE_COMMITTED_MADAROS=1`. Raw compiler SHA256:

`e33dcf87f3a7cbca50b2c5c008da2a06153bfa552c73475b01239aff87415514`

This is not a current-source compiler rebuild or fixed-point result. Measured
calls explicitly used `--science-boundary off` to avoid the Python attestor
under the author's ADR009 calculation policy. No automatic boundary
attestation is claimed. An initial exploratory default check invoked that
advisory helper before its dependency was identified; it is not a numerical
validation receipt.

| Fixture | Source SHA256 | Native ELF SHA256 |
| --- | --- | --- |
| Analytic | c4598326c98fe17db2724e669674910f11caed9c27a8855840168d7373c01e07 | 3beb2900fa3dd4219fa0dad3a21f2f7e4fc48f488b1b6a87659a073b5b4e0db9 |
| One channel | 2ad01c539fb3b4ea08c56f9605ed303dbed7fcb7961a9a4ac86e06697573541d | c7e1818eb9061ec4b2b464bcb887018c6ca69329c9972d6a4809cfada3eee34c |
| Four channels | c0702c4c79440a732d9058b4c69442afdaddeb5b1cd751216cecd098e6263082 | 52c46d45c26696d227f276f23ca95e52ee1868a3471221a9ef4c686e92b0ee69 |

In publication-package job 18075, the input manifest recomputed and verified
all three source hashes against the actual compiled payload. The resolved
committed raw compiler hash was also verified. The native-ELF hashes above
identify the historical artifacts, not newly asserted hashes of the package run.

## Observed gates and negative controls

| Gate | Historical Slurm job(s) | Observation |
| --- | --- | --- |
| Analytic native fixture and canonical harness | 17888 | 36 records; 1 PASS, 0 FAIL, 0 SKIP; compiled incorrect assertion rejected |
| Actual IDAS baseline/candidate producer | 17894 | Separate clean builds and per-source hashes; complete native grids |
| Analytic C++23/native concordance | 17897 | 36 exact records, 144 unchanged native controls, 72 bounded-accuracy values; three negative controls rejected |
| One-channel native API, existing controls and harness | 17907 | Six exact API values; three existing controls pass; 1 PASS, 0 FAIL, 0 SKIP; corrupted output and compiled incorrect assertion rejected |
| Four-channel native fixture and harness | 17925 | Eight states and 32 asserted partials; 1 PASS, 0 FAIL, 0 SKIP; compiled incorrect derivative rejected |
| Four-sensitivity IDAS producer | 17926 | Original/corrected solves for both parameter profiles |
| Four-channel exact/native concordance | 17929 | Eight exact Sounio states and 32 exact partials; candidate IDAS accuracy checks pass; swapped channels and zeroed gradient rejected |
| Publication-package native/harness rerun | 18075 | Actual publication payload; each fixture 1 PASS, 0 FAIL, 0 SKIP; all three source hashes and raw compiler identity verified |

Job 18075 invokes the canonical harness separately with `--filter-exact` and
the full `.sio` basename for each fixture; a wrapper supplies the explicit
`--science-boundary off` measurement profile. Its manifest and result JSON
remain in private custody, not a hosted CI/release gate. Preceding job 18074
stopped before tests when a wrapper version probe required omitted source/Git
provenance; that diagnostic failure is retained separately.

The one-channel run's existing controls were `sensitivity_chain_product`,
`sensitivity_multi_channel` and `kl16c_fo_across_user_fn`; the last emitted its
strong PASS marker. Every incorrect-assertion variant was separately compiled,
exited nonzero and failed to emit its final success marker.

The first external IDAS grid used rtol = 1e-8, atol = 1e-12 and a one-million
step budget. Its declared absolute global-error criterion was 1e-4. A selected
cubic baseline error was approximately 0.635567 versus 2.23599e-13 after the
correction. Maximum controlled error on that grid was approximately 6.80705e-5.
For the four-sensitivity experiment, observed maximum state and partial errors
were 4.568115308765941e-12 and 4.0555016012078227e-9. Original profile 0's final
maximum partial error was 1.2711342577226734. These are particular global
errors on declared grids, not theorems connecting local and global tolerance.

Incomplete packaging jobs 17882 and 17885 were retained. They ran the analytic
fixture but could not complete the bundled harness. A different private bridge
attempt reused an original object after restoring a source with an old timestamp;
its apparent candidate is not accepted evidence. The final producer 17894 uses
clean builds and source hashes. No failed attempt is relabeled as a complete pass.

## Current semantic lane

Semantic-Lane-ID: idas-bidirectional-publication-20261027
Owner: Codex publication lane
Concept-IDs: none; no primitive concept or uncertainty meaning changes
Intent-Preserved: an accurate or cancelling state must not certify sensitivities
Transformation: publish three bounded regression fixtures and this evidence note
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: execution observations limited to the stated grids and artifact
Claims-Forbidden: general AD, solver parity, global-error theorems, physical uncertainty
Assumptions: binary64 control arithmetic and the selected dyadic values
Write-Set: three named tests/run-pass/idas_*.sio fixtures and this document
Read-Set: SUNDIALS #1065, existing sensitivity tests, canonical compiler/harness
Positive-Witness: asserted native values, exact C++23 checks and scoped harness runs
Negative-Witness: corrupted output, swapped channels, omitted gradient, wrong assertions
Acceptance-Gate: all three native/harness fixtures plus the separately identified historical checks
Integration-Target: draft PR from test/idas-bidirectional-witness-20261027 to main
Authoritative-Only-If: source/artifact identity and executed result remain explicit

## Reproduce the selected Sounio fixtures

From a Linux checkout with the canonical compiler available:

```bash
validation_dir=$(mktemp -d)
for fixture in idas_manufactured_sensitivity idas_parameter_sensitivity_ad idas_multichannel_sensitivity_ad; do
  bin/souc --science-boundary off check "tests/run-pass/$fixture.sio"
  bin/souc --science-boundary off compile "tests/run-pass/$fixture.sio" -o "$validation_dir/$fixture"
  "$validation_dir/$fixture"
done
```

These commands run the selected asserted fixtures, not the full compiler corpus,
a compiler rebuild, automatic boundary attestation or the private IDAS experiment.

## Review and limits

Independent local review checked the bounded mathematics, fixtures and private
verifiers. An external xAI/Grok review of the publication text identified stale
lane metadata, unclear archive availability and readability issues; those were
corrected in this note. Other external review routes were unavailable. The
operational audit stays in private custody; a second independent external
review remains pending. No maintainer or independent-vendor approval is claimed.

Nothing here establishes dynamic-channel behavior, arbitrary helper/loop or
recursive differentiation, general AD correctness, an IDAS port/FFI, physical
uncertainty propagation, performance, full-corpus success or compiler soundness.
All existing solver/compiler paths remain unchanged.
