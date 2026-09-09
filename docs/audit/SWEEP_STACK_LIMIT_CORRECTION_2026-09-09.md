<!-- docs:meta
topic_id: repo.docs.audit.sweep-stack-limit-correction-2026-09-09
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.sweep-stack-limit-correction-2026-09-09
-->

# Correction: the sweep numbers in three commits on this branch were measured
# under a stack limit this project does not use

Date: 2026-09-09
Branch: `feat/chemistry-surface-microkinetics`
Commits affected: `f14f5dfe46`, `1611638ecd`, `e0b5b70722`

## What is wrong

All three commit messages report `tests/run-pass` sweep results measured with
the default 8 MiB stack. This project does not run the compiler that way. Its
own gates raise the limit first:

    scripts/ci/aggregate_field_identity_gate.sh:38   ulimit -S -s 524288
    scripts/ci/canonical_compiler_gate.sh:41         ulimit -s 1048576

Measured again under `ulimit -S -s 524288`, on the same 1928 fixtures and the
same binaries:

| build                                   | OK   | SIGSEGV | err |
|-----------------------------------------|------|---------|-----|
| `bin/madaros-linux-x86_64` (baseline)   | 1793 | 0       | 135 |
| units + ontology kernel                 | 1800 | 0       | 128 |
| + operator rule and boundary fixes      | 1800 | 0       | 128 |

At 8 MiB the same three binaries scored 1296 / 1383 / 1374 with 514, 429 and
438 SIGSEGVs respectively. **Every one of those segfaults was the measurement,
not the compiler.** A 15-fixture sample drawn from the 429 passes 15 of 15
under the raised limit.

## The corrected claims

| claim as committed | actual |
|---|---|
| `f14f5dfe46`: 1296 -> 1383, 87 gains | baseline 1793; the branch ends at 1800 |
| `1611638ecd`: 87 gains, 0 regressions | 7 gains, 0 regressions, for the whole branch |
| `e0b5b70722`: 1382 vs 1383, 2 gains 3 regressions in a "flaky family" | 1800 vs 1800, 0 gains, 0 regressions |

Seven fixtures, not eighty-seven. The gains all come from the units and kernel
work; the operator rule moves the pass count by zero, which is what its own
commit message already says about itself -- its effect is the E151/E152/E257
diagnostics, and those were verified by running the programs and reading the
error codes, not by counting the sweep.

## What still stands

- **Every build-to-build comparison.** All of them ran at the same 8 MiB, so
  the deltas between builds were like-for-like. The regression found and fixed
  mid-branch was real: a guard placed in the call-argument loop took
  `graphics_quality_band_smoke` from 4 SIGSEGVs in 12 runs to 12 in 12, a
  deterministic worsening, and moving the locals out brought it back into
  noise. Under the raised limit that fixture does not crash at all, on any of
  these builds -- but the frame growth that caused it was real and the fix is
  the right one regardless.
- **Every boundary and operator claim.** Those were measured by compiling
  specific programs and reading specific error codes:
  `needs_mg(t: second)` rejected with E041, an untagged f64 rejected with
  E150, a disjoint class rejected with E151, `H2 + CO2` rejected with E257,
  `unit_derived_binding_keeps_value` accepted. None of that depends on the
  stack limit.
- **The ontology fixture counts** (41 of 42, unchanged from HEAD) were the
  same at both limits.

## The "flaky family"

`graphics_quality_*` was classified during this branch as intermittently
crashing, and a 12-run protocol was built to tell noise from regression. That
protocol was measuring fixtures sitting exactly on the 8 MiB boundary. Under
the project's own limit they do not crash. The protocol was sound; what it was
applied to was an artefact.

## Why it went unnoticed

The first sweep of the branch reported 514 SIGSEGVs out of 1928 -- 27% of the
suite crashing the compiler -- and that was treated as the landscape to
measure against rather than as a result that needed explaining. Every
subsequent measurement asked "what changed since the baseline" and never "is
this baseline plausible". A compiler that segfaults on a quarter of its own
test suite is not a baseline; it is a symptom, and in this case the symptom
was in the harness.

## For anyone measuring this compiler

Raise the stack first, the way the gates do:

    ( ulimit -S -s 524288; SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc check FILE )

The checker's by-value spine carries frames that cannot fit in 8 MiB at all --
the build log reports 43 oversized frames in `self-hosted/check/check.sio`,
from 5 MB to 29.7 MB, and **all 43 are by-value methods; not one is a `*mut`
function.** That asymmetry is the whole reason the `*mut` transcription
exists, and it is why `scripts/ci/checker_spine_bridges.frozen` treats every
remaining bridge back into the by-value spine as debt to be paid down rather
than a pattern to copy.
