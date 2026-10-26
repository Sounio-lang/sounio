<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-172.ci-repair-20261025.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-172.ci-repair-20261025.readme
-->

# PR3111 CI integration repair, 2026-10-25

Original reviewed/published HEAD: ae3b3ba17c28a1e3eed188e89d95563a7f77b61c.
The hosted logs identify three actionable fixture/integration failures:

- FullSuite121043641262/run40150912361: phonon_factor_status uses
  bits_to_f64, undefined in that Stage2; phonon_factor_private uses the
  unrecognized expect-error annotation.
- Core121042447071: known_failure_pin_gate still lists the former SI quantity
  parity known failure, although the production test is now an ordinary run-pass.

The repair changes the three typed nonfinite constructors to 0.0/0.0 and
1.0/0.0 with binary64 bit-class assertions. NaN exponent/fraction and positive
infinity are checked in the actual status fixture. All original refusal inputs,
operation order, expected statuses and numerical reference values remain.
NaN payload/sign identity is not claimed. A separate small imported-free
Sounio bit probe and C++23 parser verify the actual emitted bit classes.

The private-field fixture keeps the same real owner, forbidden mutable borrow
and close operation; only expect-error becomes the recognized error-pattern.
The baseline shrinks exactly one entry, test_phonon_quantity_parity.sio. No
other known failure is removed, no new baseline is admitted and no test is
skipped or tagged to hide a failure. The full pin gate passes46 existing entries
and its16-case/two-mode self-test.

## Local compiler receipts

Immutable worker root:
 /scratch/sounio-stdlib-inference-scale-20261024/SIO172ci.

Before01/job17359 preserves the exact original status fixture: committed seed895
check fails with undefined bits_to_f64. After02/job17360 completes in46s:

| Native/reference scope | Modular c21 a292 | Committed seed895 |
|---|---|---|
| Original core,171 values | Pass | Pass |
| Production SI parity,30 values | Pass | Pass |
| Original status cases,13 values/31 statuses | Pass | Pass |
| Actual arithmetic NaN/+Inf bit classes, C++23 | Pass | Pass |
| Original private borrowed-field negative | E259 refusal | Accepted by check, ELF emitted |

Both pins have the original c21 provenance documented in the parent evidence.
The entire existing phonon_factor_gate passes on modular c21, including the
unchanged legacy wrong-variance witness, private-field refusal, all three oracle
modes and their missing/corrupted-record controls. No production numerical
kernel, physical parameter, tolerance, model or C++ analytic equation changed.

The seed privacy result is a separate compiler enforcement limitation:
the original reachable negative is preserved, not replaced by a nearby
direct-field assignment. Check output reports an emitted ELF; that negative
program is never executed. This does not establish a universal privacy claim,
nor does seed895 acceptance establish the behavior of an unidentified hosted
Stage2 binary. Updated hosted CI must be observed independently. Keep the PR
Draft while that boundary remains unresolved; local numerical agreement is not
full CI or privacy enforcement qualification.

The text packets record each retained file's exact byte count and SHA256,
actual compiler hashes, source hashes, native ELF hashes, exit codes,
resources and C++ outputs. Raw ELFs remain on the worker. Original source
fixtures and exact hosted logs are retained alongside the repair receipts.
Hosted logs use base64 framing with original byte counts and SHA256 so trailing
spaces and CR/LF bytes remain exact while Git whitespace checks stay meaningful.
