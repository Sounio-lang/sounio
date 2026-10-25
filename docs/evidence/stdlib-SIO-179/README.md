<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-179.readme
authority: repo_only
audience: users
last_validated: 2026-10-25
validated_by: Slurm17538 C++23 campaign closure gate
source_of_truth: docs/evidence/stdlib-SIO-179/campaign-17048-closure-text.tar.gz
-->

# SIO-179: exhausted profile-bootstrap campaign closure

The original qualification campaign is **incomplete**. Job 17048 ended with
process status 124 after 23:45:01; its round protocol allowed 85,500 seconds and
BUDGET_USED_SECONDS records 85,500. The preceding900-second allocation remains
separate. No campaign was restarted, no budget reset, no quota changed, and no
historical artifact removed.

This change adds a C++23 receipt reconciler and report-only gate. It does not
change production numerical code or set calibration_assessed. The task and
evidence are tracked in [Notion SIO-179](https://app.notion.com/p/4033f37f9371810899fdf1e3022d0e4c)
under the massive stdlib epic. Publication and merge are separate review steps.

## Source and execution identities

- Report integration base: remote main
  8eca8e0c086bbd074485bbfd8716accc4063ef1c.
- Original campaign source:
  6ffc1a5bd8eaebeef2d15ccced830bfbc832a648.
- Original compiler SHA256:
  d74fbebe0ff442e8c77bec024d6d92325b83a98d3f1487caa1db59ef171a41e6.
- Original functional receipt SHA256:
  c415632ec6a64d57060e1ec2c62b16443d626c1e7cd5ad9645b2be07abfabfe3.
- Report-only qualification job **17538**, Slurm UID1000, 1 CPU/2 GiB:
  COMPLETED, exit 0:0, elapsed 00:00:04.
- Report executable SHA256:
  146f4b87f5a19e8e9a56c507f66ce5fdf7ab83a8b046a46d1c91edc624d1b678.
  Its ELF stays on the worker.

Job 17538 verified the full original campaign input catalog, including its
absolute compiler/functional receipt/result inputs, before and after reporting.
Every READY batch's original FILES manifest passed. The snapshot's selected raw
text inputs and the new C++/shell sources also passed final hash checks.
This closes the missing final catalog check left by the timeout; it does not
turn the original campaign into a successful or current-main numerical run.

The final gate seals every selected original output path and content hash before
copying, compares the copied raw catalog exactly, then regenerates and compares
the original catalog after all report executions and controls. This includes
individual records from the unfinished batch that have no READY manifest. The
C++ executable is hashed before its first execution and verified afterward.

## Reconciled observations

The C++ report covers all **1,152 planned dataset identities** across the nine
declared cells, with B199 and target 128 per cell. It reads individual case
statuses and output, metadata seeds, completed batch aggregates and cell
summaries. The totals are:

- 407 attempted executions, 406 completed native records.
- 405 available fits, one completed model refusal (status 12).
- One process timeout (124), 745 not-started identities.

Six completed individual cases from the unfinished
n256_alpha.25/batch32 are included. Its old cell summary only covered 32 cases;
the reconciler finds 38 complete records before case 6 timed out. The unfinished
batch has 48 generated identities in that cell, 39 attempted executions, and89
not-started identities in its full target 128.

The machine-readable report/cases.tsv retains every planned identity and its
scheduled seeds, distinguishes actually generated data, and records the final
RNG words only when present. report/cells.tsv separates model unavailability,
process failure and unattempted cases. Work sums use observed CAL_WORK fields;
trace sums are explicitly recorded prefixes, never inferred work.

report/REPORT.md contains the per-cell rejection counts and approximate
conditional Wilson intervals. These use available completed fits only and the
prespecified plus-one bootstrap p <= 0.05 rule. Runtime selection, optional
stopping, availability selection and shared RNG streams limit their
interpretation. The zero-rejection plug-in MC standard error is not a guarantee
of zero Monte Carlo error. Planned-family bounds treat all unavailable, failed
and unattempted outcomes as unknown; they are deterministic missing-outcome
bounds, not confidence intervals. There is no pooling across cells or historical
campaigns, no full 128-per-cell calibration claim, and no DESeq2 equivalence claim.

## Reproduction and negative controls

Run only within an allocated Slurm UID1000 CPU job, with fresh output:

~~~bash
bash scripts/stdlib/count_profile_campaign_closure_gate.sh CAMPAIGN_ROOT FROZEN_SOURCE_ROOT FRESH_OUTPUT_ROOT
~~~

Original roots:

- Campaign:
  /scratch/sounio-stdlib-inference-scale-20261024/nested_qualification_final03/campaign/qualification_nested_B199
- Frozen source:
  /scratch/sounio-stdlib-inference-scale-20261024/combined_final_cache04/source
- Report-only source:
  /scratch/sounio-stdlib-inference-scale-20261024/SIO179-closure/source03
- Report-only output:
  /scratch/sounio-stdlib-inference-scale-20261024/SIO179-closure/report03

Six mutations of actual copied receipts were rejected with exit 2 and their exact
expected diagnostics: missing declared cell, changed denominator, wrong seed,
missing case completion, changed process status, and a fabricated whole-campaign
COMPLETE marker. Original receipts were unchanged.

The first report-only job 17536 parsed the same original totals and verified the
original catalogs, then failed its harness because the process-status mutant was
correctly rejected earlier (READY failed execution) than the expected text.
The final harness expects that exact diagnostic. Original17536 sources, result,
failure status and diagnostic are retained in the archive; they are not counted
as a passing gate.

The earlier full report-only pass17537 remains separate. Root review required
stronger output custody;17538 adds the original-output pre/copy/post seal and
pre-execution executable hash. The cases, cell summaries and narrative report are
byte-identical across these two passes. A seventh control appends a marker to
case6 stdout from the unfinished batch in an extracted copy only; its selected
output catalog differs with cmp exit1, proving the custody mismatch is detected.
The actual original campaign is never modified.

## Compact custody

campaign-17048-closure-text.tar.gz is a deterministic text-only packet:
**1,575,782 bytes**, SHA256
3a5b5dfb580c2043035a3060311085bb4ff1c79b1e5e96747b223b75cf9dff2d.

It contains original selected case stdout/stderr/status/resources, dataset
metadata and generated driver text; full original INPUTS catalog; original
generator/sampler/controller sources and authorization ledger; new C++/shell
sources; full generated report; pre/post integrity results; six mutation
diagnostics; and Slurm job receipts. Its internal FILES.sha256 covers each
included regular file. Original READY manifests refer also to the external
driver ELFs: those binaries are not in this packet, and their verified original
copies remain on the worker. No compiler, executable, library or toolchain is
committed by this change.

After extraction, sha256sum -c FILES.sha256 verifies packet contents. The
standalone C++23 reader can reproduce reports from its raw directory without
sampling or fitting. The full gate additionally requires the original frozen
source and external binaries to reverify the historical execution custody.
