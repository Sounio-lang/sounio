<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-deep-integration-2026-10-23.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-deep-integration-2026-10-23.readme
-->

# Final scientific stdlib integration receipt

Source commit fc78b54079275a13f431a54507a7dbbf89693bd6, tree
8faff63080680d361f49b8e7dbe47a7a6ca05d85, completed Slurm16189 with exit0:0.
RESULTS.tsv contains374 records, all status0. Elapsed wall time was00:09:03.
The frozen source archive includes main57a33, foundation870df, rendering00d9
and science74996, with the effects scanner and cancellable CI integration.

The protected source-built compiler SHA256 is
1148eabd3905ec76911d027c578fb8d6b2bd2dde2c186ee3c02c4c3b97733029.
Earlier GitHub run40009549471 independently rebuilt that same ELF and passed
330 scientific replay records. Its source/ELF transfer and tamper controls
passed before execution. The final replay adds eleven legacy rendering fixtures
on Madaros; Slurm16188 separately passed ten legacy fixtures plus first-tick
coverage on the original Stage2 CI executable and focused C++/Madaros controls.

## Evidence and scope

RESULTS.tsv records checks, compiles, native executions, completion markers,
expected-negative controls, independent C++ concordances and endpoint identity
verification. EXECUTABLE_IDENTITIES.sha256 binds fixture/executable bytes.
RESOURCES.txt concatenates original timing records with source labels.
RAW_OUTPUTS.sha256 hashes full scratch transcripts; INPUT_MANIFEST.sha256
identifies the complete listed stdlib/tests input manifest. Source and compiler
hashes were unchanged across the run. This is endpoint integrity for the listed
inputs, not a proof of whole environment immutability or complete consumption.

Full logs, input manifest and ELFs remain at
/scratch/sounio-stdlib-deep-20261023/final2/artifacts.
This receipt is kept on a separate evidence branch so the accepted source
commit remains stable during CI. Main integration is tracked by PR3052.
A documentation-only follow-up can incorporate this receipt after integration.

The selected Lean structural/RODAS rational proofs and GitHub full Lean build
are separate checkpoints. Bootstrap holds supplied alpha fixed; profile
curvature is local; TSV does not implement sample joins or a complete RNA
pipeline. These tests establish the documented cases and budgets, with no
general finite-sample calibration, global optimizer or universal performance
claim.

## Native SVG examples

examples/count-matrix-qc.svg and examples/bca-mean.svg are verbatim SVG fragments
emitted by the final source's synthetic run-pass fixtures. They illustrate
actual rendering output; statistical interpretation follows those fixture
contracts. No externally redrawn image is substituted for the native output.

## Runtime deadline and subnormal audit follow-through

Commit36188293819c5fb8db055a623775ae6cc95cdd11 adds the existing per-fixture
timeout120 to the full20000x512 normalization test. Its executable body, dataset
and assertions remain identical to fc78. Slurm16219 used the canonical runner
and recorded Pass1/Fail0/Skip0 in36.94s including compilation and execution.
normalization-timeout/ retains actual result JSON, transcript, resource and
source identities. The earlier16217 refusal occurred before numerical execution.
The722604KiB peak is for compiler+execution, separate from kernel80384KiB.

Slurm16245 investigated the earlier suggestion that literal1e-320 becomes zero.
That diagnosis was not reproduced on the exact protected compiler1148: native
output is positive and2024 minimum-subnormal quanta, matching independent C++23
bit pattern2024. Other selected subnormal literals are also positive. This is
selected input evidence, not a full proof of decimal conversion correctness.
The current Madaros parser already has its bignum/subnormal rounding path;
no compiler patch was needed. The fixture's explicit minimum-subnormal
construction names an exact boundary, rather than working around a demonstrated
current-parser defect. The corresponding historical ledger statement is corrected.

subnormal-audit/ retains the exact probe/reference text under .txt extensions,
native/C++ output, statuses and hashes. Full original sources and ELFs remain
on scratch at /scratch/sounio-stdlib-deep-20261023/subnormal_audit.

## Confirmed main acceptance

PR3052 merged at2026-10-23T21:39:24Z as
a8985fc5caeca87e04d22d6617a7bbdd9d02a1c2. The complete merge-group CI
40023582497 passed, including Linux/macOS native compiler gates, full suite,
Lean, contracts, witness and scientific replay. Managed Code Quality analyses
also completed for the source head before queue admission.

The fresh merge-group compiler has SHA256
174dd7796e28608c69f36f1c915126fa726ea53e0da6bdd1a904f26a23a62c29,
rather than the earlier1148 compiler used by the Slurm controls. The merged
stdlib blobs are identical to source361; the fresh queue gate validates them
with the updated main compiler. MERGE_GROUP_RESULTS.tsv records374 successes;
the changed-test runner separately passed39/39 including full normalization.
MAIN_ACCEPTANCE.json retains run/artifact/source identities.

Raw merge-group compiler/science artifact ZIPs are retained on remote scratch
under main-a898; their recomputed hashes match GitHub's published digests.
This documentation-only follow-up incorporates the receipt and corrects the
historical literal diagnosis while preserving numerical source and assertions.
