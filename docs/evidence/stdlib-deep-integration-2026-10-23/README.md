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
