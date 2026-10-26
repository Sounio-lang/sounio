<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-161.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-161.readme
-->

# SIO-161: scientific stdlib consolidation

[Notion epic SIO-161](https://app.notion.com/p/4033f37f937181028f25c45ef1ba6c0a)
is the execution backlog and records task ownership, dependencies, actual jobs,
publication and outstanding acceptance. This branch consolidates reviewed work;
human review and main merge remain separate gates.

## Source and compiler identity

The branch starts at remote main8eca8e0c086bbd074485bbfd8716accc4063ef1c
and includes main26c2a7a7e23c957acadb311d373d7d7f1d07e9b6, preserving KL4,
ADR020 and the V0-E.4.1 refusal classifier. It also preserves the reviewed
finite/extended rational interval proofs from Draft3158 and the exact product/
division model from Draft3164. The immutable joint execution source is
3c7c8073411a7e7254d0d9424f59f8e49eb49180. Later feature-branch custody/catalog
changes do not alter that executed source or qualify a new integration.

The modular raw compiler is the independently sourced ELF with SHA256
a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3.
It is an external pinned compiler, not a build of this current main.
The retained lean_single source has SHA256
702b4bf38036b11d61929a6135af5432de75859ff656361aa95c0dd2bd80b9a5.
Its refreshed tracked seed has SHA256
13003a8b9b1bf4bfc64ad8d13eb0795d722888fabef5bbb2913b4f7b83bb454b
and the matching committed SeedReceipt. The private-field source/seed/metadata
package from Draft3119 is imported by selected paths. Its CI workflow edit is
excluded because .github/workflows/ci.yml belongs to another active lane.
The separate scientific-contracts.yml workflow now invokes the private-field
contract and both interval proof gates directly; it does not modify the leased
ci.yml. Local workflow references, YAML parsing and the unchanged ratchet pass;
the hosted [interval proof](https://github.com/Sounio-lang/sounio/actions/runs/40239955961/job/121309423874)
and [private-field compiler contract](https://github.com/Sounio-lang/sounio/actions/runs/40239955961/job/121309423641)
jobs at source3c7 completed successfully. The [full CI run](https://github.com/Sounio-lang/sounio/actions/runs/40239955915)
remains failed, including Contracts/Core and the f64 lowering lane; its Stdlib
Deep Science job was cancelled. Two green contracts are not full CI acceptance.

The executed joint bootstrap used old G0 SHA256
8956946def36004d11ec4eca0d20c75c1e7d3d0b17ee0f0a1fdda9cd6ec0e4cc
with the verified frozen compiler manifest. Slurm17578 bootstrap status is0;
fresh M1/M2/M3 each match tracked seed13003a8b9b1bf4bfc64ad8d13eb0795d722888fabef5bbb2913b4f7b83bb454b.
Private-field/canonical/provenance and final integrity phases passed. These
successful phases do not change the overall failed joint result.

## Integrated validation paths

The deep science runner retains its original direct native/C++23 concordance
and now invokes eighteen explicit focused children: checked BDF, noncentral-t
power, count identities, named streams, Wilcoxon, survival, FEM Poisson, DP54
dense events, prepared contours, canonical palettes, FFT, numerical derivatives,
physics uncertainty, checked distributions, genomic batches, four scientific
figures, the interval structural contract and phonon factor transport.

Full source catalogs are checked before/after, including wrappers, callers and
the original phonon witnesses. All eighteen children must produce nonempty
ELFS.sha256 catalogs. The parent checks their artifacts again after every child,
along with its direct native/C++ references, compiler pins and the C++ toolchain
helpers/runtime. Added or removed source files invalidate the catalog. Compiler
error diagnostics cannot become a positive pass merely through exit status0.
This retained-artifact contract covers the deep gate; generic bin/madaros run
uses a temporary executable and does not retain that ELF.

The generic E2E runner uses the Node metadata emitter from Draft3157. Its
historical JSON is only a pinned metadata selftest fixture. A fresh job must
attempt the complete unfiltered frozen tests/stdlib census, emit new JSON,
reconcile pass/fail/skip and failure/ignore projections, and record its own
timestamp and runner exit. A nonempty subset is insufficient. Failing suites
remain failures even when their JSON is valid.

## Previously executed evidence and limits

These are separately receipted jobs with scope distinct from the terminal joint result:

| Package | Actual receipt | Scope |
| --- | --- | --- |
| Four scientific figures, Draft3154 | 17533/17534 | Both pinned engines, four workflows, rendering/export/roundtrip and ten specific controls per engine |
| Genomic batch, Draft3155 | 17535 | Six fixtures per engine, C++23 references and four specific controls |
| NB2 cache, Draft3153 | 17448 | Paired bounded workload; observed CPU ratios do not establish universal speed or memory gains |
| Historical calibration closure, Draft3156 | 17538 | Available-only descriptive report, original interrupted output and mutation controls |
| Refreshed compiler seed, Draft3119 | 17539 | Fresh fixed point, canonical/provenance/private-field checks; separate from current integration |
| Unbounded rational + / - / neg, Draft3158 | 17545 | Forty new Lean theorems plus thirty-two preserved finite theorems; conditional outward premises |
| Unbounded rational multiplication/division, Draft3164 | 17573 | Fifty-three fresh Lean audits and twelve mathematical mutant refusals; unchanged finite/additive dependencies |
| Syscall witness controls | 17575 | Six actual operational C++ controls, no scientific computation; requested one CPU, scheduler allocated two |
| Child executable custody | metadata controls only | Sixteen scripts audited; intact/tampered/missing artifact cases, no scientific execution claim |

The original 20000-gene/512-sample jobs17456/17472 qualify only the byte-preserved
experimental point-estimation composition with per-shard normalization.
They do not qualify global EB/FDR/bootstrap calibration. The generic scale
caller requires an allocated shard_start.txt input and is therefore check-only
in the universal E2E runner. Only its three-line annotation header changed;
scientific body/imports remain identical. Slurm17578 passed the original
missing-input/header probe and final original-source comparison. This is not
an executed20000-gene universal-E2E result.

Calibration job17048 exhausted its85500-second budget and ended FAILED124.
The closure records1152 planned,407 attempted,406 completed,405 available,
one model refusal,one timeout and745 not started. Descriptive available-only
intervals cannot establish family calibration. No reset or resubmission is
authorized by this consolidation.

The interval structural contract is not an IEEE interval primitive. General
rational enclosure for all four operations now has its own proof receipt;
general tightness and emitted IEEE instructions remain outside that result. Exact Lean
set enclosure, a separate C++ ISA witness and proof assumptions do not establish
frontend/IR/optimizer/backend implementation, RISC-V/RVV/PTX agreement, literal
rounding, uncertain control flow or a complete uncertainty/provenance feature.

## Terminal joint execution and separate follow-ups

Slurm17578 ended **FAILED1:0 in56m29s**, deep_rc=1 and e2e_rc=1. Its fresh
E2E JSON is valid:627 records,315 pass,99 fail,213 skip. Pass modes are272 run
and43 check-only. The measured JSON SHA256 is
1d75374a4dac5b9cfcbd0d5013dfdeffdfd8265ff472adf3cd8e33ca276b33fe.
Valid JSON preserves every failure; it does not make the suite pass.

Seventeen of eighteen focused children passed. DP54 refused before execution:
staging omitted docs/evidence/stdlib-SIO-174/before/callback_time.sio; its child
ELF catalog was consequently unavailable. Final source/catalog/compiler/runtime/
C++ checks and the terminal workspace comparison passed:10159 entries match
source3c7. The host controller records worker=1, workspace=0, overall=1. Its first
transport/monitor attempt failed DNS resolution; that failure remains beside the
successful reconciliation and did not authorize a second submission.

The REVIEW04 controller SHA256 remains
58a09ea215d11de7a2c0158f8d5dcd62124673f662b7687ee55037959bfd3b52.
Its17 synthetic controls and separate6 actual syscall controls in17575 remain
bounded controls. REVIEW03 failed before staging on the Git output buffer;
REVIEW04 bounded metadata at64MiB and read all24468 tracked paths. Failed/obsolete
attempts remain preserved. Joint reservation: one CPU requested,4GiB,75min.
No quota increase, deletion or campaign restart was used.

Slurm17589 is a **separate DP54 supplement, COMPLETED0:0 in28s**. It adds the
missing original callback witness to the explicit source catalog, retains the
original missing-input refusal and passes both engines, native/C++ concordance,
DP54 marker and final source/executable/toolchain checks. One CPU requested,
two allocated,4GiB. It does not rewrite17578 to pass or rerun the joint suite.
Reviewed PR3168/4dbb45c24a6d03dde82af0ea7b9e56f2f401e111 imports this required
historical-input catalog and nonempty formal custody guards after the frozen run.

The read-only99-case audit compares resolved imports/literal resources against
main26c: all99 fixtures are identical,90 reconstructed complete source/resource
closures are identical,9 remain unresolved (one has a changed resolved subset).
This is source parity, not an executed main baseline. The six-line excerpts and
combined compile/run30s wrapper leave many first failure phases unresolved.
The audit proves17 tracked Pireus inputs,2179580 bytes, absent from the original
stage across five failed fixtures. Their missing-file admission/lineage guards
are real; the truncated logs do not establish each recorded first failure cause.
Exact paths and the UID1000 read-only existence probe remain in the audit packet.

Slurm17597 is an **interrupted partial full-Git E2E attempt**, not a new627-test
baseline. Its fixed source was original3c7c8073411a7e7254d0d9424f59f8e49eb49180:
24468 Git blobs/modes,15 symlinks, no gitlinks and922928729 bytes were verified
before submission, including the17 omitted inputs and10159 unchanged prior
entries. The complete tree also restored executable tools/science_boundary,
which changed the operational path beyond those17 data files. The existing
wrapper invoked Python attestation in auto mode. No later root source was run.

Root directed a graceful stop under ADR009. The exact command
`scancel --full --signal=TERM 17597` returned0 at2026-10-26T00:28:42Z;
Slurm records **FAILED0:15 after19m34s**, ending00:28:43Z. Only472 original TSV
rows remain, SHA256
406360020d2ca3932bba60508de0c6cc29994be9901f417e343dd296bac05ddd.
No final JSON was emitted. There is no valid complete627 measurement or full
metadata reproduction, no old-row merge, synthetic completion, fallback or
resubmission. The original17578 counts remain a distinct selected-input result.

The first observed Python process was PID852789, evaluating the XOR material
matching fixture with --compiler naming the external a292 .ready ELF. Its
observed Slurm elapsed17:00 and recorded Start00:09:09 imply00:26:09Z at
second resolution; no separate wall-clock stamp was captured at that first
observation, and this is not evidence of the first-ever invocation. The exact
process line and timing qualification remain in the stop packet.

Post-stop regular-byte checks passed for all24453 tracked regular files.
Workspace-only analysis of the harvested physical inventory also matched all
24468 tracked modes and15 symlink targets/Git blob hashes. The strict path
census nevertheless **failed**:24470 physical entries include two additional
files, bin/madaros-linux-x86_64 and bin/.madaros-linux-x86_64.verified. The
extra ELF is125051232 bytes, SHA256
125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f;
the stamp is124 bytes. They remain preserved outside Git; no files were deleted
to force a pass. The strict source-before/after helper could not produce its
post-catalog after the failed path census. Engines/Node/runtime/CXX/packet
hash checks passed, separately from this nonzero census result.

Frozen bin/souc and bin/madaros unconditionally call the prebuilt materializer
even with an explicit raw override. That helper locally decompresses the
tracked .gz and verifies its tracked125c hash; no external download is established.
Static resolution prefers MADAROS_RAW_BIN, and captured check/attestation
commands name a292. These facts do not constitute a per-case execve audit;
unchanged pin hashes alone do not prove the compiler actually selected for
every E2E case, and the extra ELF alone does not prove it was executed. The
whole-tools activation and incidental materialization are confounds, not a
qualified canonical baseline. Any permitted future replay must replace prohibited
auxiliaries, bind the actual raw route and bind the exact expected materialized
files. Default materialization/provenance remains intact; novel paths must
still fail closed.
No Sounio tests, wrappers, attestors or scientific execution ran after stop;
only the authorized read-only metadata harvest and workspace analysis followed.
The separate SIO282 port and any future execution need their own receipts.

Imported PR3167/e2cce2752796389a1b283a46def5de94fa9630ca provides lossless
evidence storage/readers, a tree/storage change without new numerical execution.
PR3170 remains separate: global-reference in-process normalization has focused
17586 evidence and additive19th central registration80bacadc111304766b72ae2c3a324a8e8344cf90.
Neither17578 nor the old3c7 full-Git replay qualifies that increment. Global20k,
transport/out-of-core, global prior/FDR and phase5 global runtime remain outside
that focused result.

SIO281 phase separation has independent custody: six failing-before regression
controls and twenty-six after-controls are retained. Slurm 17596 completed 0:0
in 47s with two original fixtures, 2 pass / 0 fail / 0 skip, using checkpoint
95b010b0e3cd3b765b2acc0a7dd25cf643e9bd0a and scientific base 3c7. Compilation
and native execution both returned 0. This serial replay used one parallel worker;
the hosted f64 job used four. It does not reproduce hosted status 124, prove its
cause, qualify the full ChangedTests selection or make full CI green. The two
SIO281 packets are metadata/control receipts without native ELF payloads.

## Compact custody stored here

Only compact text/log/catalog/control receipts are committed. Full source snapshots, ELF payloads
and the full original worker archive remain outside Git. Original source/toolchain
hashes, failures, statuses and logs remain in these packets; hashes of excluded
payloads do not mean their bytes are included.

| Packet | Bytes | SHA256 |
| --- | ---: | --- |
| joint17578-text.tar.gz |3357987|341095285e99d7780fecd7d9cc9c7e1dd74aac09658f8d5f1f14b13bb996c208|
| dp54-addon17589-text.tar.gz |4019988|45026758cbe794172ec6c885b23617910f2a7dfe936833a4bbdf9453c9873137|
| joint17578-host-terminal.tar.gz |2115|e833073319e087152ea6f74ceaa46dab55432f93a65992e8a755add9984bbaaa|
| e2e99-source-audit-text.tar.gz |86653|dc55b5d66848ae1339a432de7d2441a01182fd5dba15bd0266e47851b6522d7b|
| native-phases17596-text.tar.gz |288272|7145fa7ef07a265b05066f5aed6ded731953ae6b913e114f63c3c8e330431fa5|
| native-phases-checkpoint02-text.tar.gz |50916|e6a7ecc4d3a4966eab6a604847ccf6bce9caf12ce2516356df9d9c9990a460c4|
| e2e17597-stopped-text.tar.gz |5245791|eae1515e1a53f151dd7e57b1eac823a2a4878209663db711b850e4c5719b326c|

Before copying, fresh extraction verified284 joint compact entries,18 terminal
host entries,14 audit entries and306 available DP54 artifact entries. Fresh
SIO281 extraction additionally verified108 replay receipt entries and852
checkpoint02 entries. The stopped17597 packet additionally verified120 included
text/catalog/metadata members, with no source bulk or ELF payload. Fourteen
DP54 executable payload entries are intentionally absent from the compact packet;
their original worker receipt remains preserved. No included mismatch, ELF payload
or source archive was found. CUSTODY_VERIFICATION.json records checks/identities;
FILES.sha256 covers this README, six unchanged packets, the separate stopped
attempt packet, verification receipt and internal fragment. Focused passes,
source parity, green hosted contracts, tree-only custody and an interrupted
partial attempt remain distinct from full-suite
acceptance and human review/main merge.

## Hosted replay of PR head26ecd

Main CI40248656923 ended **CANCELLED**. The provider annotation explicitly
reports the Deep Science job exceeded45m0s, matching the configured bound.
CI Decision then failed its required-deep check; cancellation is not a pass.
The job ran00:25:59Z–01:11:16Z, replay00:26:34Z–01:11:12Z.

The PR head is26ecd3cd1bea2a70b7166f85c32c676d1d01b099; actual tested GitHub
merge is54e02c8f1f755f322b3e262a1f3bddb18df627f7 over maina886ce9bdb9b6b9ca4f748028fca0248bfa9f125.
Its source-bound compiler ELF is
b30def13149e1d95d07dde2cf107a82c7bc99e8ec8c910d8b7f6098f5ef8d2c9,
not the a292 pin of the separate Slurm qualifications. The workflow and four
relevant scripts were matched to the tested compiler input catalog.

F64 succeeded: ChangedTests12 pass/0 fail/0 skip, including one compile-fail.
Eleven positive split receipts record compile0/native0/result0 with independent
30s/30s limits. The additional17 known failures remain known, not new passes.
This observation does not assign the result solely to phase separation: both
the tested merge/compiler and protocol differ from the earlier hosted failure.

Deep retained509 completed rows with status0 and15/18 focused children. It
entered child16 scientific-native-workflows; its inner subphase is unknown
because the existing nonrecursive upload omitted child directories. Interval17
and phonon18 did not complete, and final custody/STATUS were not reached. The
separate global3170 increment is absent from this tested source, not a19th
unreached child. Eighteen calls and the required-completion guard are preserved.

The compact hosted26ecd-terminal-text.tar.gz is1272198 bytes, SHA256
22df6d903d9d50119beaea3b003e4b50ea5b1772febd6aa004b8cb4b31ee9adf;
324 included hashes passed fresh extraction. It contains the report, cancellation
annotation, prefix records, selected phase receipts and source/compiler catalog.
The11 positive receipts were retrieved as303 text members by HTTP ranges,
without downloading the84MB artifact/ELF ZIP. No ELF or source bulk is included.

A reviewable text-only recursive collector and parent start/finish ledger are
being prepared for the active CI-workflow owner. No deadline, tolerance, gate
selection or CI decision rule is changed by this custody update.
