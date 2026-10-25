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
finite/extended rational interval proofs from Draft3158. The final joint source
HEAD is recorded in the frozen execution packet, never inferred from a compiler.

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
No private-field CI step is claimed to have been imported.

The planned joint bootstrap uses old G0 SHA256
8956946def36004d11ec4eca0d20c75c1e7d3d0b17ee0f0a1fdda9cd6ec0e4cc
only if the controller's verified runtime pin matches; the authoritative full
G0 hash is the frozen compiler manifest. Fresh M1/M2/M3 output must match the
new tracked seed and the canonical/provenance gates. A prepared controller is
not a successful bootstrap.

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

These are separately receipted jobs, not results of the pending joint run:

| Package | Actual receipt | Scope |
| --- | --- | --- |
| Four scientific figures, Draft3154 | 17533/17534 | Both pinned engines, four workflows, rendering/export/roundtrip and ten specific controls per engine |
| Genomic batch, Draft3155 | 17535 | Six fixtures per engine, C++23 references and four specific controls |
| NB2 cache, Draft3153 | 17448 | Paired bounded workload; observed CPU ratios do not establish universal speed or memory gains |
| Historical calibration closure, Draft3156 | 17538 | Available-only descriptive report, original interrupted output and mutation controls |
| Refreshed compiler seed, Draft3119 | 17539 | Fresh fixed point, canonical/provenance/private-field checks; separate from current integration |
| Unbounded rational + / - / neg, Draft3158 | 17545 | Forty new Lean theorems plus thirty-two preserved finite theorems; conditional outward premises |
| Child executable custody | metadata controls only | Sixteen scripts audited; intact/tampered/missing artifact cases, no scientific execution claim |

The original 20000-gene/512-sample jobs17456/17472 qualify only the byte-preserved
experimental point-estimation composition with per-shard normalization.
They do not qualify global EB/FDR/bootstrap calibration. The generic scale
caller requires an allocated shard_start.txt input and is therefore check-only
in the universal E2E runner. Only its three-line annotation header changed;
scientific body/imports remain identical. The joint job must reproduce the
original missing-input refusal specifically before checking the new annotation.

Calibration job17048 exhausted its85500-second budget and ended FAILED124.
The closure records1152 planned,407 attempted,406 completed,405 available,
one model refusal,one timeout and745 not started. Descriptive available-only
intervals cannot establish family calibration. No reset or resubmission is
authorized by this consolidation.

The interval structural contract is not an IEEE interval primitive. Exact Lean
set enclosure, a separate C++ ISA witness and proof assumptions do not establish
frontend/IR/optimizer/backend implementation, RISC-V/RVV/PTX agreement, literal
rounding, uncertain control flow or a complete uncertainty/provenance feature.

## Joint execution state

The final source freeze, joint Slurm receipt and fresh full E2E results are
pending. The controller will retain source/head/catalogs, phase statuses,
runtime/toolchain and artifact hashes, logs and a terminal workspace comparison.
It runs in a durable host tmux controller with one recorded submission attempt,
one CPU,4GiB and a75-minute hard limit; no quota change, historical deletion,
main merge or campaign restart. Post-check reservation is not a guarantee of
completion before the scheduler's hard limit.

No statement in this document substitutes focused tests for full-suite
acceptance. Final actual results will be appended after the terminal receipt.
