<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-170-named-into.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-170-named-into.readme
-->

# Reusable named-count streaming results

This increment is based on the published SIO-170 commit
7c4f9815e3a4b6ca1af94490e64e066481c28ad2. Its ancestry contains remote main
7ce5eddeef78c76c7e744150ed9c169c53a3d3b0. It changes the named framing adapter,
not identity, CSR, normalization or statistical policies.

## Concrete allocation change

The previous named adapter rebuilt a fixed byte array during feed/separator
calls, rebuilt the canonical scalar number parser at each feature row, and
returned a snapshot record at each feed. The new adapter retains one byte work
array and one canonical scalar parser state, and adds a caller-owned output
slot API. Every result path overwrites all ten fields. The original snapshot
functions delegate to the same implementation and retain detached values and
their signatures. The optimization includes using the new API at the call site.

Every numeric field is completed by feeding LF to the existing one-sample
count parser, then acknowledged before reuse. That parser's field validation,
integer overflow logic and reset remain canonical. Its internal position and
row counters cannot outrun accepted named bytes: numeric digits and separators
are a subset of the original stream, and the synthetic final LF at EOF is
covered by the nonempty feature ID and its separator. Original byte positions
and sticky errors come from the outer framing state. No numeric parser is
copied, and no partial scratch row is promoted to a complete event.

## Actual executable qualification

The canonical modular compiler is the source-built c21 CI artifact with SHA256
a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3.
The separately identified committed raw bootstrap seed has SHA256
8956946def36004d11ec4eca0d20c75c1e7d3d0b17ee0f0a1fdda9cd6ec0e4cc.
The raw seed runs from each correct frozen source directory; setting an import
environment alone does not establish its source selection. No new compiler is
claimed by this library change.

Slurm17383/source03 completed with STATUS0 in1m27. The focused gate executes:

- The unchanged named smoke and a reusable-slot fixture covering all fields,
  stale-value replacement, paused events, header completion, final row without
  LF, EOF idempotence, sticky post-EOF error and detached original snapshots.
- The original independent C++23 generator's15 tables at all256 chunk sizes,
  including512 samples, CRLF/split/EOF, raw ID bytes, integer maximum/overflow
  and absolute original error positions. Both original snapshot and new slot
  clients run on both native engines and compare to that same C++ oracle;
  the two clients also have byte-identical complete output.
- Long streams with100000 and1000000 rows, checking exact counts, original
  positions, header count, row count and an ID digest with C++23.
- Separate corrupted-position and missing-completion input controls, rejected
  with exit8 and9 respectively. A hostile SOUNIO_SOUC_BIN=/bin/false is cleared.
- Compiler, source, caller, reference and output ELF hashes, with unchanged
  source verification at the end and a fresh output directory.

Slurm17387/source04 repeats the same complete gate with STATUS0 in1m28 and the
final C++ resource summary mode added. Its million-row modular run takes2.43s
at1592KiB; the bootstrap takes0.59s at1612KiB. The summary performs no new
statistical estimation.

## Paired resource observations

Slurm17386/paired05 executes three AB/BA/AB pairs per compiler, pinned to the
same actually allowed CPU. Each process must exit0, pass the independent C++
oracle and produce byte-identical before/after output. The old source01 binary
uses the original snapshot API; the after source03 binary uses the new slot.
The file RESOURCES.tsv stores raw seconds/RSS/status. C++23 calculates medians,
maximum RSS and the ratio below; no Python/Rust/Julia computation is used.

| Engine | Rows | Before median seconds | After median seconds | Before maximum RSS KiB | After maximum RSS KiB | Before/after RSS |
|---|---:|---:|---:|---:|---:|---:|
| source-built modular |100000|0.36|0.24|254464|1636|155.5403423|
| committed raw seed |100000|0.06|0.06|1612|1636|0.9853300733|

These are whole-program observations at stated finite sizes. The seed has no
measured time benefit at the timer's resolution. Different compiler allocation
behavior is not inferred to be a specific ABI defect. The million-row modular
run in17383 completes in2.44s at1584KiB RSS; this is an after-only observation,
not a million-row before/after speed ratio or a universal memory guarantee.

The original15-table modular fixture uses4908KiB with the new snapshot wrapper
and3888KiB with the slot client in17383, versus the historical SIO-170 modular
fixture's419112KiB. That historical comparison is secondary to the paired
100000-row experiment and is not a controlled throughput claim.

## Retained attempts and limits

Slurm17380/source01 qualifies the100000-row original program on both engines;
17381/source02 qualifies the initial slot program on both. The first local-model
fixture candidate used invalid reference syntax and was rejected; it was not
accepted as a test. The fixture was corrected and checked by both real engines.
Paired attempt17385 was submitted before its prerequisite output binary existed
and fails at sha256sum before any timed run. It is retained separately, not
counted as a numerical or resource result. Paired05 starts only after the full
gate's prerequisite artifacts exist; the failed directory is never overwritten.

These tests do not certify biological calibration, compressed/network transport,
compact byte storage, bootstrap private-field reads or the full integrated
stdlib corpus. The existing SIO-170 evidence remains pinned to its original
source and original compiler limitations. Main merge remains human review.

The independently unpacked text-only custody archive `native-stream-text.tar.gz`
contains all successful scoped receipts and the failed premature paired attempt,
source callers, before/after module sources and a checked MANIFEST.sha256. It
is439023 bytes, SHA256
`31447b9309722910f56ef5bce7dfa81a65958556fbe49d6655e225871b09e2af`.
No compiler or ELF executable is included. Independent source review found no
blocker in the inspected state reuse/result overwrite paths; this is separate
from the executed gates and does not extend to untouched library modules.

## Additive runner registration

Slurm17389/source05 executes the exact five-line named-stream block extracted
from `scripts/stdlib/deep_science_gate.sh`, with the real caller environment,
parent record function and final source verification. Parent and child both
finish STATUS0 in1m29; parent records focused-gate/genomics-named-stream/0.
All pre-existing registrations stay present, and this block precedes the final
input hash check. This is a focused integration replay, not a claim that the
entire runner or all open PRs were executed together.

`central-stream-text.tar.gz` is327942 bytes, SHA256
`a2a336fb016e352c554b6db29dde4ce83cb77bbfc85496a786bb710c9a06b324`.
It retains the extracted block, actual runner source, source/compiler hashes,
child concordance/counterfactual results and checked MANIFEST.sha256, with no
executables. Documentation consistency/registry, shell syntax, source parity
against the frozen qualified packet, whitespace and independent extraction
checks pass before publication.
