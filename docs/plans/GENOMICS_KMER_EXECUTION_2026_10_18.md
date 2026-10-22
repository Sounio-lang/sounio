<!-- docs:meta
topic_id: repo.docs.plans.genomics-kmer-execution-2026-10-18
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.genomics-kmer-execution-2026-10-18
-->

# Streaming k-mer execution receipt — 2026-10-18

## Scope and source

Genomics continuation of the same statistics, scientific-visualization,
mathematics and physics sprint. Worktree /workspace/sounio-sprint-stats-viz,
original isolated baseline f5b2c5850be9f3d3995719ffcade44a6d851658a.
Remote main rechecked a1844465373c1ae5a335de2d0a73819e5cec2230.
Canonical checkout was observed on fix/exp-product-mean at9f24c028ba;
its local compiler build is excluded from this acceptance.
Changes remain uncommitted. No push/PR/merge, remote CI or current-source
compiler certification is claimed.

## Implemented

- New shared genomics::kmer rolling state, k1..4, fixed256 i64 histogram.
- Chunks0..256, scalar carry across arbitrary chunk boundaries; ambiguous
  bases outside0..3 reset the run and prevent crossing windows.
- Sticky invalid-k/length/state/overflow statuses. A failing base is atomic;
  earlier accepted bases of a failing chunk remain committed.
- Nonmutating empirical frequency view with an explicit no-window status.
- Public core checked adapter plus existing Kmer256 compatibility shape.
  Invalid legacy request is zero counts,k0; valid empty data retains its k.
- Embedding profile count consumes checked256-base chunks up to4096 inputs.
  Invalid bases no longer alias positional indices, and k/size are bounded.
- Public profile result/status and idempotent normalization.
- Typed Jensen-Shannon distance in bits accepts raw/normalized views,
  rejects unavailable/malformed/empty/incompatible profiles, and avoids
  cancellation near equality using positive symmetric contributions.
- Full PWM/motif/private-embedding suffix is preserved, with declared
  dimension/profile/empty guards. Its heuristic uncertainty and remaining
  projection behavior are outside this checkpoint.

[Stream contract](../../stdlib/genomics/KMER.md);
[adapter/profile contract](../../stdlib/genomics/KMER_PROFILES.md).

## Native acceptance

- Final restored candidate **15270** COMPLETED0:0,3:13: **64/64 pass**,
  zero failures/skips/errors, retaining all previous60 fixtures.
- Loss-of-carry control **15275** FAILED1:0,3:15:62 pass + two numerical
  runtime failures after compilation (reference and embedding consumers).
- Repeated-normalization control **15274** FAILED1:0,3:14:63 pass + one
  numerical runtime failure after compilation (embedding consumer).
- New fixtures: genomics_kmer_reference,genomics_kmer_status,
  genomics_kmer_legacy,genomics_kmer_embedding.

| Run | Job | Source archive SHA256 |
|---|---:|---|
| stats-viz-kmer-20261018-v2 |15270|2ea420b3b89b648a3aee4f98ec34008e00150526461d2934eba64d3bcba7c89d|
| stats-viz-kmer-boundary-20261018-v2 |15275|97cf87d11ba8e674276562a926cc6f0e2c8229ea7329922e693d32454c11f289|
| stats-viz-kmer-normalize-20261018-v2 |15274|87c64d4ae8557490aaa2606b3fc3940d428577c28c4ff62644463e39ed9170e3|

The boundary control deliberately discards incoming rolling/run carry at each
nonempty chunk. It tests the new streaming contract; it is not an untouched
historical k-mer module. The normalization control restores repeated division
of already-normalized frequencies, retaining the new checked shell. Each manifest
differs from the final candidate at only its one controlled source file.
Archived manifests were byte-compared with completed standalone packages.

Pinned committed Madaros v0.80 ELF SHA256:
125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f.
Gzip SHA256:
feaaf7fe181f1ee3efd30e7fe23b1709937ea7e93b52e812276fb093b6ccf52a.
Sounio builds/execution only through Forge/Slurm, GPUoff,twoCPUs,
8192MiB ceiling,five-minute job limit.

## Independent numerical reference

C++23 g++14.2, -std=c++23 -O2 -Wall -Wextra -pedantic, on t560.
Root re-executed both oracle executables with exit0. The window oracle's report
and generated fixture compare byte-for-byte with retained files/final package.

- kmer_reference.cpp enumerates every raw window of the entire prefix.
  It does not reuse the rolling implementation.
-72 unique snapshots,52 casefunctions; k1..4 and all12 split points of an
  eleven-base sequence containing two ambiguities. Prefix and final state
  compared, including all256 histogram bins and all scalar metadata.
-1280-base stream covers all256 four-base words separated by ambiguity;
  every256-base prefix is independently checked. Final per-bin formula and
  total counts have separate exact-integer C++ assertions.
- Native status fixture checks illegal sizes/k, inactive poison, ambiguous
  symbols, malformed selected bins/header, sticky errors, read-only frequencies,
  overflow atomicity and partial-chunk prefix acceptance at i64 limits.
- Legacy fixture retains existing k1/k2 examples and checks invalid/empty/
  ambiguous behavior through the public core.
- Embedding fixture separately enumerates windows for4096 inputs, checks
  boundary ambiguities, normalization repeat identity and raw/normalized distance
  combinations, mismatch/empty/malformed states and JSD identities.
- kmer_embedding_reference.cpp uses independent direct raw-KL long-double
  logarithms; near-equal four-bin profiles distance
  0.000293242816366117167131 is checked with relative1e-9 tolerance.

Oracle executable SHA256:
570240c3b08670cfa82af100344a846cb18d164ef6680bbe4f23f491b0cb915a;
JSD oracle:
8dcd282cfad1c7ff5ee58dcb3ca44b791d3a982eb225137671dfb90ecfe2c001.

## Review and preliminary run

Subagents owned the streaming core/contract, independent C++ generator and
profile fixture/reference/review. Root owned core/embedding integration,
status/legacy fixtures, controls and final Forge acceptance.

Independent review found that an early text-composition marker had omitted
the original PWM/motif/embedding suffix. It was restored before the final v2
snapshot. An exact comparison to the saved original suffix passes after removing
only the declared status/guard/comment changes; independent readback confirms
every prior PWM/motif/embedding function is present.

Preliminary15268/v1 completed0:0,3:06,64/64, but contains the incomplete source
composition and is **not** acceptance of the restored library. This demonstrates
why the fixture result alone was insufficient. Final v2 is required above.
The reviewer found no critical bounds/arithmetic/integration issue after repair.

## Execution/memory boundaries

The stream is O(n) for bounded k<=4 and uses one fixed histogram plus scalarstate,
without heap allocation or input-sized scratch. The4096-base adapter additionally
uses a256-base chunk and returned floating histogram; i64 counters widen storage
relative to the old i32 core wrapper. No lower-RSS or measured speedup is claimed.
Counts remain exact until checked i64 overflow; f64 probability conversions
round above the exact integer range. Public mutable state is not a validated
serialization format; headers and touched bins are checked, not every stream
history or histogram sum on every update.

The profile validator allows consistent weighted/fractional public fields and
floating mass tolerance; it does not certify observation provenance.
PWM, alignment, strand canonicalization, arbitrary k, uncertainty calibration,
whole-genomics coverage and whole-sprint completion remain outside acceptance.
Existing scientific renderer/statistics/math/physics fixtures remain selected.

## Custody and next work

`/Users/demetriosagourakis/Documents/Sounio-stdlib-sprint-2026-10-16/kmer-2026-10-18`.
Selected final sources, contracts, C++ reports, exact native run metadata/logs/
JUnit/manifests for all three final runs, tail-preservation report and master plan
are retained. Selected runtime files match the candidate MANIFEST.sha256;
DELIVERY.sha256 checks the local evidence set. The profile contract/receipt/plan
are documentation added after the executed source snapshot.

Next implementation is checked isothermal entropy and explicit standard-
uncertainty propagation; information-theory logarithm/domain contracts follow.
The larger scientific stdlib sprint remains active.
