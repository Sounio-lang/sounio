<!-- docs:meta
topic_id: repo.docs.plans.compiler-path-attribution-execution-2026-10-22
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.compiler-path-attribution-execution-2026-10-22
-->

# Compiler pathname attribution checkpoint — 2026-10-22

## Scope and source identity

Continuation branch: `sprint/compiler-path-attribution-main-a59930-20261022`.
Base: remote main `a59930e51462528d86767d5e6647b1689ef76f80`.
The unchanged observer was cherry-picked from3860980 as a950e4ff0b; its historical
execution receipt and compact evidence were restored from5c84f28288. This records
history, not an execution against the new main compiler.

This increment adds a C++23 analyzer and analyzes the retained Slurm15940 trace
from compiler source `eeeab3025c68bebfdbde284116e41176b9976770`. It does not rebuild
main's changed compiler/seed or run numerical examples/tests. The previously
executed NB2 example remains historical evidence at its recorded source identity.

## Executed analysis

The analyzer compiled with `g++ -std=c++23 -O2 -Wall -Wextra -Wpedantic` on t560;
final compiler diagnostics were empty. Standalone text analysis ran there with
exit0. No Sounio build ran on t560, the workspace or the notebook in this increment.
The original compiler execution was Slurm15940; no new Slurm job was needed to
analyze its already-retained trace.

| Observed counter | Final value |
|---|---:|
| Physical trace lines | 25292 |
| Reconstructed complete records, including signals/exits | 24694 |
| Joined unfinished/resumed records | 598 |
| Unparsed records | 0 |
| Orphan/mismatched resumes | 0 |
| Replaced pending records | 0 |
| Pending records at EOF | 0 |
| Requested pathname decode failures | 0 |
| Syscall rows without known exec-request image | 1210 |
| Unsuccessful exec requests | 8 |

| Exact-path role hint | Syscall rows | `-1 errno` rows | Decoded path operands |
|---|---:|---:|---:|
| Operator-listed compiler exec requests | 1906 | 761 | 1904 |
| `/usr/bin/sha256sum` exec request | 15177 | 19 | 15158 |
| Other known exec requests | 6106 | 253 | 5824 |
| Unknown exec request | 1210 | 178 | 850 |

Compiler role hints were configured using the exact observed bootstrap and
transient generated-seed request pathnames; they are not content attestations.
Successful exec changes the epoch; a failed attempt preserves the previous hint.
Fork inheritance is deliberately not inferred. The unknown row stays unknown.
Role totals count syscall rows; signals/exits are present in the original trace
and included in complete-record count, not in syscall-role totals.

The private absolute-source candidate set has325 distinct lexical spellings in
the compiler-hint group. This is **not325 dependencies**: only literal absolute
SOURCE_ROOT prefixes are included, with no normalization or relative/dirfd/cwd
resolution. Failed lookups, metadata operations and mutations are included.
The private operand ledger preserves arguments and source line references.

Resource receipt for this analyzer execution: elapsed0.44s, user0.20s,
system0.02s, maxRSS3252KiB, exit0. These are text-analysis observations, not compiler
or renderer benchmark results. No speedup claim follows from the observed ratio
of hash-utility events to compiler-hint events: syscall counts are not elapsed
cost, byte volume or dispensability.

## Review and custody

Independent read-only source review checked resumed-call reconstruction, strict
hex-byte decoding, failed-exec preservation, successful-exec epoch transitions,
pending-record retention, bounded line reading and balanced argument termination.
It reported no serious blocker for the stated format/scope. That review did not
execute the analyzer or independently recalculate its counters.

Development executions exposed padded `exit_group` records and errno-message
parentheses; the final parser finds the balanced syscall argument terminator.
Early intermediate summaries were superseded. Published counters and hashes
refer only to the final invocation retained by `run-path-attribution.sh`.

Private custody on t560:
`/home/devsounio/sounio-artifact-archive/compiler-path-attribution-20261022`.
It retains analyzer source/binary, compiler version, exact invocation recipe,
input identities, final raw tables and deterministic gzip copies. Every archive
was decompressed and hashed against its original table; all readbacks matched.
The source and binary hashes are in [IDENTITIES.txt](../evidence/compiler-path-attribution-2026-10-22/IDENTITIES.txt).
Public compact [evidence](../evidence/compiler-path-attribution-2026-10-22/README.md)
contains counter/resource receipts and derived-table identities, not raw paths or
per-task records. Host-local custody is not an independent physical backup.

No new source snapshot, shared-volume compiler artifact, historical deletion,
quota increase, or cache policy change was performed. The3TiB quota remains.

## Main integration queue — preserved context

Remote main incorporated earlier sprint work and changed compiler source/seed;
merging the later scientific branch wholesale produces genuine conflicts.
The prior branch `sprint/compiler-input-observation-main-f02db-20261022` remains
intact at5c84f28288. Its newer logarithmic kernels, NB2 diagnostics/deviance,
bar rendering and uncertainty geometry have **not been dropped** and have not
all been integrated into this main-based continuation.

Next scientific integration must reconcile genomics normalization, probability/
channels, entropy/thermo, linear/NB/Poisson/Shapiro statistics, IR/canvas/HTML and
the runtime packager, preserving those later improvements. Read-only comparison
also found upstream corrections to preserve:

- Explicit `beta_new` import in linear regression (`fae437e8c0`).
- Shapiro `sw_exp` wrapper around the shared exponential (`fae437e8c0`).
- Direct declarations/loop in `boltzmann_entropy` replacing a block expression
  associated with a compiler execution failure (`f8c419b606`).
- New compiler decimal/rounding/register fixes and corresponding seed identity.

A compiler rebuilt from the eventual merged source is required before native
claims about that source. The old NB2 receipt does not cover the new main compiler.
This checkpoint closes path-event attribution analysis only; scientific integration,
full input closure, numerical concordance, complete corpus, CI and merge remain
separate open checkpoints.

## Limits and next optimization decision

`candidate_only=true`, `complete_input_closure=false`, `cache_authority=false`.
No image inheritance, cwd/dirfd resolution, descriptor-content tracking, syscall
semantic closure or generated-seed content recovery is claimed. Zero parse failures
only concerns reconstruction/decoding of this supported trace format. The selected
syscalls omit content-consumption evidence and inherited/memory/network inputs.

The observation now separates utility hashing from compiler-path hints, making a
narrower-key proposal reviewable later. Before proposing such a key, bind generated
intermediate contents, reconcile negative lookups and filesystem resolution, and
identify the actual consumed source closure. Until then the broad key and guards
remain authority. See [analyzer contract](../../scripts/dev/ATTRIBUTE_MADAROS_TRACE.md).
