# One ELF: full Unicode, UCA, CLDR and BIDI native replay

Slurm **19285 COMPLETED 0:0**, with verdict
`COHERENT_TEXT_FULL_NATIVE_ONLY_OK`. Every listed collector accepted complete
coverage, the bound invocation kind and the same executable SHA-256:

`2bd52fdededc355138c796d6de92747d5a84c1e7764c9e10f6a69d9560048454`.

| Original corpus | Chunks | Rows/cases | Explicit ELF argument |
| --- | ---: | ---: | ---: |
| Unicode | 21,971 | 11,248,160 | 1 |
| UCA | 2,307 | 1,180,750 | 1 |
| CLDR defaults | 763 | 390,251 | 2 |
| CLDR options | 4,154 | 2,126,400 | 2 |
| CLDR locale metadata | 72 | 9,190 | 3 |
| BIDI | 211 | 861,948 | 4 |

The ELF was already compiled and sealed in 19270. This job replays the complete
original corpora; it does not rebuild or replace the ELF. The materializer,
original-body reverse equality ledger, dispatcher, focused argc/kind controls
and source archive are recorded in [coherent-text](../coherent-text/README.md).
The compiled-source manifest SHA-256 is
`fcbaca9a807c7d264815edd920798d35a875dc4ff3f6edf28dd268e369681049`.
Its raw compiler is
`3c39904b0e811059211b429277112a4e02b507f36d4dff9dd8173360a3bfd784`;
the platform library is
`de01597868c09725c2c26651d23203043c65fdc04a3e5e96f45dfe4a51d494c4`.
The dispatcher selects the family by the explicit argument; input-file presence
and current-directory collisions do not choose a family.

## BIDI runner and controls

`tools/unicode/run_bidi_coherent_chunks.cpp` runs the actual ELF directly with
one argument, `4`, through the existing native execution helper. It accepts
only the full original manifest of 211 chunks and 861,948 cases, even when
executing one partition. The per-part receipt binds the externally measured executable digest to the
runner-recorded argc and kind, chunk bytes, exit status, unique success marker
and observed case count. The collector checks invocation sidecars, parent executable bindings,
partition topology, unique fixture coverage and exact full-manifest totals
before creating a fresh summary directory.

The full run preserves the original generator, corpus, expected levels and
original Sounio BIDI body. Its controls reject:

- wrong kind and missing runner arguments, with status 64 before output;
- incomplete or duplicate partition collections;
- tampered kind, argc or executable binding;
- an independently generated wrong-level fixture, with native status 1,
  `BIDI_LEVEL_MISMATCH` and no BIDI success marker.

The complete positive BIDI collector is recorded in
[bidi-summary/STATUS](bidi-summary/STATUS). The same-executable scalar-family
bindings and public adapter controls remain in the preceding coherent-text
evidence. No wrapper executable substitutes for the ELF identity.

## Custody and replay

The exact Slurm script is [worker.sbatch](worker.sbatch). The phase outcomes,
input manifests, tool and runtime identities, controller accounting and final
guard-file digests are in [results](results/). The complete before/after guard
contents and all native per-chunk outputs are in the durable archive:

`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/coherent-text01/full-native-source-receipts-19278-19285.tar.gz`

SHA-256:
`3b2510af22773ade6894d1e30f14ca8ea40c382175acc9f332f1c46313afc893`.

Slurm 19424 archived and read back that file successfully. Its receipts are in
[durable](durable/). The archive includes the failed 19278 attempt and the
complete 19285 working tree. The earlier attempt failed because the worker
requested `000000.bin` while the canonical wrong-level generator writes
`wrong-level.bin`; no completed corpus family was accepted by that attempt.
The correction changed only that operational path in a fresh run directory.

19285 checked source-manifest, original compiled sources, runtime, ELF, tools
and every original corpus byte list before the final verdict. The checked-in
`FINAL_GUARD_FILES.sha256` is a compact index of the full final verification
files; their contents are retained in the archive. Those byte guards and the
native per-chunk deadlines were not relaxed.

Controller accounting reports 01:58:22 elapsed with 8 CPUs and 16 GiB requested.
The administrator extended only the outer Slurm walltime from two to three
hours while the job ran; its original attempted update refusal and successful
administrator before/update/after receipt are preserved. The job finished
before its original two-hour limit. Per-part `time -v` logs in the archive are
resource observations for those commands, not an aggregate peak-RSS or speedup
claim.

## Explicit limits

`PRISTINE_STRICT_STATUS` is still **1**. The pristine locked ICU strict
comparison remains **FAIL**. The separately qualified, isolated ICU candidate
and its normative analysis are documented in
[bidi-icu-complete](../bidi-icu-complete/README.md); that candidate does not
replace the primary oracle or corpus.

This result qualifies native conformance of this sealed, coupled executable
against the listed complete corpora. It does not qualify the later integrated
root compiler, every stdlib module, generic HeapVec/HeapDeque or the proposed
typed-storage authority. It does not establish an allocation-free text API.
