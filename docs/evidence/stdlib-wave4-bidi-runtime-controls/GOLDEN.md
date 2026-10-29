# Immutable metadata for the official BIDI corpus

Slurm19637 completed0. All214 files from the completed19618 corpus were compared byte-for-byte with coherent-text-focused01/bidi-corpus, whose durable archive has SHA-25619b5e632c3adb31942c181e22941c86e3258ec9d37de58548ce4e6182ba15442. Every file matched; no corpus file, case, expected level or ICU policy was changed.

The manifest binds all211 numbered chunks, CHUNKS.tsv, COUNTS.json and SMOKE.bin by exact bytes and SHA-256. The official producer reports861948 cases, including770241 class cases and91707 character cases. These counts are metadata copied from the existing C++23 output; no Unicode computation is implemented in JavaScript.

source_files pins the exact UCD archive, UCD source checksum list and existing C++23 generator/header from the qualified19618 snapshot. upstream_files records every extracted UCD source and verifies its checksum. Provenance identifies the original qualified program, raw03, source snapshot, old corpus and old custody archive. The source snapshot explicitly includes recorded WIP and is not represented as a clean revision.

The raw candidate and its original checksum remain in services/bidi-protocol50/results. Before publication, source_files and upstream_files were sorted lexicographically by path to make the metadata schema canonical. Counts and all214 corpus files are unchanged.

Published official-corpus.json SHA-256:
`b428aba82a8ed4da0b3e21f93a5228761c945d18769900acd153d6e961408eb5`

This manifest binds scientific inputs produced by the existing C++23 generator. It is not a new resolver, a new conformance execution or qualification of the still-pending canonical fixture hook. The independent negative protocol controls are recorded separately.
