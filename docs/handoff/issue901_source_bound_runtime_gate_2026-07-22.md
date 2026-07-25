<!-- docs:meta
topic_id: repo.docs.handoff.issue901-source-bound-runtime-gate-2026-07-22
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.issue901-source-bound-runtime-gate-2026-07-22
-->

# #901 Source-Bound Runtime Gate

## Decision

The imported-layout runtime acceptance surface has a direct nominal-layout
gate, a direct catalog-capacity gate, and one source-bound fixed-point gate:

```bash
# Behavioral raw-ELF gate. The caller supplies an ELF, so this alone is not
# evidence that the ELF came from the checkout under test.
MADAROS_RAW_BIN=/path/to/madaros.elf \
  bash scripts/ci/madaros_imported_runtime_acceptance_gate.sh

# Direct catalog boundary gate. In resolved mode, both 256 and 257 total
# layouts must compile to requested ELFs and execute their exact markers.
MADAROS_RAW_BIN=/path/to/madaros.elf \
SOUNIO_MADAROS_STRUCT_LAYOUT_CAPACITY_EXPECT=resolved \
  bash scripts/ci/madaros_struct_layout_capacity_gate.sh

# Source-bound gate. It requires a clean committed checkout, advances from the
# tracked operational Madaros seed twice, requires the two output ELFs to be
# SHA-256 identical, then invokes both direct raw-ELF gates against that fixed
# point.
bash scripts/ci/madaros_imported_runtime_source_fresh_gate.sh
```

The source-bound gate does not accept an override for its final Madaros ELF.
It is the build carried out in the current, clean checkout that establishes the
source-to-binary relation. A supplied binary can still be useful for a
behavioral replay, but must not be described as current-source evidence.

## What The Gates Check

`madaros_imported_runtime_acceptance_gate.sh` invokes the raw ELF directly via
`--native-v2-compile`; it does not use `bin/madaros`. It requires:

- an executable x86-64 `ET_EXEC` ELF;
- the typed nested-field witness to materialize an ELF, run successfully,
  print `520`, and print `ISSUE_901_NESTED_FIELD_CHAIN_OK`;
- the known-layout-miss witness to return nonzero during compilation and leave
  no requested output ELF; and
- the compile logs to contain no fallback or compact-imported-IR marker.

The prior wrapper-based negative check looked only for `a.out` below its local
work directory. The wrapper compiled into a separate temporary directory, so
the absence of that local file did not prove that the negative witness had not
materialized a native artifact.

The source-bound gate records a tab-separated receipt containing the committed
source `HEAD`, source tree, hashes of `main.sio`, `lean_single.sio`, the build
script, the tracked operational seed, and both Madaros generations. It accepts
only `stage1 == stage2` by SHA-256, then runs direct raw nominal-layout
acceptance and the resolved 256/257 catalog-capacity gate against the second
generation. It also compiles and executes direct raw contextual `scope`,
`policy`, `is`, and `study` binding witnesses against that second generation. The gate requires the checkout
to be clean before the build, after both builds, and after every direct raw
replay.

Both generations share a lock private to the gate work directory. The lock
serializes the two builds without making the source-to-binary claim depend on
an unrelated worker's global `/tmp` state.

This establishes an **operational Madaros fixed point**: a declared `M_n`
rebuilds current source to `M_(n+1)`, and `M_(n+1)` rebuilds identically to
`M_(n+2)`. It does not claim to repair the older C-to-lean bootstrap root. That
separate, heavyweight question remains `make madaros-root-audit` under #725.

## Evidence Before This Gate

On 2026-07-22, both candidate ELF inputs from the #901 worktree passed the old
wrapper gate, yet each direct negative compilation returned zero and emitted a
native ELF. This gate therefore rejects those inputs as intended. That is a
useful failure: #901 remains unproven until the compiler rejects the
known-layout miss before native artifact materialization.

No current-source full build was run in the workspace pod for this handoff.
That build is CPU-heavy and belongs on the Compiler Foundry or Slurm path.

## Foundry Handoff

Run the following from the committed candidate checkout on the Foundry/control
plane, preserving the generated temporary directory if a receipt needs to be
collected:

```bash
SOUNIO_MADAROS_ISSUE901_SOURCE_FRESH_KEEP=1 \
  bash scripts/ci/madaros_imported_runtime_source_fresh_gate.sh
```

Green evidence requires the final `PASS` receipt with a source `HEAD`, source
tree, seed SHA-256, matching stage-1/stage-2 SHA-256 values, the direct raw
#901 nominal-layout acceptance receipt, and direct raw execution of both
catalog-capacity witnesses. A green behavioral replay against an externally
supplied ELF is intentionally weaker and cannot close the current-source
runtime claim.
