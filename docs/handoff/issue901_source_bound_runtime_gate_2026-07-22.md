# #901 Source-Bound Runtime Gate

## Decision

The imported-layout runtime acceptance surface has two deliberately separate
gates:

```bash
# Behavioral raw-ELF gate. The caller supplies an ELF, so this alone is not
# evidence that the ELF came from the checkout under test.
MADAROS_RAW_BIN=/path/to/madaros.elf \
  bash scripts/ci/madaros_imported_runtime_acceptance_gate.sh

# Source-bound gate. It requires a clean committed checkout, builds a new
# Madaros ELF with scripts/ci/build_modular_madaros.sh, then invokes the raw-ELF
# gate against exactly that output.
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
script, and the resulting raw Madaros ELF. It also requires the checkout to be
clean before the build, after the build, and after the direct raw acceptance
replay. The receipt identifies a source-build event; it does not remove the bootstrap root of trust used by
`build_modular_madaros.sh`.

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
tree, and raw ELF SHA-256, plus the direct raw #901 acceptance receipt. A green
behavioral replay against an externally supplied ELF is intentionally weaker
and cannot close the current-source runtime claim.
