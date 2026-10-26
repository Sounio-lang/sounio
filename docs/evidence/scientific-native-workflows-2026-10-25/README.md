<!-- docs:meta
topic_id: repo.docs.evidence.scientific-native-workflows-2026-10-25.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.scientific-native-workflows-2026-10-25.readme
-->

# Native scientific workflows: computation to four figures

The four-workflow qualification is complete for the retained gate06 composition
and the separately frozen current-main integration. Source, compiler and actual
artifact identities are recorded below. No release, merge, calibration or
universal backend-equivalence claim is made.

The producer examples and their precise scientific limits are described in
[the example guide](../../../examples/science_native/README.md). The independent
reference is `tests/reference/scientific_native_workflows.cpp`; the fresh-output
worker gate is `scripts/stdlib/scientific_native_workflows_gate.sh`.

The dependency composition starts at remote main
`7ce5eddeef78c76c7e744150ed9c169c53a3d3b0` and merges the reviewed named170
`8a290a015aba14d7b14a0e0b5d30e7cfddcf5133`, survival169
`1d3058b45dda56c309855ba50acb0f3ca529bfc3` (already including162/168),
FEM/Poisson173 `c05f9375a0b5294604afbb14fd97a7bb15cc7010`, DP54174
`b2b1815c4c1c40a35ad0af67a0221476b49b2420`, prepared contours175
`ea176b53b4f764c9ee2acd7fd26468e56cf20c6a`, and palettes176
`ef28dbf017eb3978542bbd8dc687d74e1e5e3573`. The resulting dependency head is
`0708a91d666d698bd3ac48d67770ee3a3e1dd28b` (the full hash is retained in the composition receipt). Every
changed production file matches its reviewed branch byte-for-byte. The only
merge conflicts were additive registrations in the central gate; all eight
focused registrations were preserved. No active SIO-171 production was edited.

Only new consumers, the C++ reader, a focused replay gate, documentation and
compact evidence are added on top of that composition. Actual compiler source,
raw binary, caller code, execution, figure checks and visual observations remain
separate identities and evidence categories.

## Retained native qualification

The frozen gate06 completed as Slurm job **17451**, `COMPLETED`, exit `0:0`,
elapsed `00:00:59`. Its worker root is
`/scratch/sounio-stdlib-inference-scale-20261024/science_native_gate06`.
All four programs passed check, compile, native execution, independent C++23
science/geometry checks and actual-Canvas PNG/PPM roundtrip on both retained
compiler pins. Ten distinct changed-input controls per engine were rejected
with the required exit code and diagnostic. The strengthened caption guard
also rejected the real old normalization output from job17433 in both engines.

The compilers are retained, previously qualified executables, **not builds from
this consumer composition**: source-built Madaros
`a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3`
and strict lean seed
`8956946def36004d11ec4eca0d20c75c1e7d3d0b17ee0f0a1fdda9cd6ec0e4cc`.
The gate verifies complete stdlib/consumer/wrapper/reference input hashes and
compiler hashes after execution. Each native program ELF is hashed before run.
Reference/exporter ELF hashes in the original bundle are explicitly post-run
identities; they do not establish a pre-run hash check that was not performed.

`workflow-evidence.tar.gz` is the byte-identical copy of worker
`workflow-evidence-final02.tar.gz`, SHA-256
`36ddcad7a21e9fd2b0a92b06208e0d5834d53223d8a81b5817212d351948871b`.
Its internal `MANIFEST.sha256` covers positive phases, exact control diagnostics,
source/compiler/ELF identities, actual figures, historical attempts and custody
notes. The four final Madaros PNG/SVG pairs are retained byte-for-byte. Both
engines produced identical complete native stdout and PNG bytes for all four
cases. PNGs use the existing C++ exporter; no external raster redraw occurs.
They are gzip-compressed inside the bundle rather than stored as approximately
14MB of loose images. Original ELFs, full source, PPMs and changed-input files
remain at the worker root and are not duplicated into Git.

The same four final images were visually inspected separately from these
numeric/pixel checks. The Poisson panels preserve equal spatial aspect; the
corrected normalization captions fit their panel columns. The scientific limits
in the example guide remain part of this qualification. This focused gate does
not establish whole-corpus acceptance, calibration or universal backend identity.

## Current-main integration

The retained gate06 was run on the dependency composition described above.
Remote main `8eca8e0c086bbd074485bbfd8716accc4063ef1c` was merged without
conflicts into this own branch at `c9946adb216887a85a15c1168d76cea1e5a935d4`.
The 39 upstream stdlib visibility/effect edits and upstream cleanup were preserved.
The public gate was then made portable to arbitrary Linux CI UIDs and records
the actual process identity; the Slurm submission separately asserts UID1000.
This two-file gate/guide correction is commit
`7c5b2cbcae7862950e41b09ebc06c9c0ff744d4a`.

The final frozen replay is Slurm **17534**, `COMPLETED`, `0:0`, `00:00:58`,
worker root `/scratch/sounio-stdlib-inference-scale-20261024/SIO273-main02`.
All four cases passed check/compile/native/C++ science and geometry/actual Canvas
PNG-PPM roundtrip in both engines. Ten specific changed-input controls per engine
were rejected as required. The complete staged source and both expected compiler
pins verified before/after. A separately fetched manifest also matches every
staged file against the committed workspace tree. The first current-main replay
17533 passed too; its separate receipt retains the original UID-restricted gate.

`current-main-evidence.tar.gz`, **362623 bytes**, SHA-256
`c6be4c3fe51b841d6e32687ca1bc1cb0b33e675274d52b82ea8f3e1e25088a1b`,
contains the new replay receipts, complete source catalog, compiler/ELF identities,
positive/control diagnostics and explicit output-reuse checks. Every current
native stdout, PNG and SVG was compared byte-for-byte to the corresponding
retained gate06 artifact and is identical. The first bundle therefore preserves
those exact payload/image bytes without duplicating all outputs in this second
bundle. Current native ELF identities remain distinct measurements. Original
current outputs and ELFs remain intact on the worker. Both bundle manifests were
independently extracted and verified after copying to the workspace.

`QUALIFIED_RUNTIME.sha256` identifies the current consumer/reference/gate inputs.
`GATE06_RUNTIME.sha256` preserves the original runtime hashes, including the old
UID-restricted gate; it is historical and is not a check against the current tree.
The example scientific sources, numerical verifier and production calculation
bodies were not changed for the portability correction. The documentation registry
check passed without regenerating committed governance snapshots. These focused
passes remain separate from the coordinator's full composed runner and GitHub CI.
