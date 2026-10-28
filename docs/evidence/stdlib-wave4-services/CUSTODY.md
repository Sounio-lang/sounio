# Durable custody of initial wave4 components

Slurm 18864 completed with ExitCode0:0 (5 seconds) and marker
WAVE4_COMPONENTS_DURABLE_OK.

Archive:

`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/wave4-components-export02/components.tar.gz`

SHA256:

`1aee189877ef0fcf8093b974d602cb68df48993430d416e659666fa82592c033`

The worker listed the tar, synchronized the archive, and read back its SHA.
Source/tool/executable/input/dynamic/header/artifact guards were rechecked
before export. The original oracle-only INPUTS manifest was empty; the export
records that fact rather than treating an empty file as a checksum success.
The second oracle run includes a nonempty input seal and seals every mutated
Arrow artifact.

Included qualified components:

- 18822: native F64 GPR/RAX ABI, +0/-0/0.5/qNaN payload, half(1.0), resources0.
- 18841: C++23/Arrow26 normal oracle, valid257 rows, six corruption refusals.
- 18852: the same numerical/protocol oracle controls with input and corrupted
  artifact seals added; COMPLETED0:0,10 seconds.

Historical failed export attempts remain explicit:

- 18848 FAILED1:0: the exporter tried sha256sum -c on the original empty
  INPUTS manifest; the scientific component's earlier verdict is separate.
- 18858 FAILED0:53: the job could not open its output in a root-owned staging
  directory and never entered the worker. The next staging directory was
  created as the scheduled sounio user.

The archive contains exact component source snapshots, native ELF/shared
library, compiler identity, C++ toolchain and Arrow SDK identities, all
component stdout/stderr and seals, original failed-export logs and receipts.
The raw21b3 compiler itself already has durable custody in the previous
wave3 qualified archive referenced by native-pool/CUSTODY.md.

This package does not qualify the uncommitted WorkerFn stdlib adapters,
normal pipeline, full-scale numerical concordance, or whole stdlib.
