# Independent BIDI runtime-fixture protocol review

The controls in tests/fixtures/wave4_bidi_runtime/protocol_controls.mjs operate only on file-format metadata, byte identities, census and deliberate payload corruptions. They do not implement Unicode resolution, recalculate expected levels or replace the existing C++23/Sounio science.

The complete real214-file corpus from job19618 is the positive. Each negative begins as a fresh hardlink clone (ordinary copy only across filesystems). Every replacement first unlinks the destination; no write is performed through an inode shared with the original. Full corpus identities, qualified original Sounio source, subject module and golden manifest are checked afterward. Mutation directories and exact results remain in the receipts.

The22 cases cover complete acceptance, dropped files/rows, duplicate rows, duplicated payload with adjusted manifest/header metadata, extra listed/unlisted files, bad magic/count headers, simple and adjusted truncation, mutated payload, malformed/incorrect COUNTS, incorrect SMOKE header/level/body, symlinks at each control file and a symlinked corpus root.

Slurm19641 preserved the failing-before result against the actual19618 helper99f7a6a243a2f496d97cbaee7b9e3f48b98cf46316ab2743145bec4909fec4bd:14 controls passed and8 malformed datasets were falsely accepted. Protocol status and Slurm exit were1. These expectations were not relaxed.

Slurm19643 completed0 with22/22 against intermediate helperf4f459. Slurm19648 repeated the unchanged22 controls against the final helper8b35a59e4b63c2b0df0797ec5c57aea7f01c24e6067a23b1334a31ec2e1b4627 and completed0/16s. All original corpus/source/module/golden identity flags remained true and source manifests verified. Only the latter helper is the final protocol qualification.

## Findings and disposition

- The original diagnostics walker would reject every runtime-fixture subtree as unexpected capture files. The integration lane replaced the implicit tree allowance with an explicit artifact manifest bound to native/start and native/result.
- The original validator did not bind COUNTS/SMOKE or chunk payloads to the official reference. Eight before-controls demonstrated the hole. The final validator binds all214 files to the independently crosschecked immutable golden.
- The original helper retained an ELF copy but executed the temporal path. The integration candidate executes the retained program for all211 chunks plus the two negative controls, and preserves durable identities after temporal-file deletion.
- Independent review found a fixture/output bytes collision: native result bytes overwrote the fixture length, while verification required that same field to equal both file sizes. The integration lane separated fixture bytes, stored output_bytes and observed_output_bytes. Its operational C++23 partial controls use fixture sizes199928/215668 and log sizes46/36, plus an output-limit case, without claiming to resolve BIDI in that test program.
- The runner had three contract edges: dangling argv sidecars bypassed an exists-only guard; the legacy no-output check-only fallback could bypass a declared runtime fixture; and no-capture cleanup removed the retained artifact directory. The integration lane reported focused controls for their fixes. No production file was edited by this review lane.
- Cancellation was reviewed against the existing bounded-command process-group contract: native and producer children inherit the externally bounded group. This22-case protocol does not claim a new live cancellation or full canonical BIDI execution.

The actual scientific direct qualification remains job19618, which executed the original Sounio resolver over all211 chunks/861948 cases. The newly integrated canonical hook still requires its own complete scheduled execution and source/ELF guards. Protocol controls and operational partial fixtures are not substitutes for that gate.
