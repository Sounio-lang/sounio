# WorkerFn source closure profile (native issuance remains closed)

This component adds a transitive source profile to the opaque signature admission
in `18ab3f749fe33123ca4524563985e7354a77f409`. It records resolved `FnSigID`
call edges at the actual checker call-resolution site, then audits requests only
after all function bodies have been checked. A bare spelling is not a call-graph
identity. This commit does not issue a native certificate or enable the public
WorkerFn constructors in lowering.

The profile accepts scalar helpers, recursion and references to scalar stack
locals forwarded through audited helper parameters. References cannot escape,
become numeric words, derive from globals, or flow into aggregate/indirect/extern
calls. Global values, allocation/aggregate expressions, captured/local function
aliases and unresolved calls are refused. The AST and call-graph depth bounds
are independent and fail closed. Metadata is reset for each checker graph;
auxiliary ontology checking does not erase the active checked graph.

Protected raw extraction is admitted by the checker only for the physical worker
module identity supplied by the loader's trusted-origins registry. The native
lowering refusal from the preceding component remains in force. The physical
origin helper in the qualified source is the dual-bootstrap syscall-based
implementation from root `2434a4bd70` (adopted locally as `3d81508c272f`).

## Paired native component evidence

Slurm **19030**, `COMPLETED 0:0`, elapsed **00:01:18**, executed on the DL380.
The executable contains the actual parser, specializer and checker. Its input
loader accepts one file; it does not traverse a real imported-module graph.
All executable builds and test runs occurred inside Slurm with a 512 MiB stack.

Durable bundle:
`/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/worker-source-v2`.

- Overlay SHA256: `78f785da7cf52e94bbcf50dd86439734afd23e5f9b1d336a757dee40255a4e2e`.
- Actual checker ELF SHA256: `a5298c1196b30f9a27ef3eb0562567ca977fe6c8d2d46c192a4f22df370f3835`.
- Opaque type component ELF SHA256: `536687183b625cefaf21e6833c7c87561307808b3b67a594539047cf3aa26c50`.
- `WORKER_VERDICTS.tsv`: **18 cases, 5 positive, 13 negative**; C++23 verifier
  passed and its **7 deliberately wrong output controls** were rejected.
- The eight new fixtures all returned checker 0 under the preceding signature-only
  checker from job18903. Under this source profile, the three valid transitive,
  recursive and stack-ref fixtures remain 0; the five transitive-global,
  indirect, 24-byte aggregate, global-reference and reference-to-integer cases
  return 1 with the WorkerFn closure diagnostic. Parser/E137/E264 failures do not
  count as these semantic refusals.
- Existing returned-loan regression: **46 cases, 16 positive, 30 negative**,
  C++23 verdict and seven wrong-output controls passed.
- Source, seed, feature input, control and reference hashes match before/after.
  Explicit bootstrap errors and dropped-store diagnostics were checked and absent.

The earlier job18989 preserved explicit bootstrap errors in the superseded origin
helper; it is not substituted for this clean qualification. The signatures and
source-profile results alone do not establish imported BFS identity preservation,
final IR purity, isolated worker clones, emitter behavior or native thread safety.
Those gates remain open. Native issuance is still refused rather than inferred
from the source profile.
