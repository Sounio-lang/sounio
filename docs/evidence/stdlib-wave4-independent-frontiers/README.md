# Three frontier diagnostics and retained Graph/DP54 execution

All three original sources checked and compiled successfully in both loop-region modes with accepted raw04 and the current root05 SDK. These results qualify the tested combination, not the failed root05 compiler. No compiler or stdlib source was edited, no original test was changed, and no new bootstrap was run.

## Fixed identities

- Accepted compiler: `/scratch/sounio-stdlib-wave3-20261027/root-integrated04/results/madaros`, SHA256 `a4f5e8d655754202493bbb04d8fbb15890b10e8fc861ac3dc60438452704c333`.
- Current SDK: `/scratch/sounio-stdlib-wave3-20261027/root-integrated05/source/stdlib`, from root HEAD `237ff2d674eb8e8f350d948c4f6f6393706cd67d`, Git tree `b413d6cd7409aade6b6b66f14761f40b750d008a`, with declared snapshot overlays. Actual overlaid contents are bound by `root-integrated05/results/SOURCE.sha256`; Git census independently binds every path, regular-file mode and symlink target.
- The failed root05 executable was never executed. No `RAW_READY` was created or filled.
- Private qualified tools: root04 `tools-env-v3.sh` and `private-tools-v3/g++`, retaining original C++23 flags and GNU time. No system packages were modified.

| Byte-exact original source | Source SHA256 | LOOP0 check/compile | LOOP1 check/compile |
|---|---|---|---|
| `tests/stdlib/os/test_env_present.sio` | `d73d64baeb98ba8fc15b4cc09abb387ed81539732d7a264018b970adff8464d4` | 0/0 | 0/0 |
| `tests/stdlib/graph/test_graph_types_e2e.sio` | `c4108863375941686aab1b53b6603c5c378418744ffa0199108f6f7ba87ebe02` | 0/0 | 0/0 |
| `tests/stdlib/ode/dp54_invalid_probe.sio` | `c25f96cb8ca8bbc8ed8982f7bd5dee5c6a293bd640bd65e9db7a88decb152893` | 0/0 | 0/0 |

Slurm19814 array tasks0/1/2 completed0 in10/10/17seconds. Every phase had independent process status, argv, stdout, stderr, GNU time and a1200-second timeout with kill-after30. The collector can record compiler failures independently; here all12 actual phases returned0. The LOOP1 compiler log includes informational `loop_checkpoint_elision ... refused=N ... status=0` census lines. Those lines are not compilation failures or Worker capability refusals.

The original `os/test_env_present.sio` remains a check-only test and was never executed natively.

## Existing original ELFs

Slurm19822 completed0/5seconds. It ran byte-identical executable copies of the already retained Graph and DP54 ELFs, one of each per loop mode. It did not recompile Sounio. All four processes returned0 within the original runtime bound and printed:

- Graph: `GRAPH_TYPES_OK`.
- DP54: exactly12 `DP54_INVALID` rows, indices0..11, each with fields `3 0 0 3 0 0 0 0 0`, then `DP54_INVALID_PASS`.

Hash guards bind original retained ELFs, executable copies and SDK before/after. Graph's original self-guards plus marker prove its own assertions passed; they alone do not supply independently observed metrics.

## Independent observations

Slurm19829 completed0/11seconds, using the separate `graph_metrics.sio` companion and the same accepted raw04/current SDK. Both loop modes checked, compiled and ran0. The C++23 adjacency-set reference accepted exactly25 lines per mode:

- Four nodes, four undirected edges, degrees `[3,2,2,1]`.
- All16 adjacency pairs agree with independently constructed `std::set` adjacency, including every nonedge and diagonal.
- Node3 has exactly one neighbor, node0.

The companion is an additional observability fixture. No original source, assertion or ELF was replaced.

For DP54, the recipe extracted and hashed the actual existing `invalid_control` C++ function from the current SDK reference file. Both original native stdout files were accepted: all12 invalid rows, error/status/evaluation-counter values and the final marker. Domain/state values that the original Sounio fixture only asserts internally are covered by original assertions plus native0; they are not independently printed numeric observations.

Both actual validators rejected wrong-value, dropped-row and extra-row outputs in both loop modes:12 negative controls, all exit1. Strict C++23 compilation used `-O2 -Wall -Wextra -Werror -pedantic`.

## Preserved failed harness attempts

- Slurm19825 failed before running the companion because a C++ range-loop structured binding copied an edge under `-Werror`. The final fixture uses `const auto&[a,b]`.
- Slurm19828 accepted the first DP54 oracle but the companion source did not parse because `Seq<NodeId>=` was lexed as an operator. The final source separates the annotation and assignment: `Seq<NodeId> =`.

Both attempts, inputs and logs are preserved. Neither failure came from an original test, and neither was repaired by changing the original tests or production source.

## Custody

Scratch run: `/scratch/sounio-stdlib-wave3-20261027/root-independent-frontiers01` on DL380. It includes original independent-phase logs, retained ELFs, native logs, all observability attempts, exact recipes, the generated literal DP54 reference, validator mutants and source/input/mode/link guards.

The complete root05 source/seed/failed-build custody already exists at:

`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/compiler/raw05-failed-export01/raw05-failed-full.tar.gz`

SHA256 `4a3008899c98a2e0aa491c2a33af6a54077d6ea7a15f5b50d183d85040b9bcf8`, export19798 plus independent readback19799. The new frontier export is recorded separately in `CUSTODY.md`; it adds accepted raw04, current SDK stdlib, exact original sources/reference and all diagnostic/retained/companion evidence.

The scope is these exact tests and configurations. No general Graph/ODE correctness claim or new modern-compiler qualification follows from these observations.
