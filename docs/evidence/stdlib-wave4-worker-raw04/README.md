# Raw04: real Worker loader qualified; complete imported diagnostics remain open

The immutable compiler source is root commit `ccc137cf642353502f39a1457b2e28cf5af147ea`, archived from a clean tree. Its source archive SHA-256 is `8a2f69bd28aebb39248c64d60f3ad6b655c7a7f5577770e5fba1da1e553452a0`. This includes the qualified ABI0 proof (`e50736a7…`), Loop pass (`673c2c0f…`), frontend preseal hook and schema2 formatter (`0e370fdd…`). Storage is absent.

Only the existing capacity overlay `eaa61d90e8d2d1897f545636c06496f8b0791fbbf206dca5bd4fa4151926ac60` and the documented main Worker opt-in were applied to the snapshot. Production main was not opened. The seed remained `9cfb6a970d73718b09091331dc514d015e3d4e6f402ee756cdc7f7b0cc55110a`. No lowerer edit, new hoist, workload reduction or compiler rebuild beyond this one bootstrap was made.

## Accepted compiler and loader

Slurm **19664 completed 0**: exact source/component/SDK identities, capacity applicability, private C++23 tool smoke and the published Loop component (97 controls) passed. No whole modern frontend OOM workload was repeated.

Slurm **19665 completed 0 in 23m16**. The retained compiler is:

```
/scratch/sounio-stdlib-wave3-20261027/root-integrated04/results/madaros
a4f5e8d655754202493bbb04d8fbb15890b10e8fc861ac3dc60438452704c333
```

The build emitted an ELF of 132711839 bytes, rejected explicit compiler diagnostics and verified source, seed, capacity and compiler manifests before publishing RAW_READY.

Slurm **19676 completed 0 in 23s**, using the original loader fixtures and sealed SDK04 (`SEALED_SOURCE.sha256` identity `13f80e8c2afc2de5027ea2c070dd708c8dca8855c22e54cdb1b6bbcd90b10aaa`):

- `direct`, `facade`, `homonym` and `alias_bridge`: check, compile and native all 0.
- Six forged/copy/spoof/wrong-ABI controls: separate check and compile both refused semantically, without ELF.
- The real imported canonical normal closure: four kernels, 60 rows, modes LOOP_REGIONS=0 and 1; check, compile, native and independent C++23 oracle all 0. Bit concordance was exact, with the unchanged relative probability and absolute log tolerances `2e-12`.
- Mutated values and reordered output were rejected by the oracle in both modes.

Root, SDK, inputs, source, tools and executable final guards passed. The complete phase table is `results/loader/RESULTS.tsv`. This qualifies the original CLI/BFS loader and these imported callbacks. It does not qualify the full services pool/pipeline stress suite or open production admission.

## Complete source diagnostics and the observed Loop result

Slurm **19667 ended status 1**, preserving all phases and full untruncated stdout/stderr. It used the same raw, original direct fixture, published WIDE projection `f2192e251936bdf331ff3a84e21cf0821de0e904a4b2044b0e55d71a99042931`, and unchanged imported fast `tests/stdlib/qpu/test_qpu_chsh.sio`. LOOP_REGIONS=1 was fixed; ELISION=0 and then 1 were explicit. The slow 62.9-billion-draw program was not executed.

All six checks and compilations passed. WIDE and QPU native executions passed in both modes; WIDE also passed its independent C++23 LCG oracle. Direct native execution is covered by the original loader above.

The published C++23 schema2 parser, explicitly invoked with `--check source`, accepted both complete WIDE captures: 19 functions, 634 instructions, 37 arguments. The additional C++23 report compares those validated captures by function identity and reports checkpoint flags. The literal and previously published hoisted cores retain their instruction counts (265/263) and one enter/exit each. With elision enabled the observed WIDE audit is `entries=2 eligible=0 neutralized=0`: this real input shows no elision benefit.

The formatter refused the imported direct/QPU captures before BEGIN:

```
direct: NV2_IR_FULL_REFUSED 2 0 4 2 0
QPU:    NV2_IR_FULL_REFUSED 2 0 4 176 0
```

Both refusals recur with ELISION=0 and 1. The source guard identifies validation of parameter register 0, rather than an instruction budget or a complete arena. The exact descriptors are not established by this receipt; an external placeholder is only a hypothesis. No validator was relaxed and no partial/legacy trace was promoted. QPU's enabled audit reports 197 functions, 16 entries, 8 eligible loops and 16 neutralized slots; this is an audit observation, not proof of speedup or a complete imported capture.

## Private tooling recovery and retained failures

The current worker image lacked `time`, `rg`, `g++` and development pthread linkage. It was not modified. The immutable GCC13 bundle came from `services/bigframe-abi07/gcc13-private/root`; GNU time came from the preserved inference tools. All SDK files were hashed. Host glibc was checked as 2.39. Runtime `libgcc_s.so.1` and `libstdc++.so.6` were copied and hashed in a private prefix. The final wrapper retains the original test flags and adds only explicit SDK/header/tool/library search paths.

The authentic installed workspace `libpthread.a` was copied with dpkg ownership/version and package md5 evidence from `libc6-dev:amd64 2.39-0ubuntu8.7`, matching host libc. Its SHA-256 is `f0a17a43c74d2fe5474fa2fd29c8f14799e777d7d75a2cc4d11c20a6e7b161c5`; package md5 is `ab0a4c8c62da160eaae565341c07f202`. This is the official eight-byte compatibility archive, not a fabricated stub. The shared runtime's unchanged `-pthread` build and full native loader qualified it operationally.

Final reusable files under the raw04 directory:

- `private-tools-v3/g++`: `e2a716dc91265963764978b627f030bbfeb945ecb9a46f94312d94f4b304dba9`.
- `tools-env-v3.sh`: `12f7db33de62f117365eb04894571462ddb1c5768e12da9de6e85706812135eb`.
- `pthread-installed-v3`: library plus `DPKG_STATUS`, `DPKG_VERSIONS`, `DPKG_MD5`, verified md5 and `ORIGIN.sha256`.

The loader gate resets PATH internally and does not honor CXX, so its snapshot copy changes only that PATH line to the private wrapper. The Loop component copy changes only the absolute time path. The compiler source and all assertions, flags, deadlines and semantic fixtures remain unchanged. All attempt versions and the exact diffs are retained in the durable archive.

Failures are preserved rather than counted as passes: 19655 (missing tools), 19658 (my environment insertion was inside the trap), 19661 (private libgcc runtime missing), 19666 (missing `-lpthread` before compiler execution), and 19675 (HTTPS recovery could not start because curl was absent). Their unstarted dependent builds/loaders were cancelled; `JOBS_FINAL.tsv` records each state.

## Durable custody

Slurm **19679 completed 0 in 47s**. Sources, exact seed, raw ELF, SDK04, private GCC SDK/runtime, all attempt recipes, loader executables, complete capture logs and guards are retained at:

```
/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/compiler/raw04-export02/raw04-sources-seed-sdk-elfs-gates-v2.tar.gz
SHA256 22e7f121aff116216da7f1acc35221acdcbde5d92c63b36aa60b9d89f0e43b60
```

The exporter required the real `/orangefs/training` Ceph mount, audited the tar listing, read the archive back, and reverified source/compiler/SDK/capture inputs. The preceding export 19669 also completed 0, before pthread recovery; its archive SHA is `2ec321b55cefd169e7956ebe0d3592f5d0174a93aec2a5a208feed90215bace2`. Archive02 is the complete final supplement.

The compact committed receipts bind the full guard files and complete compile logs by SHA. Those full files are in the archive; they were not truncated to produce this summary. Compiler source, old capacity WIP, claims of other lanes and root's canonical BIDI task remain outside this delivery.
