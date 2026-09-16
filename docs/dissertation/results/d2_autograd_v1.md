<!-- docs:meta
topic_id: repo.docs.dissertation.results.d2-autograd-v1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.d2-autograd-v1
-->

# D2 Autograd v1

## Scope

This lane adds a closure-free reverse-mode tape for the D.1 dynamic tensor stack.
The branch is based on the local `codex/sounio-tensor-stack` commit because D.1
was not present on `origin/main` when this lane was launched.

The tape uses explicit op tags and a reverse-order `if` dispatcher. It does not
store closures, callbacks, or function pointers in tape nodes.

## Implementation Notes

- `stdlib/tensor/tape.sio` defines tag constants for the D.1 arithmetic,
  matmul, reduction, shape, and activation operations.
- `TensorTape` uses fixed-capacity primitive arrays rather than `[TapeNode; N]`
  or `[Tensor; N]` fields. The array-of-struct form compiled but was unstable
  in this compiler/runtime lane.
- Gradients are allocated per tape value and accumulated in flat storage. The
  worked matmul chain uses flat backward loops to avoid relying on brittle
  intermediate shape reconstruction.
- Shape and activation backward functions are implemented, but the focused gate
  only proves forward recording for the deep shape/activation chain. The
  finite-difference proof covers the critical `matmul -> add -> sub -> mul ->
  sum` training loss.

## Validation

Pinned compiler:

```text
SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64
sha256=3cbea2b475e79737046f8ccf463c07d22cd5fb678fd479a032ee04bd8e19da93
```

> **Reproduction command corrected (2026-09-13).** The command first recorded here was
> `SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 bin/souc run <test>`. `bin/souc`
> execs that override with its arguments unchanged (it already did at the merge that added
> this file, `bebd78d74c`). Measured on 2026-09-13 with `bin/souc-linux-x86_64`, before `bin/souc`
> began refusing the form: lean_single stopped at
> `error: no main` (a current-source lean_single shows why: it opens `run`, which does not exist,
> as a 0-byte source). How the recorded values were originally produced cannot be established
> from this repository's history. `bin/souc` now refuses the form (exit 64). The command below uses the
> ELF's raw `<source.sio> <output>` interface. Run it from the repository root: lean_single
> resolves stdlib imports relative to the working directory. The pinned binary (sha256
> `3cbea2b4…`) is no longer in the repository; the re-run used `bin/souc-linux-x86_64`, sha256
> `a63ca2c960183aafcdca56e57a0c2da88b5a2005db9df5c4f2dc6a6434b8a694`.
> Re-run 2026-09-13: lean_single printed `D2_AUTOGRAD_TAPE_CLOSURE_FREE_PASS`. Only that marker
> was re-checked; every other value in this file is the original record. The default Madaros
> engine (`bin/souc run <test>` with no `SOUNIO_SOUC_BIN` set; `bin/madaros-linux-x86_64` sha256
> `7ba4e70b6fd3a073697c629b5f17c68041afe604ebf6bb630e7c78e11e31eedb`) rejects this test today
> with `error[E037]` in `stdlib/tensor/ops.sio`.

Focused run (from the repository root):

```text
cd "$(git rev-parse --show-toplevel)" && bin/souc-lean-single-x86_64 tests/stdlib/tensor/test_tensor_autograd_d2.sio /tmp/d2_autograd.elf && chmod +x /tmp/d2_autograd.elf && /tmp/d2_autograd.elf
D2_AUTOGRAD_TAPE_CLOSURE_FREE_PASS
```

Worked example:

```text
loss = sum((W @ x + b - y)^2)
```

The tape gradients for `W` and `b` match central finite differences within
`1e-6` absolute tolerance on the focused 2x2/2x1 witness. The dispatch also
checks repeated-parent accumulation with `x2 = (x + x)^2` and scalar division.

## Static Audit

The no-closure audit over `stdlib/tensor/tape.sio` found no closure syntax,
callback fields, or stored function pointers. The only grep matches were
`Option` match arms (`None =>`, `Some(...) =>`) and the test marker print.

## Caveat

This should be treated as a D.2 bootstrap proof, not as a fully mature tensor
autograd library. The op tags and backward functions are present, but the
compiler/runtime still shows fragile behavior around stored shape metadata for
deep intermediate tensors. D.3/D.5 tensor work should keep the same flat,
explicit-gradient style until a broader shape-metadata regression suite lands.

## Reproduction commands (2026-09-16)

The `bin/souc-lean-single-x86_64` line above runs the lean_single seed (sha256
`9d7892132aa0a9cf839df4560bf968628dc30fda4af978a4efc4a99b4e8f89f5`, most recently changed by merge
commit `09aedffafa`) and includes a `chmod +x` step. Until 2026-09-16 that line invoked
`bin/souc-linux-x86_64` (sha256
`a63ca2c960183aafcdca56e57a0c2da88b5a2005db9df5c4f2dc6a6434b8a694`) and contained no `chmod +x` step.

Run as written today, a line without `chmod +x` exits 126, with `Permission denied` reported
by the launcher: both binaries write the ELF with mode `-rw-r--r--` under umask 0022. How the
earlier runs recorded in this note were invoked is not recorded here, and earlier paragraphs
naming `bin/souc-linux-x86_64` describe those runs.

Measured 2026-09-16 at HEAD `169ac84463`, from the repository root, at the documented `/tmp`
ELF paths (none existed beforehand; each was removed afterwards). Each source below was
compiled and run with the seed and with `bin/souc-linux-x86_64`; every build exited 0, every
run exited 0 after `chmod +x`, and for each source the two binaries' stdout was byte-identical
(stdout only, not the ELF bytes):

- `tests/stdlib/tensor/test_tensor_autograd_d2.sio` (ELF `/tmp/d2_autograd.elf`): stdout contained `D2_AUTOGRAD_TAPE_CLOSURE_FREE_PASS`

Any other line in the blocks above — expected-output markers, `rg` searches, gate scripts —
was not run and is not part of this comparison. The values recorded elsewhere in this note
were not re-derived.
