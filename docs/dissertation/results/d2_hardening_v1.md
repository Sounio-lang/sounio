<!-- docs:meta
topic_id: repo.docs.dissertation.results.d2-hardening-v1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.d2-hardening-v1
-->

# D2 Autograd Hardening v1

## Scope

This lane is a surgical hardening pass on the D.2 closure-free tensor tape. It
targets the two blockers surfaced by D.3:

- `D3-DENSE-001`: `tape_tensor_matmul` backward failed for the requested dense
  shapes beyond the original 2x2 bootstrap witness.
- `D3-ACT-001`: `TAPE_TANH` backward exited during `tape_backward`.

No new tape op tags were added.

## Root Causes

### D3-DENSE-001

The matmul backward path reconstructed input `TensorShape` values from tape
metadata during reverse traversal. On the larger dense witness, direct debug
inspection showed that rank metadata reads could return pointer-like values
even while the flat dimensions and lengths remained usable. This matched the
original D.2 self-audit warning about brittle stored shape metadata.

The fix keeps matmul backward in the flat-buffer style used by the D.2
bootstrap. It derives `m` and `n` from the saved output dimensions on the tape
node and derives `k` from the flat input length, then runs deterministic
row-major loops for both input gradients.

### D3-ACT-001

The activation backward functions allocated temporary tensors and reconstructed
tensor views from tape shapes. That path hit the same metadata brittleness.

The fix computes tanh, sigmoid, and ReLU gradients directly over the saved
forward buffers and gradient buffers:

- tanh: `grad_x = grad_out * (1 - y^2)`
- sigmoid: `grad_x = grad_out * y * (1 - y)`
- ReLU: `grad_x = grad_out` when saved input is positive, otherwise `0`

The ReLU subgradient at zero is documented as the tape convention `0`.

## Validation

Pinned compiler:

```text
/workspace/sounio/bin/souc-linux-x86_64
sha256=3cbea2b475e79737046f8ccf463c07d22cd5fb678fd479a032ee04bd8e19da93
```

> **Reproduction commands corrected (2026-09-13).** The commands first recorded here were
> `SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 bin/souc run <test>`. `bin/souc`
> execs that override with its arguments unchanged (it already did at the merge that added
> this file, `bebd78d74c`). Measured on 2026-09-13 with `bin/souc-linux-x86_64`, before `bin/souc`
> began refusing the form: lean_single stopped at
> `error: no main` (a current-source lean_single shows why: it opens `run`, which does not exist,
> as a 0-byte source). How the recorded values were originally produced cannot be established
> from this repository's history. `bin/souc` now refuses the form (exit 64). The commands below use the
> ELF's raw `<source.sio> <output>` interface. Run them from the repository root: lean_single
> resolves stdlib imports relative to the working directory. The pinned binary (sha256
> `3cbea2b4…`) is no longer in the repository; the re-run used `bin/souc-linux-x86_64`, sha256
> `a63ca2c960183aafcdca56e57a0c2da88b5a2005db9df5c4f2dc6a6434b8a694`.
> Re-run 2026-09-13: lean_single printed `D2_HARDENED_GENERAL_SHAPES_PASS` and
> `D2_AUTOGRAD_TAPE_CLOSURE_FREE_PASS`. Only those markers were re-checked; every other value in
> this file is the original record. The default Madaros engine (`bin/souc run <test>` with no `SOUNIO_SOUC_BIN`
> set; `bin/madaros-linux-x86_64` sha256
> `7ba4e70b6fd3a073697c629b5f17c68041afe604ebf6bb630e7c78e11e31eedb`) rejects both tests today
> with `error[E037]` in `stdlib/tensor/ops.sio`.

Focused hardening test (from the repository root):

```text
cd "$(git rev-parse --show-toplevel)" && bin/souc-lean-single-x86_64 tests/stdlib/tensor/test_tensor_autograd_d2_hardening.sio /tmp/d2_hardening.elf && chmod +x /tmp/d2_hardening.elf && /tmp/d2_hardening.elf

D2_HARDENED_GENERAL_SHAPES_PASS
```

The hardening test checks finite-difference gradients within `1e-6` absolute
tolerance for both matmul operands on:

- `(3x4) @ (4x1)`
- `(8x6) @ (6x1)`
- `(4x4) @ (4x4)`
- `(1x5) @ (5x3)`

It also checks finite-difference gradients within `1e-6` absolute tolerance for:

- `sum(tanh(x))`
- `sum(sigmoid(x))`
- `sum(relu(x))`, with the zero subgradient convention documented as `0`

Original D.2 bootstrap regression:

```text
cd "$(git rev-parse --show-toplevel)" && bin/souc-lean-single-x86_64 tests/stdlib/tensor/test_tensor_autograd_d2.sio /tmp/d2_autograd.elf && chmod +x /tmp/d2_autograd.elf && /tmp/d2_autograd.elf

D2_AUTOGRAD_TAPE_CLOSURE_FREE_PASS
```

## D3 Boundary

This branch intentionally does not merge or edit the D.3 worktree. The D.3 XOR
integration rerun is therefore left to the D.3 branch after it incorporates this
D.2 hardening commit. This preserves the user constraint that this lane only
touch D.2-owned files.

## Reproduction commands (2026-09-16)

The 2 `bin/souc-lean-single-x86_64` lines above run the lean_single seed (sha256
`9d7892132aa0a9cf839df4560bf968628dc30fda4af978a4efc4a99b4e8f89f5`, most recently changed by merge
commit `09aedffafa`) and include a `chmod +x` step. Until 2026-09-16 those lines invoked
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

- `tests/stdlib/tensor/test_tensor_autograd_d2_hardening.sio` (ELF `/tmp/d2_hardening.elf`): stdout contained `D2_HARDENED_GENERAL_SHAPES_PASS`
- `tests/stdlib/tensor/test_tensor_autograd_d2.sio` (ELF `/tmp/d2_autograd.elf`): stdout contained `D2_AUTOGRAD_TAPE_CLOSURE_FREE_PASS`

Any other line in the blocks above — expected-output markers, `rg` searches, gate scripts —
was not run and is not part of this comparison. The values recorded elsewhere in this note
were not re-derived.
