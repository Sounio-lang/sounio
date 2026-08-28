# Host PrincipalCell descriptor barrier v1

Status: material prerequisite measured; product release remains closed.

## Authority and role

This artifact implements the preregistered experiment in
`GARDEN_PRINCIPAL_CELL_BARRIER_V1.md`. Sounio remains semantic authority. The
C++20 program is a transitory `MATERIAL_PARITY` primitive and contains no Sounio
decision strings or expected semantic result.

The primitive has no command, path, token, shell, script, or payload surface.
Its only public mode is a bounded selftest. A child inherits a read descriptor
and result descriptor; its parent retains the only write descriptor. The child
is armed with `PR_SET_PDEATHSIG=SIGKILL` and refuses parent-identity drift.

## Frozen inputs

- Garden commit: `7f67c10c911388e818c8404ae1452ca7a5f522b5`
- Garden SHA-256:
  `2eb1f670bbc5e0254a7cda7ff14c53ed5eb0ce93ae36a3d7f00456a3d2d77dbf`
- C++20 source SHA-256:
  `9885c7a22d14baf0972b9edde00718cc19b590ec3f3bea4f1b859310a62a636c`
- Build script SHA-256:
  `ccfda8ccef60d173e8200a206ebbab8308a716b993ed7c34779783c7c34689eb`
- Gate script SHA-256:
  `51cbd061eeb9230ef4f85af6cf307de190fd5a6e514c733e21c71494498ee218`
- Deterministic binary SHA-256:
  `1a36b2e441ec1fb857c084d3f44ec4f5f599ecba2cfd34187ef65c5571bef29e`

## Material observation

The treatment and sabotage used the same process implementation, generation,
deadline, descriptor topology, and terminal-state parser. They differed only in
the parent write:

- treatment: close without write -> `BARRIER_CLOSED reason=eof`
- isolated sabotage: one exact generation-bound write ->
  `BARRIER_OPENED reason=exact-release`

Exactly one open sentinel was observed across the pair. Wrong generation,
truncation, excess bytes, duplicate release, timeout, and descriptor absence all
closed. Two independently built binaries were byte-identical. The runtime
dependency scan found no Python or Rust dependency.

## Hardware receipt

- Kernel: `Linux 7.0.2-5-pve`
- Architecture: `x86_64`
- Logical CPUs: `64`
- CPU: `INTEL(R) XEON(R) GOLD 6526Y`
- Command: `bash scripts/ci/sounio_loom_principal_cell_barrier_selftest.sh`
- Result: `PASS`

## Evidence boundary

The experiment establishes:

- `descriptor_barrier_causal=true`
- `material_threshold_measured=true`
- `user_command_surface=false`

It does not establish same-UID hostile-peer isolation. The generation exists in
the forked address space, so a hostile peer with memory-injection authority is
outside this local experiment. The distinct-principal host experiment remains
the required material identity boundary.

The following remain false:

- `material_grant=false`
- `material_execution=false`
- `launch_open=false`
- `exec_attached=false`
- `parity_open=false`
- `claim_ready=false`

