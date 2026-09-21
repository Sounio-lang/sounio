# LOOM V12 BPF LSM Peer Mediator Load

## Purpose

This stage is material parity for the frozen Sounio V12 authority plan. It asks
one deliberately narrow question: can an isolated, no-disk, no-network KVM
microhost load exactly three BPF LSM programs, pin their links, survive the
loader process exiting, and then prove that every link becomes extinct after
its pin is removed?

The three frozen hook surfaces are:

- `lsm/task_kill`
- `lsm/ptrace_access_check`
- `lsm/task_prlimit`

The BPF program, loader, init, archive packer, and host gate are transitory C++20
or C material bootstrap. They may implement and measure the already frozen plan;
they do not define semantics or expected results. Sounio remains
`SEMANTIC_AUTHORITY`.

## Causal observation

Existence of a path in `bpffs` is insufficient evidence. PID 1 therefore:

1. waits for the loader to exit with all loader-owned link descriptors closed;
2. opens each pinned object with `BPF_OBJ_GET`;
3. reads its kernel link identity with `BPF_OBJ_GET_INFO_BY_FD`;
4. removes all three pins;
5. proves that `BPF_LINK_GET_FD_BY_ID` returns `ENOENT` for every recorded link.

The positive half proves pin survival beyond loader lifetime. The negative half
proves link extinction, rather than mere path disappearance.

## Boundary

This is not the V12 peer-operation matrix. It does not prove that all ten
operations are refused before effect, nor that the intended rule caused each
refusal. Consequently every receipt from this stage must retain:

```text
material_peer_matrix=false
same_uid_peer_isolation=false
action_9025_decision=DENY451
claim_ready=false
```

Any missing hook, verifier refusal, attach failure, pin failure, loader timeout,
unexpected link survival, policy drift, toolchain absence, transport error, or
receipt mismatch fails closed.

## Isolation

The guest boots the host kernel with BPF LSM explicitly active and has:

- KVM CPU isolation;
- no guest disk;
- no guest network device;
- an ephemeral initramfs;
- a private `securityfs` and `bpffs`;
- deterministic shutdown after the measurement.

No Python or Rust process participates in the build, measurement, or decision.
