# LOOM V12 BPF LSM Peer Matrix

## Question

Can the frozen three-hook BPF LSM mediator causally refuse the complete
ten-operation Sounio V12 basis for two hostile principals with identical real,
effective, saved, and filesystem kernel UIDs?

This stage executes the decisive causal pair only. The diagnostic controls are
deliberately left for the next stage, so a successful result here still retains
`material_peer_matrix=false`, `same_uid_peer_isolation=false`, and action 9025
`DENY451`.

## Pair construction

Each operation receives a fresh principal epoch containing one target and one
attacker. They have:

- identical four-slot kernel UID vectors;
- distinct PIDs, pidfds, start ticks, and cgroups;
- an attacker with `Seccomp: 0` and all named syscalls callable;
- a dumpable target in a cell where Yama is absent, so no Yama-specific
  `PR_SET_PTRACER` aperture exists or is required;
- symmetric target and attacker capability sets containing only
  `CAP_SYS_NICE`, the kernel prerequisite for remote `process_madvise`;
- one fixed target address, target descriptor, and syscall argument set.

The decisive-pair microhost fixes the active LSM order to
`lockdown,capability,bpf,ima,evm`. Yama and AppArmor are excluded from this
cell because either could independently refuse a ptrace-family route after the
BPF mediator is removed, destroying the claimed single-delta experiment. Their
presence belongs in separately named interference controls, not in the causal
pair used to attribute refusal to the BPF mediator.

The packed root directory is fixed at `0755` and verified by PID 1 before any
pair starts. More sensitive archive members retain their restrictive modes.
This prevents root-directory DAC from being misreported as an LSM refusal for
pathname-based operations such as `/proc/<pid>/mem`.

The same live processes execute both halves:

1. `TREATMENT`: the three pinned BPF LSM links are active and the operation must
   return a named kernel refusal (`EACCES` or a wrapper-normalized `EPERM`)
   before the target witness changes;
2. `MEDIATOR_REMOVED`: PID 1 unlinks the three pins, proves all link and program
   IDs extinct through a fail-closed 5 s observation barrier, crosses a bounded
   250 ms quiescence window, and asks the same attacker to repeat the same
   operation against the same target.

The only delta is mediator presence plus policy hash. A second VM, a new PID, a new
target address, or a different syscall argument is not an admissible sabotage.

## Completion witnesses

The mediator-removed half must produce the Sounio-frozen witness:

1. target terminated by `kill(SIGTERM)`;
2. target thread terminated by `tgkill(SIGTERM)`;
3. queued signal payload observed by the target;
4. target terminated by `pidfd_send_signal`;
5. ptrace attach, stop observation, and detach;
6. exact target canary read by `process_vm_readv`;
7. exact target canary read through `/proc/<pid>/mem`;
8. target descriptor duplicated and closed by `pidfd_getfd`;
9. target resource limit changed, observed, and restored;
10. `process_madvise(MADV_PAGEOUT)` completes for exactly 4096 target bytes;
    the target subsequently observes the same canary bytes, whether the kernel
    reclaimed and faulted the anonymous page back or merely deactivated it.

Syscall success without the named witness is a refusal of the experiment, not a
completion.

## Boundary

The experiment is a bounded Linux/KVM material parity result. The trusted PID 1
guardian and the frozen BPF object are outside the hostile peer boundary. Python
and Rust are absent. C, BPF, and C++20 are transitory material producers; they
cannot revise the Sounio result or promote the receipt to semantic authority.
