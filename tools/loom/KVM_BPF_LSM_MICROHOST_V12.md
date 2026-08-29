# LOOM V12 KVM BPF-LSM Microhost

Status: `MATERIAL_BOOTSTRAP_AFTER_BACKEND_INCOMPLETE`

## Purpose

The frozen V12 backend discovery proved that the current `t560-proxmox`
AppArmor stack mediates same-kuid signals but not `prlimit64`. Although its
kernel was compiled with `CONFIG_BPF_LSM=y`, `bpf` is absent from the active LSM
order. Rebooting or reconfiguring that host is outside this experiment.

This microhost is an ephemeral kernel domain for the next receiver-mediator
candidate. It boots the node's hashed kernel under KVM with a repository-built
initramfs and an explicit `lsm=` order containing `bpf`.

## Material Boundary

```text
host hypervisor     = t560-proxmox / KVM
guest disk          = none
guest network       = none
guest initramfs     = repository-built, content hashed
guest init          = static C++20, MATERIAL_BOOTSTRAP
guest kernel        = exact host /boot/vmlinuz hash named by receipt
guest command line  = explicit lsm order containing bpf
semantic authority  = frozen Sounio V12 plan
```

The microhost proves only that a separate kernel domain can activate BPF LSM,
mount `securityfs` and `bpffs`, expose BTF, and extinguish. It does not prove
that a BPF program loads, that any peer operation is refused, or that
`same_uid_peer_isolation=true`.

## Acceptance

1. the guest boot ID differs from the host boot ID;
2. guest PID 1 is the hashed static C++20 init;
3. `/sys/kernel/security/lsm` contains `bpf`;
4. BTF, `securityfs`, and `bpffs` are present;
5. QEMU has neither a guest disk nor a network device;
6. the guest powers off and the QEMU process becomes extinct;
7. the frozen V12 semantic and negative backend manifests remain unchanged.

## Nonclaims

- This is not a production launch surface.
- KVM is material transport, not semantic authority.
- Activating BPF LSM is not evidence that all ten V12 operations are covered.
- No result may set action 9025's peer-isolation field.
- Python and Rust remain forbidden.
