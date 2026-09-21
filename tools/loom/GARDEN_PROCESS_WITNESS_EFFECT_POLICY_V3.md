# GARDEN: LOOM Immutable Root Effect Policy V3

Status: `PREREGISTERED_AFTER_FALSIFICATION`

## Falsifying Observation

The V2 Sounio policy required Landlock and prohibited fallback. Its exact
native parity binary was transported by hash into the host namespaces and
returned:

```text
host=t560-proxmox
kernel=7.0.2-5-pve
landlock_abi=-1
landlock_errno=95
landlock_errno_name=EOPNOTSUPP
```

Evidence is frozen at
`tools/loom/evidence/loom-process-witness-effect-policy-host-attempt-v1-20260828.txt`
with SHA-256
`e702ceb3e2149d2d83cd054b147f9130e97fdb2d082b4ee839e24d4fcfdd24bb`.
The host still realized all twelve seccomp refusals, but zero Landlock
treatments and zero material sabotages. V2 therefore remains
`material_coverage=false` and cannot authorize the effect cell.

V3 is not a compatibility fallback hidden under the V2 name. It is a new,
Sounio-first hypothesis produced because the preregistered V2 mechanism was
materially unavailable.

## Hypothesis: Immutable Root Principal

For one statically linked principal cell and one statically linked Sounio
payload, a systemd-created mount namespace can replace Landlock's object
boundary without weakening the closed effect claim when:

1. the cell starts inside a content-addressed, root-owned, immutable root tree;
2. that tree contains only the frozen cell, frozen Sounio payload, frozen
   manifests, and empty mount points required by the kernel boundary;
3. the entire root is read-only before the cell's first instruction;
4. `/home`, host `/tmp`, host devices, host `/proc`, `/sys`, cgroups, and the
   host mount namespace are absent or replaced by named minimal views;
5. no mutable bind, overlay upper layer, package runtime, dynamic linker,
   interpreter, or host pathname is visible to the principal;
6. fd 3 is opened from the immutable root, hashed, marked close-on-exec, and
   used by `execveat(..., AT_EMPTY_PATH)`;
7. after posture validation, the exact Sounio V2 positive seccomp filter admits
   only `read(0)`, `write(1|2)`, `exit`, and constrained `execveat(3)`;
8. the existing DynamicUser, pidfd, cgroup, namespace, descriptor quorum,
   same-PID transition, and affirmative extinction witnesses remain intact.

The object claim is now: before seccomp, trusted frozen cell code can name only
objects in one immutable root; after seccomp, it cannot issue any pathname or
mount operation. The root tree and mount namespace are part of the receipt, not
ambient configuration.

## Exact Root Shape

The V3 Sounio plan must name this closed root schema:

```text
/
|-- loom/
|   |-- effect-cell
|   |-- payload
|   |-- payload.freeze.v1
|   `-- effect-policy-v3.freeze.v1
|-- dev/
|   `-- null
|-- proc/        # absent in treatment; minimal self-only view only in named probe
`-- tmp/         # empty read-only directory in treatment
```

All regular files are root-owned, single-link, not writable, and hash-bound.
All directories are root-owned and not writable by DynamicUser. The root tree
itself is mounted read-only. The principal has an empty environment except for
frozen controller identity fields consumed before seccomp. Stdin/stdout/stderr
are the only inherited stream descriptors; fd 3 is the exact payload; every
other descriptor is closed before policy installation.

The first implementation may use an immutable directory release as
`RootDirectory=` if the host gate proves its mount is read-only and no source
path is reachable. A later immutable squashfs or dm-verity root may strengthen
the storage witness, but cannot be claimed by this experiment unless executed.

## Sounio V3 Policy

Before any V3 native executor bytes, Sounio must freeze a self-contained plan
that preserves:

- action `9025`'s exact twelve effect families and fourteen decision frames;
- V2's exact four-syscall positive surface and argument constraints;
- architecture mismatch `KILL_PROCESS` and default action `ERRNO_EP1`;
- all treatment and sabotage expected results.

It replaces only these material fields:

```text
object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE
landlock_required=false
landlock_unavailable_receipt=<host-attempt-sha256>
root_read_only=true
dynamic_linker_visible=false
host_root_visible=false
pathname_syscalls_after_filter=0
```

The V3 probe for family `10` is `personality_change`, not `bpf`. Unprivileged
BPF is independently kernel-denied on hardened systems and cannot supply the
required causal bypass. `personality_change` is unprivileged when the named
systemd `LockPersonality`/seccomp family rule is removed, so the twin can prove
that exact rule caused the treatment refusal. This is a preregistered correction
before V3 native bytes.

## Twelve Material Twin Rules

Each family owns one logical kernel rule group. A group may compile to both a
systemd namespace property and BPF instructions, but no unrelated family field
may change between twins.

| ID | Treatment rule | Single-family sabotage |
| --- | --- | --- |
| `1` | fd 3 close-on-exec plus constrained `execveat` | retain fd 3 across exec; READY receipt observes the executable route still present |
| `2` | deny `clone3` and require zero descendants | admit `clone3`; host observes a descendant before cgroup teardown |
| `3` | immutable root and deny `openat(O_CREAT)` | expose only `/tmp` as writable and admit the exact create probe |
| `4` | exact fd inventory and deny `dup3` | admit `dup3`; fd inventory gains exactly the probe descriptor |
| `5` | deny writable shared `mmap` | admit one anonymous shared writable mapping; map inventory records it |
| `6` | deny `io_uring_setup` | admit it; if the host independently refuses it, the experiment stops |
| `7` | private network plus deny `socket(AF_INET)` | remove only the address-family rule and admit the exact socket probe |
| `8` | deny `socket(AF_UNIX)`/SCM route | admit one close-on-exec Unix socket descriptor |
| `9` | deny `memfd_create` and unregistered IPC | admit one close-on-exec memfd |
| `10` | `LockPersonality` plus seccomp denial | remove only that rule group and change the process personality |
| `11` | no proc mount plus deny `openat(/proc/self/mem)` | expose only the minimal proc view and admit that exact read-only open |
| `12` | architecture check and positive default-deny | admit one otherwise unlisted `getpid` syscall |

Every sabotage uses the same cell, payload, argv, environment, generation
shape, root-tree bytes, observer, timeout, and controller. Writable `/tmp` or a
minimal proc view may be mounted only for its named sabotage; the receipt must
prove no other namespace/property changed.

## Acceptance

V3 may set `complete_effects=true` only after:

1. the self-contained Sounio V3 policy executable and manifest are frozen;
2. a statically linked native cell and Sounio payload are verified as having no
   interpreter or dynamic dependencies;
3. the actual DynamicUser process mountinfo/root/fd/seccomp facts match V3;
4. all twelve treatment effects are refused;
5. all twelve single-family twins cross exactly their named rule group;
6. family `6` really crosses on the named host rather than being independently
   refused;
7. the complete receipt is judged `ALLOW` by frozen action `9025`;
8. the same-pid ProcessWitness and full extinction gates still pass;
9. removal of any mount, BPF, identity, receipt, or extinction fact fails closed;
10. production activation and every attachment remain false.

Until all ten hold:

```text
material_coverage=false
complete_effects=false
material_execution=false
launch_open=false
recycle_open=false
exec_attached=false
commit_attached=false
ci_attached=false
parity_open=false
claim_ready=false
```

## Nonclaims

- V3 does not claim Landlock support.
- A systemd property string is not proof of the actual mount namespace.
- Read-only host storage without root-namespace isolation is insufficient.
- A seccomp-only result is insufficient for filesystem object closure.
- A dynamically linked cell is insufficient for the exact minimal root.
- Structural sabotage digests are not material sabotage receipts.
- The V2 failure is not erased or reclassified as success.
- This Garden does not open arbitrary command execution or a product path.
