# GARDEN: LOOM Identity-Typed Inert Mounts V10

Status: `PREREGISTERED_AFTER_V9_MOUNT_GRAPH_FALSIFICATION`

## Falsifying Observation

V9 crossed namespace construction, typed file bounds, principal rendezvous
opacity, root and executable object identity, incoming-source object identity,
read-only `sysfs`, seccomp installation, and READY. The root host observer then
refused exactly because `ProtectSystem=strict` materialized one additional mount
at `/proc`.

The frozen refusal is
`tools/loom/evidence/loom-process-witness-effect-root-v9-host-attempt-v1-20260828.txt`
with SHA-256
`260a993e35974bb4d1899fb376b3682fbb6813b063c271c8f7c551d6ebfc6725`.

The observed mount was not `procfs`. It was a read-only bind of the empty,
root-owned `$CAPSULE_ROOT/proc` directory from the capsule backing filesystem.
An aggregate diagnostic observed no additional mount at `/home`, `/root`,
`/run`, `/var`, or `/etc`.

## Semantic Correction

V10 replaces path-only mount absence with identity-typed mount authority.
The semantic question is not merely whether `/proc` is a mountpoint. It is
whether that mount confers process-observation authority.

V10 defines one admissible structural mount type:

```text
CAPSULE_EMPTY_BIND(
  path=/proc,
  source_object=$CAPSULE_ROOT/proc,
  filesystem=$CAPSULE_ROOT_FILESYSTEM,
  root_owned=true,
  contents=empty,
  vfs_read_only=true,
  principal_writable=false,
  procfs_visible=false
)
```

`/proc` is semantically absent as a process-observation capability only when
all fields of that type are proven. A mountpoint name, a read-only flag, or an
empty directory alone is insufficient.

No mount remains admissible at `/home`, `/root`, `/run`, `/var`, or `/etc`.

## Root-Host Authority

At READY, the root host observer must prove:

1. the V9 root, executable, temporary, incoming, `sysfs`, namespace, and
   extinction facts;
2. exactly one additional structural mount exists in the restricted set and
   its mountpoint is `/proc`;
3. `stat(device,inode)` for the mount target equals
   `stat(device,inode)` for `$CAPSULE_ROOT/proc`;
4. the mount filesystem identity equals the capsule backing filesystem;
5. the VFS mount is read-only even if the backing superblock is writable;
6. the target is root-owned, empty, and principal-non-writable;
7. the filesystem type is not `proc`, `procfs`, `sysfs`, or `tmpfs`;
8. `/home`, `/root`, `/run`, `/var`, and `/etc` have no mount entries.

The root host reads `mountinfo`, `stat`, and the directory contents. Requested
systemd properties remain configuration evidence only.

## Causal Twins

The material experiment has one treatment and four preregistered one-variable
sabotage twins:

| Case | Changed field | Expected decision |
|---|---|---|
| exact inert bind | none | `ALLOW` |
| live procfs | `filesystem=proc` | `DENY` |
| wrong empty source | `source_object=$CAPSULE_ROOT/tmp` | `DENY` |
| writable bind | `vfs_read_only=false` | `DENY` |
| nonempty source | `contents=nonempty` | `DENY` |

Every twin must retain the same V10 manifest, cell, payload, root identity,
DynamicUser, network namespace, seccomp surface, and all unaffected mount
facts. A twin that changes more than its named field is invalid evidence.

## Preserved Contracts

V10 preserves V9 effective mount truth, V8 typed file bounds, V7 observer
split, the exact backing tree, static cell and payload, zero capabilities,
`NoNewPrivileges`, private network, and the four-syscall positive seccomp
surface.

The twelve effect families remain unchanged. The bootstrap matrix grows from
four to eight cases: the original treatment and three absent-mountpoint
controls plus the four typed `/proc` sabotage twins. The action-9025 authority
matrix grows from fourteen to eighteen cases. Expected results originate in
Sounio.

## Sounio-First Order

Before any V10 native or host-gate byte exists, a Sounio executable must freeze:

1. the V9 manifest and host-refusal hashes;
2. the `CAPSULE_EMPTY_BIND` predicate;
3. the five typed `/proc` decisions above;
4. the eight-case bootstrap and eighteen-case authority matrices;
5. the unchanged twelve effect-family decisions;
6. the closed product and authority boundary.

C++ remains transitional `MATERIAL_PARITY`. Shell may transport artifacts and
compare Sounio receipts, but cannot define mount types or expected decisions.

## Acceptance

`root_treatment=true` requires the exact inert-bind treatment, the four causal
twins refusing for their named reasons, and the three original missing
mountpoint controls returning `226/NAMESPACE`.

`bootstrap_sabotage=true` requires all seven negative bootstrap controls.
Neither flag implies effect-family material coverage.

Until all material effect families and their own causal twins execute, these
remain false: `material_coverage`, `complete_effects`, `material_execution`,
`production_activation`, `launch_open`, `recycle_open`, `exec_attached`,
`commit_attached`, `ci_attached`, `parity_open`, and `claim_ready`.

## Nonclaims

- V10 does not whitelist `/proc` by pathname.
- V10 does not accept any live `procfs` view.
- V10 does not infer identity from filesystem type or mount source text.
- V10 does not reinterpret the V9 refusal as a pass.
- V10 does not claim any of the twelve effect-family sabotage twins.
- V10 does not open arbitrary commands or any LOOM product attachment.
