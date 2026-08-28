# GARDEN: LOOM Systemd Root Mountpoint Policy V4

Status: `PREREGISTERED_AFTER_V3_HOST_FALSIFICATION`

## Falsifying Observation

The frozen V3 policy required a systemd-created mount namespace whose exact
root contained only `loom`, `dev`, `proc`, and `tmp`. A static V3 cell and a
static Sounio payload passed local gates. On `t560-proxmox`, the root was then
bound read-only before `systemd-run` so that systemd could not enlarge it.
Systemd 257 refused before `exec` with status `226/NAMESPACE`:

```text
Failed to create parent directories of destination mount point node
'/run/systemd/incoming': Read-only file system
Failed to mount /run/systemd/propagate/<unit> to /run/systemd/incoming:
No such file or directory
```

The receipt is frozen at
`tools/loom/evidence/loom-process-witness-effect-root-v3-host-attempt-v1-20260828.txt`
with SHA-256
`baeb296039daf112f66d22e7ad7f57e2a605702a964b444dc8a8a1c6325c37e5`.
V3 therefore remains `root_treatment=false`, `material_coverage=false`, and
cannot authorize the effect cell.

## Minimal Correction

V4 changes exactly one semantic fact: the immutable root contains an empty
mountpoint reserved for systemd's per-unit propagation rendezvous:

```text
/
|-- loom/
|   |-- effect-cell
|   |-- payload
|   |-- payload.freeze.v1
|   `-- effect-policy-v4.freeze.v1
|-- dev/
|   `-- null
|-- proc/                         # absent mount in treatment
|-- tmp/                          # empty, effectively read-only
`-- run/
    `-- systemd/
        `-- incoming/             # empty systemd propagation mountpoint
```

`/run`, `/run/systemd`, and `/run/systemd/incoming` are root-owned directories.
They are not writable by the DynamicUser. The backing release is mounted
read-only before the transient unit is created. Systemd may mount only its
per-unit propagation directory on `/run/systemd/incoming`; the cell must
observe the directory empty at READY. No socket, file, credential, environment
record, package runtime, or executable may appear below `/run`.

The host observer must bind the incoming mount to the exact transient unit and
must reject:

- any other new top-level path;
- any other `/run` descendant;
- a writable incoming mount for the DynamicUser;
- a source outside `/run/systemd/propagate/<exact-unit>`;
- contents present at READY;
- a surviving mount or process after unit extinction.

This mountpoint grants PID 1 a rendezvous for namespace construction. It grants
the principal no mount authority: the cell has a non-root DynamicUser,
capabilities zero, `RestrictNamespaces=yes`, `NoNewPrivileges=yes`, and the
positive seccomp filter admits no mount or pathname syscall after installation.

## Bootstrap Causal Control

Before material effect claims, two otherwise identical root launches are
required:

| Case | Intervention | Expected result |
| --- | --- | --- |
| treatment | exact empty `/run/systemd/incoming` exists | cell reaches root READY |
| missing incoming | remove only `run/systemd/incoming` before sealing | systemd refuses before exec with `226/NAMESPACE` |

The same cell, payload, manifests, modes, ownership, unit properties, host,
kernel, systemd version, controller, and timeout are used in both cases. This
bootstrap control is additional to the twelve effect-family twins; it cannot
substitute for any of them.

## Preserved V3 Semantics

V4 preserves without reinterpretation:

- action `9025` and its twelve effect families;
- all fourteen authority frames;
- the four-syscall positive surface `0,1,60,322`;
- argument constraints for fd `0`, fds `1|2`, and `execveat(3, AT_EMPTY_PATH)`;
- architecture mismatch `KILL_PROCESS` and default `ERRNO_EP1`;
- family `10` probe `personality_change`;
- all twelve treatment and single-family sabotage expectations;
- static native cell and static Sounio payload requirements;
- the absence of Landlock fallback;
- every product and parity attachment remaining closed.

## Sounio-First Order

No V4 native byte may change until a self-contained Sounio executable freezes:

1. the V3 manifest and V3 refusal receipt hashes;
2. the corrected root schema;
3. both bootstrap expected results;
4. the unchanged twelve-family matrix and fourteen action-9025 frames;
5. the closed evidence boundary.

The native materializer may then implement only this frozen correction. C++ is
`MATERIAL_PARITY`, not semantic authority.

## Acceptance

V4 may set `root_treatment=true` only after the real host proves the treatment
root reaches READY and the missing-incoming twin repeats `226/NAMESPACE`.
It may set `complete_effects=true` only after that bootstrap proof plus all
twelve treatment effects and all twelve causal twins pass on the same named
host, and frozen action `9025` judges the complete receipt `ALLOW`.

Until then:

```text
root_treatment=false
bootstrap_sabotage=false
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

- V4 does not treat arbitrary `/run` content as acceptable.
- V4 does not make the propagation mount a principal capability.
- V4 does not reinterpret the V3 refusal as a pass.
- V4 does not claim portability beyond the measured host and systemd 257.
- V4 does not weaken the exact root into a conventional Linux filesystem tree.
- V4 does not open arbitrary command execution or any LOOM product attachment.
