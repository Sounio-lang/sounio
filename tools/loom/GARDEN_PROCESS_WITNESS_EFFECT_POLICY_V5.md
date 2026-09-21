# GARDEN: LOOM Systemd Sys Mountpoint Policy V5

Status: `PREREGISTERED_AFTER_V4_HOST_FALSIFICATION`

## Falsifying Observation

The frozen V4 policy added the exact empty
`/run/systemd/incoming` rendezvous required by systemd 257. Its static native
cell, static Sounio payload, and eleven-path root capsule passed all local
gates. On `t560-proxmox`, the root was bind-mounted read-only before
`systemd-run`. Namespace construction advanced beyond the V3 refusal and then
stopped at a second absent mountpoint:

```text
Failed to set up mount namespacing: /sys: No such file or directory
Main process exited, code=exited, status=226/NAMESPACE
```

The receipt is frozen at
`tools/loom/evidence/loom-process-witness-effect-root-v4-host-attempt-v1-20260828.txt`
with SHA-256
`2659e6881403784034ab0078a5de64a1eb35c2c96d8c563b98a951c45ac09b9e`.
V4 therefore remains `root_treatment=false`, `bootstrap_sabotage=false`, and
cannot authorize the effect cell.

## Minimal Correction

V5 changes one root-schema fact: the immutable backing root also contains an
empty root-owned `/sys` directory reserved as the mountpoint needed by the
systemd namespace constructor:

```text
/
|-- loom/
|   |-- effect-cell
|   |-- payload
|   |-- payload.freeze.v1
|   `-- effect-policy-v5.freeze.v1
|-- dev/
|   `-- null
|-- proc/                         # absent mount in treatment
|-- run/
|   `-- systemd/
|       `-- incoming/             # empty propagation rendezvous
|-- sys/                           # empty backing mountpoint
`-- tmp/                           # empty, effectively read-only
```

The backing `/sys` is empty and non-writable by the DynamicUser before launch.
If systemd mounts a kernel view on it, that mount is material state outside the
frozen backing tree. The host observer must identify its mount type, root,
source, options, and lifetime while the cell is at READY. The observer must
reject a writable principal view, a non-kernel source, contents in the backing
directory after extinction, or a surviving mount after process extinction.

This mountpoint grants the system manager only the namespace construction it
already attempted. It grants the principal no mount authority: the cell keeps
a non-root DynamicUser, zero capabilities, `RestrictNamespaces=yes`,
`NoNewPrivileges=yes`, and a positive four-syscall seccomp filter with no mount
or pathname operation after installation.

## Bootstrap Causal Controls

Three launches use the same cell, payload, manifests, modes, ownership, host,
kernel, systemd version, controller, properties, and timeout:

| Case | Sole intervention | Expected result |
| --- | --- | --- |
| treatment | both exact mountpoints exist | cell reaches root READY |
| missing incoming | remove only `/run/systemd/incoming` | pre-exec `226/NAMESPACE` naming `/run/systemd/incoming` |
| missing sys | remove only `/sys` | pre-exec `226/NAMESPACE` naming `/sys` |

The two denial twins prove that V5's two added schema facts, rather than an
unrelated property, cross their corresponding namespace barriers. They remain
bootstrap controls and cannot substitute for the twelve effect-family twins.

## Preserved Semantics

V5 preserves without reinterpretation:

- action `9025`, twelve effect families, and fourteen authority frames;
- the four-syscall surface `0,1,60,322` and every argument constraint;
- architecture mismatch `KILL_PROCESS` and default `ERRNO_EP1`;
- family `10` probe `personality_change`;
- static native cell and static Sounio payload requirements;
- the absence of a Landlock fallback;
- the V4 incoming rendezvous contract;
- every product, execution, commit, CI, and parity attachment remaining closed.

## Sounio-First Order

No V5 native byte may change until a Sounio executable freezes:

1. the V4 manifest and V4 `/sys` refusal receipt hashes;
2. the corrected backing-root schema;
3. all three bootstrap expected results;
4. the unchanged twelve-family matrix and action-9025 frames;
5. the closed evidence boundary.

C++ may then implement only that frozen plan as transitional
`MATERIAL_PARITY`; it cannot define or revise an expected result.

## Acceptance

V5 may set `root_treatment=true` only after the real host reaches READY and
both single-absence twins repeat `226/NAMESPACE` with the expected path. It may
set `complete_effects=true` only after those bootstrap proofs, all twelve
treatment effects, all twelve causal twins, and the frozen action-9025 judge
agree on the same host receipt.

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

- V5 does not expose a conventional host `/sys` by default.
- V5 does not accept arbitrary `/sys` mounts or writable kernel controls.
- V5 does not reinterpret either V3 or V4 refusal as a pass.
- V5 does not claim portability beyond the measured host and systemd 257.
- V5 does not open arbitrary command execution or a LOOM product attachment.
