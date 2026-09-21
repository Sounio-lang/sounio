# GARDEN: LOOM DynamicUser Var-Tmp Policy V6

Status: `PREREGISTERED_AFTER_V5_HOST_FALSIFICATION`

## Falsifying Observation

V5 added exact systemd mountpoints for `/run/systemd/incoming` and `/sys`.
The namespace constructor advanced again, then DynamicUser preparation refused
before exec with `226/NAMESPACE` because `/var/tmp` was absent:

```text
source=/run/systemd/unit-private-tmp/var-tmp
destination=/var/tmp
result=No such file or directory
```

The frozen receipt is
`tools/loom/evidence/loom-process-witness-effect-root-v5-host-attempt-v1-20260828.txt`
with SHA-256
`1cfd0bba84732d156f220b20ede0cfd9cbf22b3902f474ebada922f29506272f`.
V5 remains refused and cannot authorize execution.

## Minimal Correction

V6 adds only the root-owned directory chain `/var/tmp` to the immutable backing
tree. It does not accept DynamicUser's writable private directory as the final
cell view. The transient unit must additionally overlay the frozen backing
`/tmp` on both `/tmp` and `/var/tmp` using read-only binds. At READY:

- both mountpoints are empty and effectively read-only;
- their mount roots identify the same frozen backing directory;
- the principal cannot create a file in either path;
- no other `/var` descendant exists;
- the private staging mount is not an authority surface for the cell.

The complete backing tree is V5 plus:

```text
`-- var/
    `-- tmp/                      # empty destination; read-only overlay at READY
```

## Bootstrap Causal Controls

Four otherwise identical launches are required:

| Case | Sole intervention | Expected result |
| --- | --- | --- |
| treatment | all exact mountpoints exist | cell reaches root READY |
| missing incoming | remove only `/run/systemd/incoming` | `226/NAMESPACE` naming that path |
| missing sys | remove only `/sys` | `226/NAMESPACE` naming `/sys` |
| missing var tmp | remove only `/var/tmp` | `226/NAMESPACE` naming `/var/tmp` |

These are bootstrap controls, not substitutes for the twelve effect-family
twins.

## Preserved Semantics

V6 preserves action `9025`, all twelve families and fourteen frames, the
four-syscall positive surface, every argument constraint, static native and
Sounio payload requirements, the exact incoming and sys contracts, DynamicUser,
zero capabilities, private mount and network namespaces, and every product or
parity attachment remaining closed.

## Sounio-First Order

Before any V6 native byte changes, a Sounio executable must freeze the V5
manifest and refusal hashes, the `/var/tmp` correction, all four bootstrap
results, the unchanged effect matrix, and the closed evidence boundary. C++ is
transitional `MATERIAL_PARITY` only.

## Acceptance

`root_treatment=true` requires READY plus all three single-absence refusals on
the same host. `complete_effects=true` additionally requires all twelve
treatments, all twelve causal twins, and an action-9025 `ALLOW` for their frozen
receipt.

Until then all authority and product flags remain false, including
`root_treatment`, `bootstrap_sabotage`, `material_coverage`,
`complete_effects`, `material_execution`, `launch_open`, `exec_attached`,
`commit_attached`, `ci_attached`, `parity_open`, and `claim_ready`.

## Nonclaims

- V6 does not authorize writable `/tmp` or `/var/tmp`.
- V6 does not accept arbitrary `/var` contents.
- V6 does not reinterpret a prior refusal as success.
- V6 does not claim portability beyond the measured host and systemd 257.
- V6 does not open arbitrary commands or a LOOM product attachment.
