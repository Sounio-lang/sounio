# GARDEN: LOOM Opaque Rendezvous Observer Split V7

Status: `PREREGISTERED_AFTER_V6_POST_EXEC_FALSIFICATION`

## Falsifying Observation

V6 supplied every mountpoint discovered by V3-V5 and bound both `/tmp` and
`/var/tmp` read-only. On `t560-proxmox`, systemd 257 completed namespace
construction and started the static native cell. The cell then closed with
status `70/SOFTWARE` while enumerating `/run/systemd/incoming`:

```text
LOOM_PROCESS_WITNESS_EFFECT_POLICY_V6_CLOSED
reason=cannot inspect root directory: /run/systemd/incoming
```

The frozen receipt is
`tools/loom/evidence/loom-process-witness-effect-root-v6-host-attempt-v1-20260828.txt`
with SHA-256
`04e296b2f27c54b598f6902ee936a1d6eee1354f046bafb38e0d928a6f1941b5`.
Unlike V3-V5, this is not a pre-exec namespace refusal. It proves the V6
namespace was constructed and the native cell started. V6 failed because it
assigned a root-observer obligation to the unprivileged principal.

## Observer Split

V7 changes no backing-root path and no effect-family rule. It changes exactly
one epistemic assignment:

| Fact | Authorized observer |
| --- | --- |
| incoming path exists and is a directory | principal cell before seccomp |
| incoming metadata is root-owned and principal-non-writable | principal cell before seccomp |
| incoming mountpoint is distinct | root host observer through mountinfo |
| incoming source belongs to the exact transient unit | root host observer through mountinfo |
| incoming contents are empty at READY | root host observer through `/proc/<pid>/root` |
| incoming mount disappears with process/unit extinction | root host observer |

The principal must not enumerate, open, or read the rendezvous. Principal
opacity is a positive security property, not missing evidence. The root host
observer carries the vacuity proof in the material receipt.

## Preserved Material Contract

V7 preserves the exact V6 backing tree, DynamicUser, private mount and network
namespaces, zero capabilities, `NoNewPrivileges`, `RestrictNamespaces`, both
read-only temporary mounts, static cell and payload, and the four-syscall
positive seccomp surface. The host treatment still requires:

- `/run/systemd/incoming` sourced from the exact unit propagation path;
- `/sys` as read-only `sysfs`;
- `/tmp` and `/var/tmp` sourced from the same immutable backing `/tmp`;
- exact fd inventory `0,1,2` and process extinction.

## Bootstrap Causal Controls

The V6 four-case matrix is unchanged: treatment reaches READY; removing only
incoming, sys, or var-tmp produces the corresponding `226/NAMESPACE`. The
incoming treatment adds a root-observer empty-directory proof and a
principal-opacity proof. No bootstrap case substitutes for an effect-family
twin.

## Sounio-First Order

Before a V7 native byte changes, a Sounio executable must freeze:

1. the V6 manifest and post-exec refusal hashes;
2. the observer-role assignment above;
3. the unchanged root schema and four bootstrap expected results;
4. the unchanged twelve-family matrix and action-9025 frames;
5. the closed product and authority boundary.

C++ remains transitional `MATERIAL_PARITY`. Shell may transport and compare
receipts but cannot invent expected results.

## Acceptance

`root_treatment=true` requires READY, root-observer proof of an empty exact-unit
incoming mount, principal opacity, and all three absence controls. Material
effect closure still requires twelve treatments, twelve causal twins, and a
frozen action-9025 `ALLOW` on the same host receipt.

Until then every promotion flag remains false: `root_treatment`,
`bootstrap_sabotage`, `material_coverage`, `complete_effects`,
`material_execution`, `launch_open`, `recycle_open`, `exec_attached`,
`commit_attached`, `ci_attached`, `parity_open`, and `claim_ready`.

## Nonclaims

- V7 does not weaken incoming ownership or permissions.
- V7 does not make the principal a host observer.
- V7 does not reinterpret the V6 refusal as a pass.
- V7 does not yet claim any of the twelve material sabotage twins.
- V7 does not open arbitrary commands or any LOOM product attachment.
