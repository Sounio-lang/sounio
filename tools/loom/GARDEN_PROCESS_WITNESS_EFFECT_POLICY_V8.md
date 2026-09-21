# GARDEN: LOOM Typed Immutable-Root File Bounds V8

Status: `PREREGISTERED_AFTER_V7_POST_EXEC_FALSIFICATION`

## Falsifying Observation

V7 crossed systemd namespace construction, started the static native cell, and
crossed the principal-opacity check for `/run/systemd/incoming`. It then closed
with status `70/SOFTWARE` while hashing `/loom/effect-cell`:

```text
LOOM_PROCESS_WITNESS_EFFECT_POLICY_V7_CLOSED
reason=policy manifest is not one bounded regular file
```

The frozen receipt is
`tools/loom/evidence/loom-process-witness-effect-root-v7-host-attempt-v1-20260828.txt`
with SHA-256
`a4e9bab136988e6034a775e347ecc6642d7624dedfbf35b3e52d5b14236929bb`.
The V7 static cell is 7,281,352 bytes while the shared reader's default bound is
131,072 bytes. V3-V6 carried the same latent bound; their earlier refusals
occurred before this call path.

## Typed Bounds

V8 replaces the generic file-reader bound with a closed, object-typed table:

| Object | Minimum | Maximum | Hash authority |
| --- | ---: | ---: | --- |
| `/loom/effect-cell` | 1 byte | 16 MiB | root host receipt plus self-hash at READY |
| `/loom/payload` | 1 byte | 1 MiB | frozen Sounio payload hash |
| `/loom/effect-policy-v8.freeze.v1` | 1 byte | 64 KiB | frozen V8 manifest hash |
| `/loom/payload.freeze.v1` | 1 byte | 64 KiB | frozen payload-manifest hash |

Every object remains a root-owned regular file with link count one. Executables
remain non-writable and static. Bounds limit allocation and reading; hashes,
metadata, and the exact root schema establish identity.

The native refusal must name the object, observed size, and configured maximum.
A generic message that misidentifies the object as a policy manifest is no
longer accepted evidence.

## Preserved Observer Contract

V8 preserves V7's observer split unchanged. The principal proves incoming
existence, ownership, non-writability, and opacity. Only the root host observer
may establish the distinct exact-unit mount, empty contents at READY, and mount
extinction with the unit.

## Preserved Material Contract

V8 changes no effect-family rule, syscall surface, root path, systemd property,
or bootstrap expected result. It preserves the static cell and payload,
DynamicUser, private mount and network namespaces, zero capabilities,
`NoNewPrivileges`, both read-only temporary mounts, and the four-syscall
positive seccomp surface.

The four bootstrap cases remain treatment, missing incoming, missing sys, and
missing var-tmp. Their expected results continue to originate in Sounio.

## Sounio-First Order

Before a V8 native byte changes, a Sounio executable must freeze:

1. the V7 manifest and post-exec refusal hashes;
2. the four typed file bounds above;
3. the unchanged observer-role assignment;
4. the unchanged root schema and bootstrap matrix;
5. the unchanged twelve-family action-9025 matrix;
6. the closed product and authority boundary.

C++ remains transitional `MATERIAL_PARITY`. Shell transports and compares
receipts but does not choose bounds or expected results.

## Acceptance

`root_treatment=true` still requires READY, principal opacity, root-observed
empty exact-unit incoming mount, process and mount extinction, and all three
absence controls. Passing typed bounds alone is insufficient.

Until then every promotion flag remains false: `root_treatment`,
`bootstrap_sabotage`, `material_coverage`, `complete_effects`,
`material_execution`, `launch_open`, `recycle_open`, `exec_attached`,
`commit_attached`, `ci_attached`, `parity_open`, and `claim_ready`.

## Nonclaims

- V8 does not widen the post-seccomp syscall surface.
- V8 does not make file size sufficient for file identity.
- V8 does not weaken principal opacity or root-observer duties.
- V8 does not reinterpret the V7 refusal as a pass.
- V8 does not claim any material effect-family sabotage twin.
- V8 does not open arbitrary commands or any LOOM product attachment.
