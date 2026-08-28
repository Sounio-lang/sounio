# GARDEN: LOOM Process Witness Exact Effect Policy V2

Status: `PREREGISTERED_CORRECTION`

## Trigger

The v1 Sounio policy plan correctly froze the twelve action-`9025` families,
their coverage modes, fourteen authority frames, and the treatment/sabotage
decision classes. Before any native effect-cell bytes were written, review
found that it named a positive seccomp allowlist but did not enumerate its
syscalls or argument constraints.

That omission is material. If C++ selected the list now, `MATERIAL_PARITY`
would retrospectively create part of the expected semantics. The v1 freeze is
preserved as evidence of this stop. It is not sufficient to authorize native
implementation.

## Frozen Parents

- Original effect-cell Garden commit:
  `e2fe391d6ccfc5ffc2813b4fc9d6345ba54afd8a`.
- V1 policy manifest:
  `tools/loom/process_witness_effect_policy_plan.freeze.v1`.
- V1 policy manifest SHA-256:
  `14ee27eee71f04d1aa5462426379b37bb9c775215e94e17a864dbea308e43f21`.
- V1 freeze commit:
  `cab555a3ea1ce286737afec6a459dc6aedbb261b`.
- Action `9025` and ProcessWitness host manifests remain unchanged.

## Exact Post-Sandbox Surface

For the frozen statically linked Sounio handshake payload on Linux x86-64, v2
admits exactly these syscall shapes after Landlock and seccomp are installed:

| NR | Operation | Argument constraint |
| --- | --- | --- |
| `0` | `read` | `fd == 0`; the descriptor is the frozen CLOSE channel. |
| `1` | `write` | `fd == 1 || fd == 2`; stdout/stderr remain bounded and receipt-hashed. |
| `60` | `exit` | terminal status is measured; no continuation follows. |
| `322` | `execveat` | `fd == 3`, empty path, frozen argv, empty environment, `flags == AT_EMPTY_PATH`; fd 3 is the pre-opened, hashed, close-on-exec Sounio payload. |

Every other syscall number and every nonconforming argument shape receives the
same fail-closed seccomp refusal. The architecture check rejects any audit
architecture other than `AUDIT_ARCH_X86_64`. The filter is installed only after
all policy construction, object opening, hashing, Landlock rule creation,
descriptor reduction, and process-posture validation are complete, and before
the first treatment probe or payload transition.

`execveat` is one-shot because descriptor 3 is close-on-exec. The filter may
admit only that descriptor and `AT_EMPTY_PATH`; after the transition, no valid
authorized executable descriptor remains. Path strings do not authorize an
executable.

## Sounio V2 Output

Before native code, a v2 Sounio executable must freeze:

1. the complete v1 family plan and all fourteen action-`9025` frames;
2. the four-row syscall allowlist above, including exact numbers and argument
   constraints;
3. `default_action=ERRNO_EP1`, `architecture=x86_64`, and
   `architecture_mismatch=KILL_PROCESS`;
4. `landlock_required=true`, object rules required before seccomp, no pathname
   fallback, and no blacklist fallback;
5. exact plan and syscall-surface counts;
6. `v1_sufficient_for_native=false` and `v2_required_for_native=true`.

The selftest must prove deterministic output, re-judge all Sounio-produced
frames through frozen action `9025`, reject a native-consumption attempt bound
only to v1, and keep current material evidence at `DENY447`.

## Native Acceptance Boundary

Native effect-cell bytes may begin only after the v2 executable and manifest
are committed and frozen. C++ must consume the v2 manifest and exact bundle
hash. Adding even one allowed syscall, widening one fd/flag constraint, using a
different architecture action, or falling back when Landlock is unavailable
requires another Sounio-first correction.

This correction changes no material claim:

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

## Stop Rule

Stop before native execution if the frozen Sounio payload needs a fifth syscall
or a wider argument shape. Observation may identify the mismatch but cannot
authorize it. The next admissible act is a new Sounio plan freeze, not an
unrecorded C++ exception.
