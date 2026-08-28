# LOOM InvocationCell Material Admission V1

Status: `MATERIAL_PARITY_FROZEN_EXECUTION_REFUSED`

This phase binds the transitory C++ host broker to frozen Sounio action `9029`.
It is an admission adapter, not a second semantic implementation and not a
material process launcher.

## Authority

Sounio remains `SEMANTIC_AUTHORITY`. The broker accepts only the exact frozen
action `9029` manifest and the exact source-built authority executable named by
that manifest. It reads one bounded single-line frame, invokes Sounio under a
five-second fail-closed deadline, validates the decision/exit-code pair, and
emits a hash-bound material-parity receipt.

C++20 is `MATERIAL_PARITY` and explicitly transitory. It contains no expected
`DENY481` result and cannot replace the Sounio decision. The same broker binary
must transport both a frozen positive frame and the current material frame:
Sounio returns `ALLOW` for the former and `DENY481` for the latter. The existing
Sounio causal sabotage gate proves that removing only the parent-join rule turns
the unchanged current-material witness into `ALLOW`.

## Refusal Boundary

The diagnostic does not create an `ExecGrant`, open a barrier, retain a pidfd,
resume a process, or mutate the lease journal. `LAUNCH` and `RECYCLE` remain
closed in the live broker protocol. The receipt therefore states:

```text
material_invocation=false
same_uid_peer_isolation=false
launch_open=false
parity_open=false
claim_ready=false
```

Manifest drift, executable drift, authority timeout, abnormal termination,
multiline or oversized frames, malformed decision output, and decision/exit
disagreement fail closed before any material action.

## Adversarial Gate

`scripts/ci/sounio_loom_kernel_invocation_cell_material_admission_selftest.sh`
proves deterministic C++ rebuilds, exact frozen-authority binding, positive and
current-material routing through the same binary, manifest and executable
tamper refusal, multiline-frame refusal, malformed-frame Sounio denial, and the
continued closure of every material broker operation.

`scripts/ci/sounio_loom_kernel_invocation_cell_material_admission_freeze_selftest.sh`
rebuilds the broker twice, rebuilds action `9029` from Sounio source, verifies
the implementation commit and toolchain, replays the adversarial gate, and
binds the complete result into `kernel_invocation_cell.material.v1`.

## Remaining Host Proof

Material realization remains closed until a real systemd host proves all of
the following in one observation lineage: disjoint kernel UID/GID principals,
namespace and cgroup setup, irreversible privilege drop, broker-only pidfd and
barrier custody, exact peer/ancestry/object identity, atomic one-shot grant
consumption, kill-tree timeout, broker-crash poisoning, affirmative effect
closure, and complete terminal receipts. The Pod used for this freeze has none
of that authority and is intentionally refused.
