# Garden: PrincipalCell descriptor barrier v1

Status: preregistered before implementation and measurement.

## Question

Can a host PrincipalCell prove that its execution threshold is a non-bearer,
descriptor-only barrier whose release is causally necessary, while exposing no
user command surface?

## Object under test

The barrier is a minimal transitory C++20 host primitive. A child receives one
read descriptor and one result descriptor. It receives no path, token, shell,
command, script, or semantic expected result. The parent retains the only write
descriptor. The release value is bound to a random generation digest and is
valid exactly once.

The child may report only one terminal state:

- `BARRIER_OPENED` after reading the exact release record once; or
- `BARRIER_CLOSED` after EOF, timeout, malformed data, replay, or generation
  mismatch.

No terminal state executes a user command in v1. `BARRIER_OPENED` proves only
that the material threshold was crossed in the laboratory.

## Treatment and causal sabotage

- Treatment: parent closes the release descriptor without writing. Expected
  material observation: `BARRIER_CLOSED`, no open sentinel.
- Isolated sabotage: with all other facts identical, parent writes the exact
  generation-bound release record once. Expected material observation:
  `BARRIER_OPENED`, exactly one open sentinel.

The treatment and sabotage are both preregistered so the rule cannot be chosen
after observing the result. The sabotage is not a product path and must be
unreachable from the broker socket.

## Quorum contract

Future product release requires all three objects in one broker transaction:

1. resident Sounio action 9030 returns semantic ALLOW;
2. the OCaml ExecGrantCell consumes the matching single-use grant and reaches
   `OUTCOME_PENDING`;
3. the host PrincipalCell identity and generation match, after which the broker
   writes the descriptor-only barrier exactly once.

No single object can cross the threshold. A Sounio DENY, OCaml poison/revoke,
host identity drift, deadline expiry, or broker/resident death closes the write
descriptor without release.

## Preregistered controls

1. EOF without release closes.
2. Timeout without release closes.
3. Wrong generation closes.
4. Truncated and oversized records close.
5. Duplicate release produces at most one open observation and then closes.
6. A copied textual record without the inherited descriptor cannot release.
7. Treatment and isolated sabotage differ only by the parent write.
8. Frozen action 9030 `DENY491` drives treatment, never sabotage.
9. The broker protocol continues to refuse `EXEC`, `LAUNCH`, and direct
   `BARRIER_RELEASE` commands.

## Acceptance boundary

Passing this experiment may establish `descriptor_barrier_causal=true` and
`material_threshold_measured=true`. It does not establish a material grant or
execution. The following remain false until a later integrated quorum gate:

- `material_grant`
- `material_execution`
- `launch_open`
- `exec_attached`
- `parity_open`
- `claim_ready`

