# Garden: Host-Durable Lane Supervisor v1

Status: `GARDEN`

## Question

What must remain true when the UI, tmux, or an entire transport Pod disappears
while an interactive lane is still executing?

The answer must distinguish two physically different events:

1. **same-physical reattach**: the Guardian, PTY, harness, boot, command, and
   durable journals are the same objects; only disposable clients and the
   recoverable OCaml kernel changed;
2. **lineage resurrection**: the Guardian or host was lost, so the old PTY is
   gone; a new generation may start only from a verified predecessor receipt
   and lineage proof.

Calling the second event the first is forbidden. No implementation can recover
the same Unix PTY after its Guardian or host has ceased to exist.

## Language Authority

- Sounio: `SEMANTIC_AUTHORITY`, action `9032`.
- OCaml: `EFFECT_PARITY` and the existing Loom Guardian/kernel realization.
- C++ and Linux/systemd: `MATERIAL_PARITY` only when measuring host identities.
- External LLMs: `REVIEW_ONLY`.
- Python and Rust: prohibited from the authority and material critical path.

The mandatory stage order is:

`GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> PARITY_OPEN -> CLAIM_READY`

## Same-Physical Reattach

Sounio may classify `SAME_PHYSICAL_REATTACH` only when all of these are
affirmatively observed:

- the predecessor transport process is extinct and a different transport has
  connected;
- the durable state root is hash-bound and unchanged;
- Guardian PID, Linux start tick, instance id, boot id, and liveness match;
- harness PID, Linux start tick, command digest, and liveness match;
- the durable output prefix is unchanged;
- semantic and Guardian journals verify;
- the replacement OCaml kernel reports recovery of the same instance;
- the observation is material, sabotage-complete, and produced without Python
  or Rust.

A PID without its start tick is not identity. A matching state path without its
bound digest is not custody. A surviving output file without a live matching
Guardian and harness is not the same physical lane.

## Lineage Resurrection

When the Guardian is absent, Sounio must refuse any claim that the same PTY
survived. It may classify `LINEAGE_RESURRECTION` only when a new generation is
explicit, the predecessor receipt and both journal heads are bound, the
lineage verifies, the command and durable root remain bound, and the old
transport is extinct.

This is recovery of work identity, not recovery of the old process identity.

## Preregistered Sabotage

The load-bearing rule is the Guardian start-tick equality check. The positive
same-physical frame is copied exactly except that `guardian_start_equal=0`.
The shipped Sounio policy must refuse it. An isolated mutant that removes only
that equality guard must admit the unchanged witness.

## First Material Canary

The first alpha is deliberately destructive but isolated:

1. a temporary privileged hostPID transport Pod A launches the existing OCaml
   Loom Guardian on the host;
2. the Guardian owns a deterministic OCaml child through `forkpty`, with state
   outside the Pod filesystem;
3. Pod A is deleted and its UID is observed absent;
4. a fresh Pod B enters the same host namespace and verifies the same Guardian,
   harness, PTY generation, command, state root, and output prefix;
5. the OCaml kernel is killed, Pod B runs `recover`, and the same physical
   identities are verified again;
6. the measured frame is judged by the frozen action-9032 Sounio executable;
7. the lane is stopped and affirmative process extinction is verified.

This canary does not promote production fleet custody. It proves the primitive
needed to make the workspace Pod and every presentation layer disposable.
