<!-- docs:meta
topic_id: repo.docs.internal.concepts.loom-kernel-peer-activation-capsule
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.loom-kernel-peer-activation-capsule
-->

# SOUNIO-LOOM-KERNEL-PEER-ACTIVATION-CAPSULE

## Status

Semantics frozen in Sounio action `9031`. The canonical semantic surface is
`stdlib/coordination/loom_kernel_peer_activation_capsule_authority.sio`.

## Concept

A kernel peer activation capsule is an affine, non-bearer transition witness
joining two independently frozen authorities:

- action `9025`, which establishes the materially observed same-UID peer
  mediation and effect-closure facts;
- action `9030`, which establishes the one-shot `ExecGrantCell` lifecycle.

The join remains indexed by the live boot, BPF policy epoch, authenticated
principal, harness ancestry, guardian custody generation, and request sequence.
The serialized capsule or its digest never authorizes an operation.

## State Law

The only nonterminal path is:

`EMPTY -> SEALED -> CONSUMED -> EXTINCT`

Any uncertainty from `SEALED` or `CONSUMED` moves to terminal `POISONED`.
Neither terminal state can be recycled. A second operation requires a fresh
generation and fresh parent decisions.

## Absence Law

Terminal absence is a conjunction of three positive observations:

`registry_absent AND kernel_extinct AND replay_refused`

Missing data, timeout, EOF, lookup failure, or silence proves none of them.
This makes revocation evidence constructive and prevents a crashed guardian
from laundering uncertainty into success.

## Write Law

The old graph, proposed graph, all identity edges, parent receipts, future
terminal obligation, and transition legality are validated before mutation.
An invalid proposed state is refused without consuming the parent grant. The
material implementation must use one single-writer compare-and-exchange bound
to the exact Sounio decision frame.

## Authority

Sounio is `SEMANTIC_AUTHORITY`. OCaml may later implement the operational
kernel but cannot encode expected decisions. C++20/Linux/BPF remains a
transitory material observation layer. Lean 4 and Koka may open parity only
after the Sounio semantics freeze. Python and Rust are prohibited.

## Nonclaims

Preregistration does not create a material capsule, attach execution, open
launch or recycle, or establish product readiness.
