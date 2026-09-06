<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-27-pireus-xor-operation-parent-link
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-27-pireus-xor-operation-parent-link
-->

# Pireus XOR Operation: A Frozen Receipt Is Also An API Boundary

> **Status**: Garden addendum | **Date**: 2026-08-27 | **Authority**: founder direction plus executable compiler evidence

## Why This Addendum Exists

The causal Garden for `SOUNIO-PIREUS-XOR-CONVOLUTION-OPERATION` required the
first executable to import both frozen parent modules. Before any semantic
result was produced, Loom admitted the exact source and two public Sounio
compiler paths were asked to check that composition.

The checks exposed two distinct compiler limitations:

- `lean_single` bundles imported modules into one function namespace. The
  independently valid `xor_convolution.sio` and `graph_identity.sio` parents
  both contain private helpers such as digest builders and negative-surface
  functions. Their names and arities collide after bundling, so calls inside
  the frozen parents are rebound incorrectly.
- the available modular prebuilt identifies itself as Madaros `v0.80.0` and
  does not correctly resolve the imported struct surfaces in this seven-module
  program. It emits broad empty-field diagnostics and cannot establish a
  narrower semantic failure.

No Sounio operation result was produced by either check. These diagnostics do
not falsify either frozen parent, because each parent already executed through
its accepted authority receipt. They do show that co-bundling the two modules
would require editing a frozen parent or weakening the current bootstrap.

Neither is allowed.

## Corrected Parent Interface

The operation executable will consume the parents through two different,
equally explicit interfaces:

1. `stdlib/algebra/xor_convolution.sio` remains a live imported API. It creates
   the operation output and inherited algebra classification in the same
   Sounio execution.
2. Pireus graph identity is consumed as its immutable Sounio authority receipt.
   The operation binds the frozen semantic document, identity-module source,
   authority stream, and all six Sounio-produced composition digests.

The Pireus parent binding is:

```text
identity_module_sha256=caedf51babd450db0af50f9755e677786cc8b563ad923f3598153759859f9985
semantics_sha256=8dc9c6c90d4f21b13c07d8ec3e914839b9f3bfaa1e32f222a25bdcb267c943cb
authority_stream_sha256=5b3efa606d86805aa222ced72a37ed87e7b3dab66b21e58e0547163aa19c83dd
registry_digest=9b56f6f0306d949e2266776ee34f05f3ba1dec4239e0bba9411b3aed9c2b27ce
dependency_digest=4dd37bf1cdd774e4ab840e5444d7b18b8a1d0990063901b8a85743a7ac2abbcc
lifted_graph_digest=0bcf3ef8b9598cb4363864d9ba75d9b050a22df501b80a09eda7290b3e331765
occurrence_digest=57218fbb4a6d640e4651dea0d14a17a54559a2f559e45e3186a46df7d8a05950
collision_digest=3a72cc5158aa0e841b4b13de2a924d1bca516778b651ae3f1fe9be80d26925bb
provenance_digest=1e962677cfb1846a5e5b9dd70c13c25cae5f92ad905f6ad795a8912b4e352f20
```

All values above were born in the earlier Sounio execution and are already
frozen in
`docs/research/receipts/pireus_graph_identity_composition_20260827.md`.
This addendum creates none of them.

The operation graph will use the `lifted_graph_digest` as its immutable parent
identity and bind the other five digests as provenance. It will allocate a new
producer scope only inside its own overlay digest. It will not mutate, union,
renumber, or recompute the seven-graph parent.

## What Is Not Being Relaxed

This is not permission to:

- copy or reimplement the graph-identity algorithm;
- treat prose as a replacement for a Sounio result;
- accept a parent by concept name without exact hashes;
- drop collision, occurrence, or provenance digests;
- edit a frozen parent so the bootstrap happens to compile;
- infer an instruction, lowering, cost, support, or performance claim;
- open parity or claim readiness.

The receipt is an executable API result with stronger byte identity than an
unversioned source import. A one-bit change to any parent binding remains a
mandatory negative failure.

## Updated First-Executable Gate

The original exit-gate item requiring both live imports is replaced only for
the graph-identity parent. The first executable must now:

1. live-import and execute the frozen XorConvolution parent;
2. bind all exact Pireus graph-identity receipt values listed above;
3. make the frozen lifted graph the parent of the new operation overlay;
4. include every parent digest in its dependency digest;
5. reject zero, missing, reordered, or one-bit-changed receipt fields;
6. produce the overlay graph, result, witnesses, and new digests in Sounio;
7. preserve `parity_open=false` and `claim_ready=false`; and
8. create no material observation.

The mandatory stage order remains unchanged:

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

This addendum remains part of `GARDEN`. The unsuccessful compiler checks do not
advance the durable stage.
