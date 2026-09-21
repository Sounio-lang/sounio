<!-- docs:meta
topic_id: repo.docs.internal.concepts.loom-causal-workflow-mid-exec
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.loom-causal-workflow-mid-exec
-->

# SOUNIO-LOOM-CAUSAL-WORKFLOW-KERNEL / mid-exec-v1

Concept-ID: `SOUNIO-LOOM-CAUSAL-WORKFLOW-KERNEL`

Subordinate-Contract: `mid-exec-v1`

Kind: `executable`

Owner: `founder`

Semantic-Lane-ID: `loom-causal-mid-exec-v1-20260831`

Parent-Contract: `docs/internal/concepts/loom-causal-workflow-kernel.md`

Intent-Preserved: transport loss cannot turn a replacement process into the
continuation of material work already installed in the kernel's exec image.

Transformation: refine Sounio action 9037 with the versioned subordinate
contract `mid-exec-v1`. It judges two observations of the same material
execution: admission to release a host-held exec barrier after transport loss,
and continuity after the material process completes.

Types-Changed: add a bound mid-exec witness containing HostGuardian generation,
systemd unit InvocationID, material PID, process start tick, cgroup, action-9030
run-grant generation, digest-bound barrier nonce, and successor/predecessor
relationship. No parent type changes meaning.

Effects-Changed: none. The Sounio executable judges observations but does not
launch, signal, detach, or otherwise control a material process.

IR-Changed: none.

Claims-Introduced: `RELEASE_ADMISSION` accepts only the exact authenticated
mid-exec witness after transport loss with one compile, ticket, and launch and
zero completion, result, and attestation records. `CLAIM_CONTINUITY` accepts
only the same witness after completion with exactly one record of all five
kinds.

Claims-Forbidden: material execution, ptrace barrier realization, Pod-loss
survival, operational recovery, HostGuardian-loss recovery, production
activation, `PARITY_OPEN`, and `CLAIM_READY` remain unsupported until separately
implemented and measured.

Assumptions: the root-owned HostGuardian generation and held material process
survive transport-Pod loss; loss of either must fail closed.

Write-Set: this versioned subordinate contract, Sounio subordinate authority
and packed adapter, freeze manifests, gates, and evidence. The parent contract
and Concept-ID registry remain byte-identical to their frozen versions.

Read-Set: frozen action 9037 parent semantics, action-9030 run-grant identity,
the source-built compiler, and authenticated host observations. Parent
semantics are referenced by hash and are not modified.

Positive-Witness: the release observation binds the exact held material
identity with pre-completion counts, and the later continuity observation binds
the same identity with exactly-once completion counts.

Negative-Witness: an otherwise unchanged release substitutes only the material
PID and is refused with `DENY593`; an otherwise unchanged continuity claim
substitutes only the barrier nonce and is refused with `DENY600`.

Acceptance-Gate: `scripts/ci/sounio_loom_causal_workflow_mid_exec_selftest.sh`.

Integration-Target: a C++ material cell that proves `PTRACE_EVENT_EXEC` before
holding the process, an OCaml HostGuardian journal, and a real Pod A deletion /
Pod B reattachment experiment.

Authoritative-Only-If: the unmodified Sounio executable refuses both negative
witnesses while mutants deleting only the respective identity rule admit them.

Semantic-Boundary: a successor may request release or claim continuity only by
presenting an observation of the frozen witness. It cannot create a process,
run grant, barrier nonce, expected result, or receipt, and it cannot promote a
replacement material identity into the predecessor's execution.
