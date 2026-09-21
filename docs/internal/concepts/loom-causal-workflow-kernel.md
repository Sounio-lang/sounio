<!-- docs:meta
topic_id: repo.docs.internal.concepts.loom-causal-workflow-kernel
authority: repo_only
audience: users
last_validated: 2026-08-30
validated_by: codex-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.loom-causal-workflow-kernel
-->

# SOUNIO-LOOM-CAUSAL-WORKFLOW-KERNEL

Concept-ID: `SOUNIO-LOOM-CAUSAL-WORKFLOW-KERNEL`

Kind: `executable`

Owner: `founder`

Semantic-Lane-ID: `loom-causal-workflow-kernel-v1-20260830`

Intent-Preserved: a workflow remains reconstructible after controller or Pod
loss because its authority and progress live in a verified receipt chain, not
in tmux, a provider conversation, or process memory.

Transformation: introduce Sounio action 9037 as the semantic authority for the
exact `COMPILE -> RUN_EXACT -> ATTEST` graph, its expected result, its state
transitions, and its recovery relation.

Types-Changed: add typed workflow, artifact, observation, and attestation
identities; no existing type changes meaning.

Effects-Changed: add two explicitly bounded material effects: `COMPILE` creates
exactly one immutable Guardian-custodied artifact record, and `RUN_EXACT`
executes it once. The HostGuardian-custodied run ticket is a non-bearer durable
idempotency record bound one-to-one to the action-9030 grant identity and
generation; action 9030 remains the only launch authority. No arbitrary
execution effect is added.

IR-Changed: none.

Claims-Introduced: the frozen Sounio executable can judge valid workflow and
recovery observations and refuse a predecessor-receipt substitution.

Claims-Forbidden: operational recovery, Pod survival, material exactly-once,
HostGuardian-loss recovery, arbitrary shell execution, production activation,
`PARITY_OPEN`, and `CLAIM_READY` remain unsupported until separately measured.

Assumptions: controller or Pod loss does not destroy the root-owned
HostGuardian generation or its durable receipt store; stronger loss fails
closed.

Write-Set: Garden, this contract, Concept-ID registry row, Sounio action 9037,
packed executable adapter, freeze manifests, gates, and evidence.

Read-Set: frozen actions 9030 through 9036; the source-built compiler; and the
host-proven operation-cell receipt. Actions 9031, 9033, and 9034 contribute
result transparency, physical-principal evidence, and rebuild admission,
respectively. Action 9032 is predecessor provenance only: it does not authorize
a replacement controller while its Guardian remains live. Action 9037 owns its
new controller-successor relation. Actions 9035 and 9036 contribute measured
compiler and result-record provenance only; neither authorizes `RUN_EXACT` or
`ATTEST`, and their write-absent claims are not inherited.

Positive-Witness: the canonical workflow reaches `ATTESTED`, and recovery from
`COMPILE_SEALED` preserves its generation and predecessor receipt without
recompiling, then binds one run-ticket record to one action-9030 grant and
launches exactly once.

Negative-Witness: the otherwise unchanged recovery witness substitutes only
the predecessor receipt and is refused with `DENY589`.

Acceptance-Gate: `scripts/ci/sounio_loom_causal_workflow_kernel_selftest.sh`.

Integration-Target: OCaml append-only workflow journal, followed by three
fresh DynamicUser ExecCells under the existing HostGuardian.

Authoritative-Only-If: the Sounio executable refuses the unchanged receipt
substitution with `DENY589`, while a mutant deleting only the receipt equality
admits it.

Semantic-Boundary: a controller may replay observations and request the next
frozen transition. It cannot create workflow nodes, expected results, handles,
run-ticket records, or execution grants. Artifact handles and run tickets are
non-bearer identities and never caller paths or launch authority.
