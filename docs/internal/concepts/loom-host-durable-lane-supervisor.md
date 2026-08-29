<!-- docs:meta
topic_id: repo.docs.internal.concepts.loom-host-durable-lane-supervisor
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.loom-host-durable-lane-supervisor
-->

# Loom Host-Durable Lane Supervisor

Concept-ID: `SOUNIO-LOOM-HOST-DURABLE-LANE-SUPERVISOR`
Status: executable
Authority: founder
Canonical surface: `stdlib/coordination/loom_host_durable_lane_supervisor.sio`

## Semantic Lane Declaration

```text
Semantic-Lane-ID: loom-product-exec-ingress-20260829
Owner: codex-1
Concept-IDs: SOUNIO-LOOM-HOST-DURABLE-LANE-SUPERVISOR
Intent-Preserved: Lane execution must survive disposable presentation and transport processes without laundering a new process into the identity of the old one.
Transformation: Distinguish same-physical reattach from proof-linked lineage resurrection.
Types-Changed: New private observation and Sounio verdict classification.
Effects-Changed: none
IR-Changed: none
Claims-Introduced: A host Guardian can preserve one physical PTY and harness across UI, tmux, transport-Pod, and recoverable-kernel loss when all identities remain equal.
Claims-Forbidden: Same PTY after Guardian or host loss; exactly-once effects; storage replication; partition tolerance; Byzantine consensus; production fleet promotion.
Assumptions: Linux PID start ticks, boot id, filesystem durability, and OCaml journal measurements are accurate within the measured host boundary.
Write-Set: Garden, Sounio action 9032, native fixture, OCaml canary, host canary, freeze receipts.
Read-Set: Existing Loom Guardian, continuity adapter, Beagle bridge, and host ExecCell capsule.
Positive-Witness: Same Guardian, harness, instance, command, boot, state root, output prefix, and verified journals after transport replacement and kernel recovery.
Negative-Witness: Guardian PID reused with a different start tick; Guardian loss without verified lineage.
Acceptance-Gate: scripts/ci/sounio_loom_host_durable_lane_supervisor_host_freeze_selftest.sh
Integration-Target: Existing OCaml Loom Guardian and future host fleet supervisor.
Authoritative-Only-If: Frozen Sounio action 9032 judges a real separate-Pod host receipt and the preregistered one-rule mutant admits the unchanged refused witness.
```

## Required Distinctions

```text
same physical Guardian and PTY != successor generation with lineage
transport replacement          != execution replacement
durable journal replay         != exactly-once external effect
single-host persistence        != replicated storage
verified host identity         != semantic correctness of model output
```

The OCaml Guardian is the operational custodian. It cannot define or rewrite
these distinctions. The Sounio executable is the first executable authority.
