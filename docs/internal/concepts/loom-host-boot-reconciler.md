<!-- docs:meta
topic_id: repo.docs.internal.concepts.loom-host-boot-reconciler
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.loom-host-boot-reconciler
-->

<!-- docs:meta
topic_id: repo.docs.internal.concepts.loom-host-boot-reconciler
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.loom-host-boot-reconciler
-->

# Loom Host Boot Reconciler

Concept-ID: `SOUNIO-LOOM-HOST-BOOT-RECONCILER`

Status: executable

Authority: founder

Canonical surface: `stdlib/coordination/loom_host_boot_reconciler.sio`

## Semantic Lane Declaration

```text
Semantic-Lane-ID: loom-hostd-boot-reconcile-20260829
Owner: codex-1
Concept-IDs: SOUNIO-LOOM-HOST-BOOT-RECONCILER
Intent-Preserved: Host reconciliation may restore a surviving physical lane but must never describe a lost Guardian or PTY as physically recovered.
Transformation: Map bound host observations to noop, same-physical recovery, or explicit hold decisions.
Types-Changed: New private host boot observation and Sounio verdict classification.
Effects-Changed: none
IR-Changed: none
Claims-Introduced: A boot-started reconciler can safely automate same-physical kernel recovery when Guardian and harness birth identities remain equal.
Claims-Forbidden: Same PTY after Guardian or host loss; automatic lineage resurrection; production activation; exactly-once effects; distributed consensus.
Assumptions: Linux boot ID and PID start ticks identify the observed host processes within the measured boundary.
Write-Set: Garden, Sounio action 9041, frozen executable, OCaml reconciler, staged systemd installer, receipts.
Read-Set: Existing Loom Guardian/kernel descriptors, fleet catalog, journals, and action 9032 continuity semantics.
Positive-Witness: Active noop and same-Guardian recovery under a frozen desired catalog.
Negative-Witness: Guardian start-tick mismatch and Guardian loss without a successor-lineage transition.
Acceptance-Gate: scripts/ci/sounio_loom_host_boot_reconciler_freeze_selftest.sh
Integration-Target: Existing OCaml Loom Guardian/kernel runtime.
Authoritative-Only-If: Frozen Sounio action 9041 judges every operation and the isolated start-tick mutant admits the unchanged refused witness.
```

## Semantic Boundary

The concept owns the Sounio decision that maps a host observation to
`NOOP_ACTIVE`, `RECOVER_SAME_PHYSICAL`, `HOLD_LINEAGE_REQUIRED`,
`HOLD_DISABLED`, or `HOLD_UNENROLLED`.

It does not own Guardian/PTY implementation, provider authentication, systemd
semantics, or future successor-generation creation. OCaml may observe and
materialize an allowed same-physical recovery, but cannot redefine the result.

## Authority

- Canonical semantic surface:
  `stdlib/coordination/loom_host_boot_reconciler.sio`.
- Producing language: Sounio, action `9041`.
- Operational parity: OCaml Loom runtime.
- Installation transport: systemd and shell.
- Python and Rust are excluded from the authority and operational critical
  path.

## v1 Non-Claim

`HOLD_LINEAGE_REQUIRED` is not a resurrection. It affirmatively records that
the old PTY is gone and prevents the reconciler from opening a successor until
a separately frozen lineage transition exists.
