<!-- docs:meta
topic_id: repo.docs.handoff.issue901-bootstrap-recovery-2026-07-25
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.issue901-bootstrap-recovery-2026-07-25
-->

# #901 Madaros Bootstrap Recovery

## Decision

Madaros is the operational seed for the modular compiler. A legacy bootstrap
may be used only as a quarantined, auditable bridge when the tracked Madaros
ELF cannot parse the current compiler closure. It is not a normal bootstrap
mode and it is not evidence that the legacy path remains authoritative.

The recovery gate is:

```bash
bash scripts/ci/madaros_bootstrap_recovery_901_gate.sh
```

It deliberately proves this chain rather than accepting a prebuilt artifact:

```text
declared legacy raw ELF -> bridge Madaros -> stage1 Madaros -> stage2 Madaros
```

## Acceptance Contract

The gate succeeds only when all of the following are true:

1. The declared legacy bootstrap produces a raw-ELF bridge from the current
   `self-hosted/compiler/main.sio` source using the explicit `lean-audit`
   mode.
2. The bridge reports a complete, parser-clean compiler closure.
3. The bridge rebuilds the source as stage 1, and stage 1 rebuilds it as stage
   2, both through the explicit `madaros-seed` mode.
4. Stage 1 and stage 2 are byte-identical by SHA-256.
5. Stage 2 passes the direct raw-ELF nominal imported-layout witness.
6. Stage 2 passes the direct raw-ELF 256/257 layout-capacity boundary gate.
7. Stage 2 compiles and executes direct raw-ELF contextual `scope`, `policy`,
   `is`, and `study` binding witnesses, while their keyword identity remains
   structural outside ordinary local binding and expression positions.

Every build in one recovery run receives a lock private to that run's work
directory. This prevents scheduler-worker residue from being mistaken for a
compiler result while still serializing the three generations.

The receipt records the source commit, the legacy and generated hashes, both
direct runtime gate outcomes, and the fact that promotion requires a separate
reviewed tracked-seed update.

## What It Does Not Claim

Passing this recovery gate does not automatically replace
`bin/madaros-linux-x86_64`, close the independent AST-closure blocker in #1194,
or prove the historical C-to-lean bootstrap root. It proves that a specified
current-source snapshot has regained an operational Madaros fixed point and
that the #901 semantic witnesses hold at that fixed point.

The normal source-bound gate remains:

```bash
bash scripts/ci/madaros_imported_runtime_source_fresh_gate.sh
```

Once a reviewed tracked-seed promotion makes that ordinary gate green again,
the recovery gate becomes archival evidence rather than routine execution.
