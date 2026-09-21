# Loom Transactional Custody Transfer

> **Status**: Garden seed | **Last validated**: 2026-08-27 | **Source**: live Fable lane and truthful-fleet observation

## Butterfly

> o fable ta vivo ainda? to vendo um monte de lanes sem fazer porra nenhuma

Fable exposed two different failures at once. Its Claude process and tmux pane
still existed, but its coordination heartbeat and delivery endpoint had expired.
The legacy catalog continued to call the lease active. Moving that process to a
new supervisor by editing a catalog field would create a second lie: desired
custody would change without proving that execution authority moved.

## Core Idea

Custody migration is a transaction over authority, not a restart command.

```text
source observed and rollback-capable
-> target sealed and identity-bound
-> source quiesced
-> target started
-> target presence + endpoint + session proved
-> catalog committed atomically
-> transfer complete
```

Before catalog commit, the source remains authoritative. If source and target
are ever active together, the target must be stopped. If the target cannot be
proved after the source is quiesced, the source must be restored. The catalog
may change only after the target is positively proved.

## Authority Invariants

1. Exactly one catalog authority is committed: `agentd XOR loom`.
2. A transfer begins only from a positively verified active source.
3. The staged target binds the source session identity and sealed prompt digest.
4. Source quiescence is a positive receipt, not an absent heartbeat.
5. Target readiness requires matching session identity, live process presence,
   and an active authenticated endpoint.
6. Source and target active together means `ABORT_TARGET`, never success.
7. A target deadline after source quiescence means `START_SOURCE_ROLLBACK`.
8. Catalog commit precedes neither target proof nor source quiescence.
9. A missing policy, missing rollback path, malformed phase, identity drift,
   or incomplete observation fails closed.
10. Provider terminal text cannot confirm custody.

## Transaction Phases

| Phase | Meaning | Allowed next action |
| --- | --- | --- |
| `PREPARE` | Source is still authoritative; target descriptor is sealed. | `QUIESCE_SOURCE` or refusal. |
| `SOURCE_QUIESCING` | Stop requested; source authority has not yet been surrendered. | `START_TARGET`, `WAIT`, or abort. |
| `TARGET_STARTING` | Source has a positive quiescence receipt; target is provisional. | `COMMIT_TARGET`, `WAIT`, rollback, or abort target. |
| `COMMITTING` | Target proof is complete and catalog mutation is authorized. | `COMMIT_TARGET` only. |
| `COMMITTED` | Catalog names Loom and target proof still matches. | `COMPLETE`. |
| `ROLLBACK` | Target is not authoritative; source restoration is required. | `ABORT_TARGET`, `START_SOURCE_ROLLBACK`, or `ROLLED_BACK`. |

## Decision Algebra

The first Sounio executable returns exactly one decision:

| Code | Decision |
| --- | --- |
| `1` | `QUIESCE_SOURCE` |
| `2` | `START_TARGET` |
| `3` | `COMMIT_TARGET` |
| `4` | `COMPLETE` |
| `5` | `START_SOURCE_ROLLBACK` |
| `6` | `ABORT_TARGET` |
| `7` | `WAIT` |
| `8` | `ABORT_TRANSFER` |
| `9` | `ROLLED_BACK` |
| `101+` | typed refusal |

`WAIT` is bounded. It does not authorize catalog mutation, process launch, or
process termination. Deadline handling is part of the next input frame.

## Positive Proof Frame

The decision consumes positive observations rather than inferred absence:

- policy is present and successfully evaluated;
- phase is known;
- source catalog says agentd or target catalog says Loom, exclusively;
- target descriptor exists and its digest is sealed;
- source process identity is verified;
- source quiescence receipt exists;
- target process exists under Loom Guardian custody;
- target presence is live;
- target endpoint is active and authenticated;
- target session matches the source-bound session;
- target prompt digest matches the staged descriptor;
- rollback path is available;
- bounded deadline state is known.

Silence, a missing record, a stale endpoint, or an expired heartbeat is not a
quiescence receipt.

## Claude Persistence Boundary

The first operational consumer is Claude Code because the live Fable lane is a
Claude session. The provider adapter must use the provider-native interactive
contract:

```text
new:    claude --session-id <uuid> --setting-sources user,local
resume: claude --resume <uuid> --setting-sources user,local
```

Initial input travels through the authenticated Loom wake lease. The raw prompt
does not appear in the process argv. `--fork-session` is forbidden because it
would break the source-bound identity. `--continue` is forbidden because it
selects by directory history rather than the frozen session identity.

The provider keeps its own credentials and subscription authority. Loom never
copies or interprets the token store.

## Semantic Lane Declaration

```text
Semantic-Lane-ID: loom-transactional-custody-transfer-20260827
Owner: codex-1/loom-transactional-custody-transfer-20260827
Concept-IDs: SOUNIO-LOOM-MULTIPLEXER
Intent-Preserved: migrate a live provider from legacy agentd to Loom without dual authority, identity drift, or silent loss
Transformation: replace operator restart with a proof-carrying prepare/quiesce/prove/commit transaction
Types-Changed: add custody-transfer phase, positive proof frame, and typed decision
Effects-Changed: process stop/start and catalog commit become decision-gated effects
IR-Changed: none
Claims-Introduced: catalog authority moves only after source quiescence and target identity/presence/endpoint proof
Claims-Forbidden: editing custody transfers authority; target process existence proves migration; silence proves source quiescence; two active authorities are harmless
Assumptions: the legacy launcher can restore its own stopped slot; provider-native session resume preserves the conversation
Write-Set: Sounio custody-transfer semantics, frozen manifest, OCaml parity, provider adapter, fleet transfer CLI, tests, receipt
Read-Set: fleet catalog, legacy status, Loom session descriptor, coordination presence and endpoint snapshot
Positive-Witness: a sealed Claude resume transfer reaches COMPLETE with exactly one authority
Negative-Witness: deliberate target start failure restores the legacy source and leaves the catalog unchanged
Sabotage-Control: removing only the dual-authority rule permits COMMIT_TARGET while source and target are active
Acceptance-Gate: Sounio exhaustive decision gate, frozen semantics gate, OCaml parity gate, transactional fixture with crash and rollback controls
Integration-Target: Loom fleet catalog, TUI/GUI intervention surface, native provider custody
Authoritative-Only-If: Sounio expected cases pass, semantics are frozen by hash, and OCaml parity is opened only against that hash
```

The registry entry for `SOUNIO-LOOM-MULTIPLEXER` remains serialized behind the
current registry owner. This lane has requested a coordinated append and will
not edit `docs/internal/concepts/registry.tsv` concurrently.

## Evidence State

| Layer | Status |
| --- | --- |
| `GARDEN` | Captured by this seed. |
| `SOUNIO_EXECUTABLE` | Not yet. |
| `SEMANTICS_FROZEN` | Not yet. |
| `PARITY_OPEN` | No. |
| `CLAIM_READY` | No. |

## What This Is Not

- Not permission to migrate Fable before the fixture and rollback controls pass.
- Not permission to kill an unresponsive lane.
- Not a claim of zero downtime.
- Not hostile same-UID process isolation.
- Not host-loss or storage-loss continuity.
- Not a semantic role for Claude, Codex, Kimi, Python, Rust, shell, or OCaml.
- Not a replacement of the provider's native credential or session store.

## Next Executable Bridge

Implement the decision algebra and expected cases in Sounio. Compile and run it
with the shipped Sounio compiler, then commit that executable source as the
immediate child of this Garden commit. Only after that result exists may the
semantics be frozen and OCaml parity begin.
