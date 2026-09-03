<!-- docs:meta
topic_id: repo.docs.internal.coordination.ci-watch-contract
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.coordination.ci-watch-contract
-->

# Asynchronous CI Watch Contract

No human or principal agent waits for CI. A low-cost, event-driven watcher
observes completed CI runs and interrupts creative work only when new evidence
exists.

The watcher emits one of four public receipts:

- `GREEN`: selected checks passed and repository policy may permit merge.
- `FIXABLE`: a code or contract gate failed with evidence an owning lane can
  reproduce.
- `BLOCKED`: the run lacks a trustworthy code verdict because of runner,
  credential, quota, or infrastructure failure.
- `TIMEOUT`: a selected job exceeded its declared execution window.

Cancelled superseded runs are silent. The current CI concurrency policy owns
cancellation; the watcher records no obsolete verdict.

The watcher may read Actions metadata and logs, classify operational evidence,
and create or update one receipt comment on the associated pull request. It may
not write repository contents, merge a pull request, change compiler semantics,
repair scientific code, or promote a scientific claim. `GITHUB_TOKEN`
permissions are restricted accordingly.

When a receipt is not `GREEN`, a capable agent receives the failed jobs, a
bounded evidence excerpt, and the next operational action. That agent remains
responsible for reproducing the focused gate and deciding whether any patch is
semantically valid.
