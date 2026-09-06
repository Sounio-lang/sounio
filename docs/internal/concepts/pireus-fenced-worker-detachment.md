<!-- docs:meta
topic_id: repo.docs.internal.concepts.pireus-fenced-worker-detachment
authority: repo_only
audience: users
last_validated: 2026-09-06
validated_by: Codex
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.pireus-fenced-worker-detachment
-->

# Pireus fenced worker detachment

Concept-ID: `SOUNIO-PIREUS-FENCED-WORKER-DETACHMENT`
Semantic-Lane-ID: `continuity-20260906`
Owner: `codex-pireus`
Status: native controls and independent review pass; both workers detached with
FENCED postconditions. Complete canonical recovery still refuses HOST_MEMORY_FLOOR.

The Sounio continuity task exposed a recovery ordering failure: the existing
frozen arbiter tries to prove worker cgroups empty while kubelet recreates
worker processes. Two canonical recover attempts failed at that boundary.
This extension has one action: stop recreation by removing the frozen
Slurm selector from both UID-pinned Spark nodes, already fenced, after an
independent native Sounio admission. It does not claim predecessor admission.

The 19-field observation contract, exact predicates, refusal behavior,
source/executable lock, raw evidence and per-node CAS effects are specified in
`tools/pireus/continuity/runtime/README.md`. The semantic authority is
`recovery_detach.sio`; `recovery_detach.py` is observation/effect transport.
Missing facts, unknown booleans, stale or foreign leases, active allocations,
non-fenced hosts and free memory below 32768 MiB refuse the action.

No GPU grant, device barrier change, watchdog shutdown, protected service
mutation, threshold reduction, lease transition or resume is in this scope.
Pair-wide atomicity is not claimed. Partial detachment is retained as such;
the unchanged frozen arbiter must subsequently prove complete recovery and
Slurm ownership. A successful native control test is not live recovery.
