<!-- docs:meta
topic_id: repo.docs.audit.gate-workflow-reference-asymmetry-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.gate-workflow-reference-asymmetry-2026-08-19
-->

# The gate/workflow reference check runs in one direction only

## Answer

`scripts/dev/check_workflow_script_refs.sh` enforces **workflow → script**: every
`scripts/...` path a workflow names must exist and be executable. The reverse —
**script → workflow** — is enforced nowhere.

A gate script can be written, committed, reviewed and merged without any workflow
naming it. Nothing breaks. That is how the unnamed set reached its present size:
**473 of 590** gate scripts are named by no workflow.

## Why it went unnoticed for so long

The existing check protects the **workflow author** from a broken reference. It
places no obligation on the **gate author** to be reachable. Under
`SOUNIO-EFFORT-LOCATION`, the effort sits on whoever later reads the workflow
directory and wonders what runs — which is to say, on nobody, until an audit.

This is the same shape as every defect measured on 2026-08-19: not a decision
anyone made, but an obligation nobody carried.

## What is measured, and what is not

**Direct invocation**, not coverage. A script no workflow names may still be run
by a parent that a workflow does name — `SOUNIO-EFFORT-LOCATION` measured exactly
this difference and found **45** such scripts. So a rise in this number means
*one more gate that no workflow names*, not *one more gate that never runs*.

The distinction is stated in the gate's own output and artefact because omitting
it is the error corrected in `#1989`: an invocation count written as a coverage
claim.

## Instrument

`scripts/ci/gate_workflow_reference_ratchet.sh`, wired into the `Contracts` job
immediately after the check whose blind side it covers.

| control | expectation |
|---|---|
| positive — `concept_status_gate.sh`, known to be named by a workflow | must not appear as unnamed |
| negative 1 — the named-reference list | must be non-empty (an empty list makes every gate look unnamed) |
| negative 2 — the gate list | must be non-empty |
| negative 3 — a path written `./scripts/...` in a workflow | must still count as named |

The gate refuses to emit a number if its own controls fail.

## Ratchet

Frozen at **474**.

> **First firing, 2026-08-19, hours after the gate was written.** The frozen
> count was 473. `#1987` merged carrying `scripts/ci/soir_roundtrip_gate.sh`,
> which no workflow names, and the ratchet refused at 474. It caught a real
> regression on the day it was born, in a PR written by a different lane, before
> anyone had a chance to notice by hand. The count is raised to 474 and the
> tolerance is recorded in `gate_workflow_reference.frozen` with its file, its
> origin and its date — naming another lane's gate in a workflow is that lane's
> decision, not this PR's. Adding a gate that no workflow names now fails; naming an
existing one passes and lowers the frozen count. Refusing all 473 today would
refuse the repository. Refusing the 474th costs nothing.

## Claims forbidden

- That 473 gates never run. The number measures naming, not execution.
- That this gate makes CI coverage complete. It stops one number from growing.
