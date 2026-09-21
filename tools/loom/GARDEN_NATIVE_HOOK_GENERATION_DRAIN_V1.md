# Garden: Native Hook Generation Drain

Concept-ID: `SOUNIO-LOOM-NATIVE-HOOK-CUTOVER`

Status: `GARDEN`

## Butterfly

> A hook configuration can change while the process that read it remains old.

The first live native-hook promotion proved that configuration state and
execution state are different facts. Existing provider processes retained a
cached Python hook command after the shared runtime had become bridge-free.
The new runtime was correct in isolation, yet activating it underneath those
processes removed an executable they still possessed authority to call.

The missing object is a **hook generation**: the conjunction of provider
process identity, loaded hook command, runtime capsule, and admission policy.
Changing a file does not mutate that generation retrospectively.

## Hypothesis

A bridge-free runtime may become `current` without a stop-the-world cutover
only when the absence of legacy generations is established positively.
Absence is not an empty query result. It is the conjunction of:

1. a complete, freshness-bounded inventory of live provider processes;
2. an exact classification receipt for every member of that inventory; and
3. zero members classified as legacy, unknown, drifted, or unresponsive.

This is the **affirmative absence triple**. If any component is missing, the
cutover remains in `DRAINING` and the final runtime symlink cannot move.

## Candidate State Machine

```text
GARDEN
-> CANDIDATE_ATTESTED
-> NATIVE_ENTRY_OPEN
-> LEGACY_DRAINING
-> AFFIRMATIVE_ABSENCE
-> CUTOVER_READY
-> BRIDGE_FREE_CURRENT
```

`NATIVE_ENTRY_OPEN` means new provider processes enter an exact candidate
generation while cached legacy processes continue against their old runtime.
It does not mean the candidate is current. `CUTOVER_READY` requires the
affirmative absence triple plus four-provider canaries bound to the same
candidate capsule.

## Semantic Pressure

- A provider process is evidence-bearing state, not a disposable shell around
  a configuration file.
- A runtime symlink is a generation switch, not merely a filesystem update.
- A cached command remains an outstanding capability until its process exits
  or proves it loaded the native generation.
- A stale heartbeat cannot count as either live or absent.
- A native canary cannot discharge a different legacy process.
- Rollback must restore both the runtime generation and the configuration
  generation, or it is not a rollback of the same system.

## First Executable

Sounio action 9046 will consume a bounded observation containing:

- parent action 9045 freeze binding;
- old and candidate runtime hashes;
- candidate and final configuration hashes;
- inventory epoch and freshness binding;
- total, classified, native, legacy, unknown, and unresponsive counts;
- exact process-generation and hook-capability bindings;
- four-provider candidate canary mask;
- rollback readiness; and
- causal sabotage count.

It will admit `DRAINING` while known legacy generations remain. It will admit
`CUTOVER_READY` only when the affirmative absence triple holds. Missing
inventory, incomplete classification, stale observations, unknown members,
runtime drift, or a synthetic zero count must refuse.

## Sabotage Controls

1. Report `legacy=0` while `classified < total`.
2. Report a complete classification from a stale inventory epoch.
3. Replace one native capability receipt with a receipt from another process
   generation.
4. Change the candidate runtime after the provider canaries were recorded.
5. Omit one provider from the canary mask.
6. Attempt the final flip without a tested rollback pair.

Each sabotage must refuse because of action 9046, while its paired control is
admitted. Python, Rust, LLM output, and provider exit codes cannot define the
expected result.

## Evidence Boundary

This seed does not claim a working live drain, priority over existing blue/green
deployment systems, or a completed no-Python fleet. It identifies the semantic
object that the first live experiment made visible and defines the smallest
executable witness that can falsify the proposed cutover rule.
