# Truthful Lane Health

> **Status**: Garden seed | **Last validated**: 2026-08-27 | **Source**: live founder observation of the running fleet

## Butterfly

> o fable ta vivo ainda? to vendo um monte de lanes sem fazer porra nenhuma

A process can exist while a lane is absent from the work it appears to own.
Conversely, a quiet lane is not proved idle merely because no output was seen.

## Core Idea

Loom must stop collapsing different operational facts into one green light.
These are independent observations:

1. a claim record exists and is leased;
2. the harness process identity is verified;
3. a delivery endpoint is verified;
4. an obligation is held;
5. durable output or evidence advanced;
6. the harness positively reports that it is ready for work.

No one fact implies the others. In particular:

```text
active claim != verified process
verified process != reachable endpoint
reachable endpoint != active obligation
active obligation != observed progress
no observed progress != proved idle
pane exists != lane is working
```

## Affirmative Absence

An absence claim is valid only as a positive triple:

```text
expected signal
+ authorized observation
+ completed bounded window
= affirmative absence
```

Therefore `IDLE` requires all of the following positive evidence:

- the process and endpoint are verified;
- the obligation census is complete and reports no active obligation;
- the progress window is complete and reports no progress;
- the harness positively reports ready-for-work;
- the observation authority and freshness checks pass.

Missing records, expired heartbeats, stale endpoints, partial scans, timeouts,
and observer errors produce `UNKNOWN`, `DISCONNECTED`, or `UNRESPONSIVE`.
They must never be laundered into `IDLE` or `DEAD`.

## Spectral State

The first executable bridge will classify one lane into exactly one derived
intervention state:

| State | Meaning |
| --- | --- |
| `WORKING` | Verified process plus authorized evidence of an active obligation or recent progress. |
| `IDLE` | Affirmative absence of obligation and progress, plus a positive ready signal. |
| `BLOCKED` | An active obligation and an explicit blocker, observed without progress over a completed window. |
| `DISCONNECTED` | The lane is expected and a pane or harness exists, but its verified delivery path is absent or stale. |
| `UNRESPONSIVE` | The expected process exists but failed its bounded response or heartbeat contract. |
| `ORPHANED` | Durable records remain after the owning process is positively observed missing. |
| `DEAD` | The expected lane, process, endpoint, and recoverable custody are all affirmatively absent. |
| `CONFLICTED` | Mutually incompatible positive observations are present. |
| `UNKNOWN` | Evidence is incomplete, stale, unauthorized, or otherwise insufficient. |

The state is not a performance score and does not rank providers. It is an
epistemic classification of current operational evidence.

## Semantic Lane Declaration

```text
Semantic-Lane-ID: loom-truthful-lane-health-20260827
Owner: codex-1/loom-mainline-core-20260827
Concept-IDs: SOUNIO-LOOM-MULTIPLEXER
Intent-Preserved: fleet UI and reconciliation must distinguish existence, reachability, custody, obligation, and progress
Transformation: replace claim-first green status with an evidence-complete Sounio classification
Types-Changed: add a typed lane-health decision and positive observation frame
Effects-Changed: none
IR-Changed: none
Claims-Introduced: a lane state is supportable only when its required positive observations are present
Claims-Forbidden: silence proves idle; an active claim proves work; a pane proves delivery; a missing heartbeat proves death
Assumptions: observation authority and bounded-window receipts are independently verifiable
Write-Set: stdlib/coordination/loom_lane_health.sio, tools/loom lane-health runtime and operational realization
Read-Set: coordination cockpit snapshot, Loom session descriptors, obligation views, durable output metadata
Positive-Witness: Fable-like stale endpoint plus expired presence does not classify as working or idle
Negative-Witness: removing only the affirmative-absence rule admits a silent incomplete lane as idle
Acceptance-Gate: scripts/ci/sounio_loom_lane_health_selftest.sh and frozen-semantics parity gate
Integration-Target: Loom fleet API, TUI, GUI, and non-destructive reconciler
Authoritative-Only-If: Sounio executable expected cases pass, semantics are frozen by hash, and OCaml parity is exhaustive over the bounded input domain
```

## Evidence State

| Layer | Status |
| --- | --- |
| `GARDEN` | Captured by this seed. |
| `SOUNIO_EXECUTABLE` | Not yet. |
| `SEMANTICS_FROZEN` | Not yet. |
| `PARITY_OPEN` | No. |
| `CLAIM_READY` | No. |

## What This Is Not

- Not an instruction to kill idle or unreachable processes.
- Not permission to release another lane's claim automatically.
- Not proof that terminal text reveals model cognition.
- Not a provider quality or productivity ranking.
- Not a claim that tmux is a durable authority.
- Not permission for OCaml, shell, Python, Rust, or an LLM to invent the state.

## Next Executable Bridge

Create the state machine and its expected-result suite in Sounio. Freeze the
source, entrypoint, toolchain, hardware, command, and result hashes. Only then
may the OCaml runtime implement parity and expose the result to Loom surfaces.
