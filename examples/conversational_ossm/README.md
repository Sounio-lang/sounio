<!-- docs:meta
topic_id: repo.examples.conversational-ossm.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.examples.conversational-ossm.readme
-->

# Conversational O-SSM

This directory now has a repo-native conversational O-SSM scaffold instead of only isolated octonion demos.

## Runtime shape

- `o_ssm_core.sio`
  - octonion carrier state `h_t`
  - non-associative update rule
  - online associator telemetry
- `o_ssm_router.sio`
  - 7-basin Fano router
  - top-line selection plus routed utterance gain
- `o_ssm_conflict.sio`
  - sedenion conflict head
  - zero-divisor proximity and freeze pressure
  - alternate-branch input proposal
- `o_ssm_memory.sio`
  - branchable memory with primary/alternate state lanes
- `agent_cli.sio`
  - end-to-end conversational control loop
  - emits `answer | clarify | split | abstain`

## Why this matters

This is the first integrated pass where the algebra is the control law of the conversational engine:

- octonions handle ordered, non-associative recurrent state
- the Fano plane routes utterances into associative basins
- sedenions detect action-null conflict geometry
- branch memory prevents forced collapse to one interpretation

## Conservative status

This is an executable example surface, not yet a canonical stdlib module and not yet a training/runtime stack.

It is designed to prove the architecture inside the main Sounio repo before any larger refactor.

## Run

```bash
./bin/souc run examples/conversational_ossm/agent_cli.sio
./bin/souc run examples/conversational_ossm/associator_telemetry.sio
./bin/souc run examples/conversational_ossm/bidirectional_ossm_v0.sio
```
