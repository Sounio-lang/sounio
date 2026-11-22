<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-11-02-plan-m7-default-cutover-design
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-11-02-plan-m7-default-cutover-design
-->

# Plan M7: default-on cutover to the device paged path — design

## Why

M5 wired the device path behind opt-in pagedev; M6 made the
kernel actually correct (per-head scratch). Both specs defer
default-on cutover. The fallback contract already bounds the
risk: any device anomaly serves host math for that turn.

## Current

Example defaults pagedev=false with 5th CLI arg opt-in.
llm_serve takes an explicit bool; no other caller exists.

## M7 changes

1. Example default true; usage text updated. No signature
   changes anywhere; flag stays as opt-out.
2. No test changes: dual legs already cover both paths.

## Verification

CPU 4/4; RTX4000 4/4 with CUDA context and proof; live serve
default vs explicit 0 vs explicit 1 byte-identical.

Out of scope: flag removal, multi-GPU, sampler/HTTP changes.
