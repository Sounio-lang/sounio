# Garden Seed: Sovereign Material Change V2

Status: Garden
Date: 2026-08-31
Concept-ID: SOUNIO-LOOM-SOVEREIGN-CHANGE-KERNEL
Parent: action 9043, frozen at `tools/loom/sovereign_change_kernel.freeze.v1`

## First Phrase

The provider may describe a change, but it must not possess the filesystem
authority that makes the description true.

## Hypothesis

A model-facing CLI can operate on a useful writable surface while the actual
Git worktree and common directory are mounted read-only in its namespace. A
resident Loom kernel can prepare an isolated staging cell, bind it to one exact
tool call and material descriptor, and materialize the result only after the
Sounio authority admits the corresponding phase.

The material descriptor commits to the complete event, mutation bytes,
canonical file set, pre-image, expected post-image, Git HEAD, Git index,
kernel generation, session, peer identity, and tool-call identity. The Sounio
authority receives the descriptor commitment and the lifecycle facts. OCaml
and C may realize the boundary, but may not invent an ALLOW result.

## Required Distinction

`PostToolUse` is evidence that a provider operation completed against staging.
It is not permission to repair an unauthorized write. The original worktree
must remain unchanged until the kernel consumes the one-shot grant.

## Ordered Experiment

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PROVIDER_READ_ONLY
-> STAGED_CHANGE_CONSUMED
-> COMMIT_ADMITTED
-> CI_RECEIPT_CONSUMED
-> CLAIM_READY
```

## Falsifiers

- the provider can mutate the worktree or Git common directory directly;
- two tool calls can consume the same resident grant;
- a wrong staged image can be repaired after the grant was refused;
- a path, HEAD, index, pre-image, or ungranted file can drift unnoticed;
- OCaml, C, an LLM, Python, or Rust can manufacture an ALLOW decision;
- commit or CI accepts content not byte-identical to the consumed change;
- CI recomputes policy instead of consuming the frozen receipt.

This seed is not a claim of completion. It authorizes an action 9044 Sounio
experiment without changing the already frozen action 9043 semantics.
