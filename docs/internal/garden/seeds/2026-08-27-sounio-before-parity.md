<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-27-sounio-before-parity
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-27-sounio-before-parity
-->

# Sounio Before Parity

> **Status**: Garden seed | **Source**: founder direction on 2026-08-27

## Butterfly

> Continue, but incorporate this as an acceptance criterion.

The system must not discover its own meaning by looking backward from a proof,
a benchmark, a foreign implementation, or an LLM review. Sounio must speak the
first executable proposition; every later language must identify which Sounio
artifact it is comparing and remain inside its declared role.

## Core Idea

Loom already separates runtime custody, semantic journals, projections, and
review. The missing boundary is language authority. The founder's ordering is:

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

Sounio is `SEMANTIC_AUTHORITY`. Lean 4, Koka, C++, and Haskell may prove,
compare, or measure only after a hash identifies the frozen Sounio semantics.
External LLMs are `REVIEW_ONLY`. Python and Rust are forbidden. Renaming an
oracle or hiding it behind another launcher must not change its authority.

The enforcement pressure is pre-action rather than retrospective: a forbidden
process must not spawn, a forbidden write must not land, and a forbidden commit
must not become authoritative. Missing policy, policy error, and policy timeout
must all deny and leave a reasoned decision receipt.

## Connections

- [`FOUNDER_INTENT.md`](../../../../FOUNDER_INTENT.md) defines the founder's
  evidence progression and protects semantics from convenient replacement.
- [`loom-multiplexer.contract`](../../concepts/loom-multiplexer.contract)
  defines the current Loom authority boundaries and migration surface.
- [`loom_language_authority.sio`](../../../../stdlib/coordination/loom_language_authority.sio)
  is the intended first executable bridge from this seed.
- [`sounio_coord_agent_hook.py`](../../../../scripts/dev/sounio_coord_agent_hook.py)
  is the current Python hook bridge that prevents any honest no-Python claim.

## Evidence State

| Layer | Status |
| --- | --- |
| `Garden` | Captured: continue the existing Loom, but make Sounio-first language authority an acceptance criterion. |
| `Hypothesis` | One Sounio-owned state machine can prevent foreign parity and review artifacts from retrospectively defining semantics. |
| `Executable` | No at seed capture. The next artifact must be Sounio and must own its expected ALLOW/DENY results. |
| `Claim-ready` | No. No native guardian or full pre-action enforcement exists yet. |

## What This Is Not

- Not permission to build a second multiplexer or coordination kernel.
- Not a claim that the current OCaml kernel or Python hooks already comply.
- Not permission to replace Python cosmetically with shell, Node, or another
  foreign oracle.
- Not evidence that parity, formal proof, or LLM agreement establishes semantic
  authority.
- Not a production-readiness or novelty claim.

## Next Executable Bridge

Implement the ordering, language roles, receipt completeness, founder waiver,
and fail-closed decisions in Sounio. Put the negative expected results in that
same Sounio artifact. Freeze its source and executable hashes before a
transitional C++ guardian is allowed to reproduce the decisions.
