<!-- docs:meta
topic_id: repo.docs.research.llm-collaboration-manifest
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.llm-collaboration-manifest
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# LLM Collaboration Manifest — Tapestry Project
## Octonion Conversational State Space Model (O-SSM-C)

---

## Campaign Overview

This is a **multi-agent LLM collaboration** to design and implement a conversational engine based on non-associative octonion algebra. We are distributing specialized briefs to different LLM systems, each focused on their comparative advantage.

---

## Specialist Tracks

| Track | Target LLM | Focus | Brief File |
|-------|-----------|-------|------------|
| **General** | All systems | Overview, open questions, references | `llm_collaboration_brief.md` |
| **Architecture & Optimization** | GPT-4/5, o1, o3 | Bidirectional design, curriculum training, backprop, multi-head | `llm_collaboration_brief_gpt.md` |
| **Scale & Systems** | Grok, DeepSeek-R1 | Flop analysis, partial parallelism, GPU kernels, deployment | `llm_collaboration_brief_grok.md` |
| **Formal Verification** | Claude, Gemini-2.5-Pro | Theorem sketches, safety bounds, interpretability, personality subspaces | `llm_collaboration_brief_claude.md` |
| **Internal Codex seed** | Codex | First-pass architecture + safety synthesis | `llm_collaboration_codex_response.md` |

---

## How to use these briefs

1. **Copy the relevant brief** into your LLM interface of choice
2. **Add context:** Mention that you are part of a multi-agent collaboration on the Tapestry project
3. **Request specific depth:** Ask for code sketches, proof sketches, or numerical analysis as needed
4. **Collect responses** in a shared document or thread

When distributing the briefs, keep the evidence model explicit:
- repo-backed facts come from local files such as `docs/papers/paper_a_ossm.tex`, executable `.sio` examples, Lean formalizations, and GPU test lanes
- conversational architecture claims are open research questions
- absolute novelty claims should be phrased as working research hypotheses unless separately validated

---

## Response Collection Template

When you receive a response from an external LLM, record it using this template:

```markdown
## Response from [LLM name] — [Date]
### Track: [Architecture/Scale/Formal]
### Questions addressed: [1, 3, 7]

#### Key insights:
- [Bullet points]

#### Concrete suggestions:
- [Actionable items]

#### Risks identified:
- [Potential problems]

#### Verdict: [Adopt / Adapt / Reject / Needs validation]
```

---

## Synthesis Plan

After collecting responses:
1. **Week 1:** Compile all responses into a single technical document
2. **Week 2:** Resolve conflicts between recommendations (e.g., if GPT recommends full independence and Claude recommends G₂ constraints)
3. **Week 3:** Produce unified architecture document
4. **Week 4:** Begin implementation in Sounio

---

## Repository Context

All briefs reference files in the Sounio repository at `/workspace/sounio`. Key locations:

- `stdlib/algebra/octonion.sio` — Verified octonion arithmetic
- `stdlib/algebra/sedenion.sio` — Sedenion extension with zero-divisor enumeration
- `examples/ossm_associator_attention.sio` — First associator attention mechanism
- `examples/cognitive_ossm/cognitive_ossm.sio` — Cognitive state regimes
- `self-hosted/gpu/kernels/ossm_forward.sio` — PTX forward kernel
- `tests/gpu/test_ossm_backward.sio` — Backward GPU validation lane
- `formal/OctonionAlgebra.lean` — Lean 4 formalization
- `docs/papers/preregistrations/2026-04-21_ossm_168_depression.md` — Pre-registered scientific protocol
- `docs/research/llm_collaboration_codex_response.md` — Internal first-pass synthesis from Codex

---

## Contact & Attribution

This project is led by Demetrios C. Agourakis. Contributors (human and synthetic) will be credited in repository history. The goal is not publication but **functional innovation**.

**Status:** We have a compiler, an active O-SSM research line, a GPU forward-kernel path, and a 15-benchmark paper draft. Conversational octonion modeling is still a blank slate.

**Next milestone:** Unified architecture document after multi-agent synthesis.
