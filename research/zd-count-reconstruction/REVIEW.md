# Review adjudication

One bounded offload fan-out was invoked for NOTE.md and BIBLIOGRAPHY.md through
/workspace/sounio/bin/llm-offload, as required by repository policy.
Providers requested: xai (grok-4.6), qwen, mistral.
Timeout configured: 120 seconds. The remote job ran in tmux.
Qwen and Mistral returned text. Grok returned an empty JSON file and no
review; this is an unavailable leg, never a pass.

The reviewers inspected the draft before the final explanatory clarifications.
No second model round was needed to change mathematical statements; none changed.
All findings are adjudicated below. LLM opinions do not establish priority.

| Finding | Decision and evidence |
|---|---|
| Qwen: the n=4 reset has E=840 rather than 168. | Rejected. The recurrence table describes depth d+1 from predecessor q_d. At current d=1 the predecessor is q_0=3, so m=3·7=21 and E=168. q_1=7 belongs to a step producing depth 2. The direct native graph and CountChecks.lean both verify E=168. Added an explicit predecessor-index sentence. |
| Qwen: 3E+τ is inconsistent with 2aE+bτ at a=3,b=2. | Rejected. The draft explicitly states that the formal scalar is twice the displayed scalar. Multiplying a scalar by two preserves fibres. Added its explicit value 2,013,552. |
| Qwen: divisibility of inverse steps is not justified. | Expository clarification accepted. Valid forward C states have (4m,8p); valid T states give 4m after subtracting 3q and 8p after subtracting 6m. Added these equations and the re-encoding argument. Tests also check rejection outside the image. |
| Mistral: hub triangles number four per edge rather than two. | Four are supported but exactly two are positive. Their sign is (−1)^(α+β)M(r,s), so two of four spin choices qualify. Added the explicit sign calculation. The coefficient 6=4+2 remains unchanged. |
| Mistral: the hub/same-index negative triangle and unequal sharp edge counts are omitted. | Rejected as omission claims: both facts were already explicit in the reviewed draft. They remain explicit. |
| Mistral: C(e_W) requires e_a e_{a⊕W}=0, and a=W is included. | Rejected. Zhilina Notation 3.13 requires the coordinate product to equal ±e_W, not zero. Corollary 3.21(3) excludes a second coordinate ±e_0 in the principal case, hence excludes a=W. Added the exact exceptions to the note. |
| Mistral: the draft claims a new recursive classification. | Rejected as a reading of the draft: that phrase appears as a claim explicitly being discarded. The narrow novelty wording is retained. |
| Reviewers favour weighted compression as the novelty candidate. | Recorded as an opinion only. The bibliographic assessment rests on the inspected primary sources and remains bounded. |

No valid counterexample to the mathematical statements was produced.
The arithmetic and implementation controls pass independently of the reviews.
