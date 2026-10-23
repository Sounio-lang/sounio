<!-- docs:meta
topic_id: repo.docs.research.sco-math-review-2026-10-12
authority: historical
audience: researchers
last_validated: 2026-10-12
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sco-math-review-2026-10-12
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# SCO mathematical review record — 2026-10-12

- Reviewed pin: `dfe4db44da3ac96b3c736c4d3464bd71d403f139`.
- Literal source: `docs/research/scientific_compilation_obligation_2026-08-01.md`; SHA256 `d7798b3be3228d8e49bff51a19a760e392f09c543c1cf7d75678203fa016b3c4`.
- Scope: logical review of definitions, observable comparison, typed refinement, source entitlements and controlled ablation. No execution, numerical recalculation or Lean proof verification.
- Prompt SHA256: `4e06ef9c66359cedd63b300eebafe163277d60b497e26d3e4deba160e193f364`.
- External review A: DevPass, actual `aws-bedrock/claude-opus-4-7:global`, finish `stop`, nonempty answer; provider-reported 6151 input / 2598 output tokens and cost `0.095705`. Raw artifact `outputs/PR2628-M1-Claude-Actual-Review-20261012.json`, SHA256 `bdd776621a54a3f8e5e546552233cdc3c51be8750c9fffd88787fc0bffa0519b`.
- External review B: DevPass, actual `aws-bedrock/grok-4-7:global`, finish `stop`, nonempty answer; provider-reported 3804 input / 619 output tokens (52 reasoning), cost `0.011322`. Raw artifact `outputs/PR2628-M1-Grok-Actual-Review-20261012.json`, SHA256 `d034d639f10101e3dd9661ebcb6e5b738818253a94bb190cd5c0e5643318f5e1`.
- Independence boundary: distinct Claude/Anthropic and Grok/xAI model families, shared DevPass gateway and AWS-Bedrock-labelled routes. Do not claim separate authenticated gateways.
- Additional local review: Darwin `coder-next`, raw artifact `outputs/PR2628-M1-Darwin-Actual-Review-20261012.json`. Its unqualified transitivity claim was rejected because it omitted the permitted tolerance convention.
- Objections: tolerance-based value acceptance does not guarantee a preorder; typed refinement composition and independent source entitlements need specification; named strategy ablation does not discharge semantic non-vacuity. Presence critique is qualified by the existing same-shape domain. The suggested conjunction of forbidden rewrite and changed output is stronger than the current alternative and requires author judgment.
- Disposition at review completion: reviews recorded; definition remains a historical proposal; no preservation theorem or compiler certificate. Root/author disposition and any later V3 or semantic witness validation must be logged separately against their actual pin. No automatic thread resolution or acceptance is implied by obtaining two responses.
- Judgment receipt: `outputs/PR2628-M1-Independent-Review-Receipt-20261012.md`.


## Accepted disposition and provenance boundary

The definition now uses a source-anchored acceptance relation and explicitly limits composition of ULP-tolerance comparisons. This addresses the documented mathematical objection without asserting a Lean proof or a compiler certificate. The two external responses use distinct model families through the same DevPass gateway; backend weights or independent transport infrastructure were not verified.

This is a retrospective reconciliation record. It does not claim that the reviews occurred before the original PR creation. Later checker/runtime changes have separate source pins and execution receipts; the mathematical reviewers did not certify those changes. Scientific non-vacuity remains open after a bounded numerical probe produced equal enforce and ablation results.

The generated `.claude/llm_offload_log.md` is retained locally, untracked, under the repository snapshot policy. This governed document is the durable review record in this PR. No generated-snapshot guard was weakened.
