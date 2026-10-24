<!-- docs:meta
topic_id: repo.docs.website.scientist-intake-repair-2026-10-18
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.website.scientist-intake-repair-2026-10-18
-->

# Repairing the observed scientist-intake failure

Base: `c4f1d3ce73987de9e2694af393711180f6a57fb1`.

## Failure and root cause

The previous actual `darwin/coder-next` probes did not conform: one invented domain-source inspection, while the strengthened repeat still expanded the scientist's question and introduced an unsolicited domain-range suggestion. Merely documenting that limitation did not repair the public workflow.

The one-shot request mixed scientific intake with a recipe goal, domain limitations, historical execution records and complete language examples. These were useful implementation references but inappropriate priming before the scientist's actual inputs were confirmed. There was also no interface path for rejecting an LLM's altered or invented intake before preparing implementation instructions.

Both original, unedited failed replies are now retained as explicitly rejected fixtures, not numerical references or accepted scientific advice. The new gate rejects both. The original research question is preserved rather than replaced by the recipe's question.

## Implemented boundary

1. Initial instructions contain a closed `SOUNIO_INTAKE_V1` receipt, the supplied brief and protocol instructions only. They contain no recipe goal, domain-source reference pack, code examples or historical execution output. Supplied field values remain quoted data, never tool instructions. Empty fields remain missing, not zero or independent.
2. The LLM returns the supplied inputs unchanged, the exact missing-field list, request-local `source_access: "not_verified"` and `execution: "not_executed"` statuses. This is factual correspondence, not a claim that the study was understood, validated or executed.
3. The scientist pastes the reply into the page. A local parser rejects additional prose, duplicate or unknown rows, changed questions or inputs, invented source access, execution claims and malformed or oversized replies. Whitespace in JSON values, CRLF and one complete text fence are supported without accepting extra content.
4. A matching reply still cannot prepare implementation instructions until the scientist explicitly confirms the current brief. Changing any scientific field, recipe, reply or confirmation revokes acceptance and clears stale implementation output. Each implementation action checks the reply and current state again, even if a button is forcibly enabled.
5. Only the separately confirmed implementation-planning request contains the canonical language pack and pinned domain references. Missing scientific inputs remain missing; the implementation contract still requires clarification and approval before material model choices, real source inspection, authorized tools and actual execution evidence.

Scientists write no protocol or program themselves: they describe the study, copy an initial request, paste its response and confirm the scientific brief. The interface checks correspondence only. It does not authenticate the LLM, prove scientific truth, authorize execution or attest to source access. Free-form scientific reasoning and calculations still require scientific review and executable evidence; twenty-five intake replies are not twenty-five scientific solutions.

## Regression evidence

`website/scripts/check-scientist-behavior.mjs` runs as part of the existing scientist quality gate. Its 465 controls cover malformed/invented replies, stale inputs across all six fields, changed language/recipe, duplicate rows, explicit confirmation, quoted instruction injection and worst-case supported field encoding. Both actual old failing responses are rejection controls.

`website/scripts/fixtures/scientist-intake/observed-responses.json` retains twenty-five actual responses with prompt and reply SHA-256 bindings:

- xAI returned `grok-4.3` for twenty-one cases: all six recipes in English, Portuguese and Greek, plus a quoted-instruction case in each language.
- The independent Kimi gateway route completed the three quoted-instruction cases. Response metadata identified `runpod/kimi-k3` for English and `moonshot/kimi-k3` for Portuguese/Greek; those actual identities are preserved instead of assuming the older AWS routing identity.
- The previously failing `darwin/coder-next` alias completed the reproduced Portuguese missing-data intake faithfully. Its concrete model identity and completion metadata were not exposed by the MCP tool and are not fabricated.
- `darwin/qwen38-27b` returned a tool error; it is recorded as unavailable and not counted as a passing response. The gateway tests supplied the independent completed coverage.

All twenty-five completed replies pass the exact local correspondence checker. Every completed gateway response has `finish_reason: stop`. CI reconstructs the current generated prompt, compares its digest to each recorded prompt and revalidates every response; a changed prompt cannot silently reuse a stale passing receipt. These are historical, source-bound behavioral observations, not a promise about every future model response.

Actual isolated-browser checks passed in all three languages: old failed reply rejected; checkbox alone insufficient; forcibly enabling a button cannot bypass the handler; valid reply still needs explicit confirmation; edits clear output and consent; stale replies fail; recipe changes and unchecking revoke acceptance. Synthetic fixtures are identified separately from actual model responses. Greek mobile layout was checked without horizontal overflow and screenshots inspected.

The actual new `coder-next` response also passed the browser round trip. The downloaded initial request digest matches the prompt sent to that model, and the gated implementation download equals the canonical implementation builder byte for byte. Adding an unsolicited claim to that actual response revokes acceptance and clears implementation output.

`npm run check:quality` passed, including the source-bound scientist gate and Astro checks over 246 files with zero errors, warnings or hints. The documentation registry and public-claims gates are separate checks. The existing macOS asset-render path still retains pre-rendered Linux assets; no compiler, scientific workflow, clinical algorithm or numerical calculation was executed to establish these intake results.

## Independent review and unavailable fallbacks

The existing remote `bin/llm-offload` wrapper was invoked for precommit review. xAI's actual `grok-4.3` response found no actionable major/blocker in the supplied source. Raw review: `/tmp/llm-offload-yezZfv/grok-fast.json`; SHA-256 `12ab34d57ccdf0ad8c227ffd29997a8f38b3b798e374830bd7b7c0f22c648c23`.

Fallback attempts are not represented as successful reviews. The initial Kimi review timed out; focused gateway Kimi and DeepSeek responses exhausted their output budget with `finish_reason: length` and no final content, even after a bounded larger-budget retry. Direct DeepSeek rejected its configured credential. Gemini through OpenRouter rejected the request with HTTP 402 for insufficient credits. These operational failures were logged; no credential, billing or quota setting was changed and no further paid review attempts were made after the credit failure was identified. Only the xAI review completed. The independently completed Kimi **behavior probes** above do not substitute for a completed Kimi **code review**.

This nonnumerical interface repair does not claim two-vendor mathematical review or completed reviews from unavailable providers. Its release evidence consists of the owner review, completed xAI review, actual bounded behavior observations, negative controls and real browser checks, with the fallback limitation explicit.

## Remaining independent compiler blocker

The full push-triggered CI at the base commit failed separately at self-compile registry capacity, `E-USE-MOD-CAP`. It is not fixed by this website repair. [The exact blocker handoff](SCIENTIST_INTAKE_COMPILER_BLOCKER_2026-10-18.md) records issue #2966, diagnostic artifact, ownership gap and the current-source acceptance gate. Successful website, PR or merge-group checks must not be presented as full compiler health.

Protected merge, production deployment and live verification of this repair are separate publication states and must be reported at their actual source identities.
