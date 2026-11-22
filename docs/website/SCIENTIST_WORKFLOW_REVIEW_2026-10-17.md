<!-- docs:meta
topic_id: repo.docs.website.scientist-workflow-review-2026-10-17
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.website.scientist-workflow-review-2026-10-17
-->

# Scientist-first prompt workbench: bounded review receipt

## Scope and source identity

Base: `106f7a988d5b007472d0eead724f58ac50f92200`. The scientist describes a research question, available data, units, uncertainty, assumptions, deliverables and acceptance criteria; the connected, authorized assistant handles code and tools. This change prepares instructions, not scientific results or a hosted execution service.

Six research recipes cover hydrogen ignition, surface mechanisms, acid–base measurements, synthetic oral pharmacokinetics, synthetic sequential trials and PBPK distribution. English, Portuguese and Greek supply eighteen prepared language-context packs. The other four site locales visibly fall back to English for this surface. All recipes retain `prompt-template-not-executed`.

Domain sources are bound to the base commit by path, SHA-256 and pinned URL. The independent checker compares current bytes and prepared hashes against `git show` at that commit; changing both current bytes and stored hashes still fails. Canonical language examples retain their own baseline, historical compiler profile, controls and receipts. Those examples teach syntax; they do not validate a domain adaptation. Historical receipts are explicitly unauthenticated, not new execution evidence.

Preparation is a Node-only source-material step. Pages consume committed static context; public builds and browsers do not run Git, a compiler or an LLM to assemble prompts. The quality command explicitly fetches the named source commit if a shallow checkout lacks it, without changing the checkout. Partial clones that also omit blobs require materializing the referenced blobs; the commit-existence preparation is not a complete partial-clone repair service. Verification assumes the normal trusted Node/Git toolchain, not protection against a compromised developer PATH.

No compiler, standard-library implementation, clinical pathway, numerical algorithm or theorem was changed. Scientific engine limitations remain visible per recipe. No automatic diagnosis, prescribing, clinical validation, universal model compliance or present scientific execution is claimed.

## Validation performed

- `npm ci --prefix website`: completed. Existing dependency/audit and install-script warnings were not treated as newly fixed vulnerabilities.
- `npm run check:quality` in `website`: passed, including the scientist gate, navigation, route, locale, brand, search and highlighting contracts. Astro checked 245 files with zero errors, warnings or hints. Navigation checked 29,939 internal links.
- `node website/scripts/check-scientist-workflows.mjs --self-test`: six templates, eighteen packs, forty-five source bindings, seven pages and fifty-four mutation/boundary controls; no errors. Report scope is prompt preparation, source identity and interface only; `compiler_invoked` and `llm_execution_verified` are false.
- Running the new checker against the preceding deployed static site fails because all seven scientist routes are absent. Source pinning and field/context corruption have explicit negative controls.
- `bash scripts/dev/check_docs_registry.sh`, the existing public-claims gate and `git diff --check`: passed before publication. The metadata above remains conservatively unvalidated rather than claiming a separate registry attestation.

The existing render-assets gate on this macOS host cannot execute the Linux artifact and explicitly retains pre-rendered assets. That path is not fresh native rendering or scientific execution. The new generator and scientist checker invoke no compiler.

## Actual browser checks

The connected browser runtime reported no available browsers. Testing therefore used a fresh, isolated Playwright CLI session against the local production build, without user logins or browser profiles.

- A blank scientific question produces the localized required-input status and no prepared output. Recipe goals are placeholders, not fabricated study requests.
- A synthetic Portuguese brief produces the actual question, `INTAKE_ONLY`, ADR009, source-bound language context and the unexecuted template state.
- Editing updates the open preview; clearing a question clears stale output. Switching a recipe preserves the scientist's question.
- Copy reported success. The downloaded Markdown file exactly equals the canonical prompt builder's output for the entered synthetic brief.
- An isolated local response mutation removed the prepared contexts. Copy then produced the visible localized assembly-failure message and no stale result. The interception was removed before subsequent checks.
- The Greek page rendered all six recipes at a 390-pixel mobile viewport, with an empty question and no horizontal overflow. Portuguese desktop and Greek mobile full-page screenshots were visually inspected.

These checks exercise instruction preparation and interaction, not research calculations or a model's tool execution. No study data or patient records were used. The form has no named scientific fields, automatic model endpoint or persistence path. Manual copy/download remains a deliberate transfer; users must select an institutionally approved model and environment rather than sending confidential records to an unapproved service.

## Independent review and disposition

Review used the existing remote `bin/llm-offload` wrapper without modifying the remote compiler workspace. Routing labels were not accepted as model identity: raw response JSON reported xAI `grok-4.3` and Moonshot `aws-bedrock/kimi-k3:global`, both with `finish_reason: stop`.

Initial review artifacts: `/tmp/llm-offload-erORrg` on the review host.

| Response | SHA-256 |
|---|---|
| `grok-fast.json` | `c316f2cce16e995d058ab7296b8f059fb223257a5bc7ceff8bb0788ce06fcaab` |
| `kimi.json` | `89df43465850afef2c56d4f7a32cb5cb15b305faed5d868395416ae546330474` |

Kimi's four major findings were accepted and fixed: independently anchor pinned source bytes; surface assembly/configuration failures; send only active-language, nonduplicated client context; require the scientist's actual question instead of a prefilled recipe goal. The source-forgery regression covers the first fix. The actual browser fault injection and question checks cover the second and fourth.

Initial xAI requests for universal sensitive-data detection, cryptographic receipt authentication and mechanically guaranteed model obedience were not treated as delivered guarantees or code-supported release blockers. This interface performs local preparation, explicitly marks receipts unauthenticated, warns against confidential transfers and has no hosted model execution. Those residual risks remain real; a keyword filter would not establish privacy or clinical compliance. Source anchoring was separately strengthened as above.

Post-fix review artifacts: `/tmp/llm-offload-l38mBR`. Both independent responses found no supported major or blocker issue in the supplied fixes. Minor observations about partial clones, structurally missing DOM elements and the trusted toolchain remain bounded limitations, not claims of adversarial sandboxing.

| Response | SHA-256 |
|---|---|
| `grok-fast.json` | `11ad54bdbc4c83e8c76a2f2889042b8046f94879906ddaddc6ee5fd145cc4950` |
| `kimi.json` | `5966ffe20d6ab6501e9efdb0283ffc5db3dc046cdf4d9a32bd034f28d54fa12a` |

## Actual local-model behavior: not certification

Darwin's `coder-next` alias supplied an early clarity review. Its useful distinction between estimated uncertainty with an explicit basis and invented uncertainty was incorporated in all three languages. The alias is recorded as a router alias, not an independently verified provider/model identity.

A separate missing-data behavioral probe sent the actual generated Portuguese acid–base prompt to that local model with no tools, no domain-file bodies and no study data. Before the intake-phase revision, the model falsely implied it had inspected a domain source and introduced unsolicited numerical material. This was a failed behavioral probe, not an executed scientific example.

The prompt was strengthened to require an `INTAKE_ONLY` first response, at both the beginning and end: clarify the question without narrowing it; distinguish supplied study data, domain bodies actually read and execution status; do not supply code, formulas, illustrative numbers or invented domain limits. URL, filename, hash and historical receipt are explicitly not source inspection.

In the repeated local probe, the model correctly stated that domain files had not been read and execution had not occurred. It still expanded the requested scope and introduced an unsolicited domain-range suggestion. The repeat is therefore only a partial behavioral improvement, not a complete conformance pass. No numerical content from either response was adopted as a reference or calculation. Scientific approval and inspection of actual sources remain necessary; prompts cannot certify arbitrary LLM behavior.

The final intake delta was independently reviewed by the same two vendors in `/tmp/llm-offload-QcNcjm`; both returned no supported major or blocker. Kimi's minor observation about the ambiguous `source-inspected-only` label was accepted: the emitted label now reads `source-identity-pinned; model-access-not-established; not-executed`, with a checker assertion rejecting the old wording. This removes implied model access without claiming that the remaining behavioral failure is solved.

| Final delta response (`finish_reason: stop`) | SHA-256 |
|---|---|
| `grok-fast.json`, `grok-4.3` | `13977e7be403cddb1ac6da85e549b1c447e96db2f7cfdc67b1ca1cbad6bb80f1` |
| `kimi.json`, `aws-bedrock/kimi-k3:global` | `4494762e615f8f45799dd3abf02ebe82e9d897ef8d1343d3487b4d7dd0eb2fd9` |

Protected CI, merge, production deployment and live-site acceptance must be reported independently from this prepublication receipt.
