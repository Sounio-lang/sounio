<!-- docs:meta
topic_id: repo.docs.scientist.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.scientist.readme
-->

# Research workflows in natural language

[Português](README.pt.md) · [Ελληνικά](README.el.md) · [Scientist workspace](https://www.souniolang.org/scientist/)

Start with your scientific question. Describe your data and its origin, measurement units, model assumptions and desired artifacts in ordinary language. Identify missing observations explicitly and describe measurement uncertainty and covariance separately; an absent value is not zero, and absent covariance is not evidence of independence. You do not need to supply code or terminal commands.

Choose a recipe in the [Scientist workspace](https://www.souniolang.org/scientist/), answer its scientific clarifications and take the prepared request to your chosen LLM. Ask for the intended tables, plots, comparison report and evidence record. The connected LLM inspects the complete existing source, prepares code and handles actual commands within the execution environment and authorization you provide. It should report unknowns and compiler refusals instead of silently changing your model, units, uncertainty representation or engine.

An optional local connected agent can read the checkout and execute authorized checks where a compatible compiler and dependencies are available. A chat-only LLM cannot execute the workflow: its code, suggested commands and predicted output remain unexecuted until an actual connected environment returns evidence. This interface supplies prompts; it does not provide a current hosted runtime or execute research on the website.

## Two stages, with scientist confirmation

The first copy, download and preview actions prepare only your scientific brief and the closed `SOUNIO_INTAKE_V1` protocol. This stage includes no code, domain source links or language examples. Paste the request into your chosen AI, then paste its complete reply unchanged into the clarification box. You do not need to write JSON or code. Use the field hints and scientific sidebar to resolve missing information in your brief.

Select **Validate clarification**. The local validator checks receipt syntax and correspondence with the current six scientific fields, workflow and language. Review the matching brief and reply, then explicitly check the scientist confirmation. Only then can you copy or download the second-stage implementation instructions, which include the full canonical source-bound context pack. Each implementation action validates the current reply again. Editing a scientific field, changing workflow or reply, or unchecking confirmation revokes acceptance; validate and confirm again. Replies above 60,000 characters are rejected.

This gate does not establish scientific truth, LLM authorship, source attestation or actual tool use. `source_access: not_verified` and `execution: not_executed` describe evidence for this request, not everything a model may previously have read. Confirmation prepares instructions; it neither executes a study nor automatically approves a scientific model. Consequential scientific choices still require scientist review.

## Available research recipes

The [catalog](../../website/src/data/scientistWorkflows.json) binds these recipes to existing sources at `106f7a988d5b007472d0eead724f58ac50f92200`. Every recipe has state `prompt-template-not-executed`. Its `starter_task` is an existing language-learning task, not a domain-validation ID. None of these domain programs is an example ID in the [LLM execution manifest](../llm-guide/example-manifest.json).

| Recipe | Scientific inputs → requested artifacts | Source and historical evidence boundary |
|---|---|---|
| Hydrogen ignition and uncertainty | Mixture, initial conditions, units and declared uncertainties → species, ignition-delay and uncertainty-attribution reports. | [Demo](../../examples/chemistry/h2_ignition_uq_demo.sio); [historical golden](../../benchmarks/chemistry/golden/h2_ignition_uq_demo.lean_single.txt), selected by the [lean_single gate](../../scripts/ci/chemistry_probe_golden_gate.sh). A retained output is not a fresh gate result or current Madaros evidence. |
| Surface mechanism and pressure response | Adsorption hypothesis, pressures, units and parameter provenance → coverages, convergence residuals and apparent orders. | [Demo](../../examples/chemistry/h2_surface_reaction_order.sio); [historical measurement](../chemistry/SOUNIO_FOR_SURFACE_MICROKINETICS.md). The source names gen3 execution and a Madaros lowering failure; keep the profiles separate. Missing absolute loss rates cannot be invented. |
| Acid–base measurements and uncertainty | Concentrations or buffer/titration inputs, units and measurement uncertainty → bounded result table and implementation limits. | [Functions](../../stdlib/chemistry/acids.sio), [textbook driver](../../tests/stdlib/chemistry/test_acids_stdlib.sio); [private-function identity audit](../audit/MADAROS_PRIVATE_FN_IDENTITY_2026-09-21.md) documents a historical pH import repair, not full metrology validation. Some uncertainty expressions are explicitly rough; the combined driver is check-only. |
| Oral pharmacokinetic model check | Synthetic absorption, elimination, volume and time assumptions → concentration trajectory, peak summary and analytic comparison. | [Program](../../examples/science/pkpd_simulation.sio); [historical check ledger](../audit/USER_FRONTIER_EXAMPLES_CHECK_2026-08-19.tsv) records a check pass only. The source has no pharmacodynamic response model or uncertainty propagation. |
| Synthetic sequential trial updates | Synthetic binary outcomes, prior and batch assumptions → posterior update history and interval explanation. | [Program](../../examples/science/bayesian_clinical_trial.sio); the same historical check ledger records a check pass only. Intervals use a clipped normal approximation; they are not exact posterior quantiles or a validated trial design. |
| PBPK distribution and mass audit | Synthetic profile, compartment units and declared model boundaries → tissue trajectories and bounded mass checks. | [Smoke fixture](../../tests/run-pass/darwin_pbpk28_smoke.sio); [historical cross-engine census](../audit/CROSS_ENGINE_RUNPASS_CENSUS_2026-08-17.md) records matching mass/step markers. The [later mass audit](../audit/PBPK28_CN_RANNACHER_MASS_BALANCE_2026-09-26.md) explains why that fixture does not establish general conservation. Read executable settings rather than its stale header. |

## What to expect from the assistant

Before any execution, the agent must identify the selected source revision, compiler engine/artifact, target and standard-library dependencies. Imported examples need their actual providers. Historical lean_single, gen3 and Madaros observations are distinct prerequisites and cannot be silently substituted. Unsupported inputs, missing rates or unavailable covariance need an explicit decision before adaptation.

Request the input record, separate assumptions, named output artifacts and the actual commands, source identity, exit status and raw output from every authorized run. Keep checking, compilation, execution, tests, scientific comparison and clinical validation separate. Copied source, passing assertions and historical receipts do not validate an adaptation. No scientific workflow was executed to prepare this catalog.

Use synthetic demonstrations. Do not send patient records to public or unapproved services, including through a local agent's model connection. Clinical and research authority remains with the responsible professionals: these recipes provide no automatic diagnosis, dosing, patient-specific prescribing or clinical-readiness claim.

For the connected agent, use the existing [LLM workflow guide](../llm-guide/README.md), [English language guide](../guide/LLM_PROGRAMMING_GUIDE.md) and [task context tool](../../scripts/docs/sounio_task_context.mjs). They supply the language context and evidence contract; the scientist supplies the research question. Prompt generation belongs to the Scientist workspace, so these pages do not duplicate a large prompt.
