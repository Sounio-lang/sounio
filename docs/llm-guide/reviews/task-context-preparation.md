<!-- docs:meta
topic_id: repo.docs.llm-guide.reviews.task-context-preparation
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.llm-guide.reviews.task-context-preparation
-->

# Task-context extension: preparation and review receipt

## Scope and source identity

Prepared on 2026-10-17 in an isolated worktree based on
`6861acfdf75c29e1174a9f668dea2877020f6012`, the published guide integration.
The existing example manifest remains bound to
`f5b2c5850be9f3d3995719ffcade44a6d851658a`. No example source, intended
contract, historical execution result or quarantine classification changed.

This extension adds thirty bilingual adaptation exercises and a read-only
context generator. Sixty context packs are the two-language renderings of
those thirty tasks, **not sixty independent programs or execution witnesses**.
The selected reference/control inputs cover thirty-six distinct existing
source files. The original atlas still has fifty cases over forty-seven
distinct sources; its eight unexpected failures remain quarantined.

The new curriculum and translation/repair guide are linked from the existing
English/Portuguese manuals, LLM-guide entry point and compact root context.
No compiler, standard-library, formal-proof, resolver, MCP-server, bootstrap
or legacy implementation was modified or replaced. The generator uses Git
batch reads and Node standard-library operations, not a Sounio invocation,
numerical oracle, model API call or shell-evaluated candidate command.

## Author source review and corrections

- The Input/Measured fixtures only exercise annotation spellings and print
  markers; their annotated helpers are not called from the entry point.
  The task cards explicitly avoid calling these runtime provenance-storage,
  measurement-transport or instrument-acquisition witnesses.
- The borrow-release fixture has explicit call-site borrows and return
  boundaries, not a separate release instruction. Its card and curriculum
  were corrected accordingly; no general runtime-reclamation claim is made.
- The ordinary-functions fixture prints a result rather than containing an
  embedded result assertion. Its adaptation criterion now says printed-result
  contract, not a nonexistent assertion.
- Check-only Knowledge/import cases retain that limit in the task table and
  generated packs. Public-import packs include both provider and consumer.
- Failed or unexecuted cases cannot become starting templates or controls.
  Intended-refusal controls are labeled as negative, never working templates.
- A source archive or shallow clone can lack the historical Git objects.
  The prerequisite and direct-manual fallback are documented. The tool does
  not silently fetch history or weaken its source-identity check.

## LLM-offload reviews invoked

The parent retained architecture, semantic decisions and final review.
Provider responses are advisory and do not certify source or prose.

| Route | Target and actual scope | Outcome |
|---|---|---|
| Darwin `coder-next` via `consult` | Pedagogical additions, no source code supplied | Returned routing/repair/evaluation ideas mixed with invented studies, percentages, flags and telemetry. Only the general routing/evaluation ideas were considered; every unsupported factual/numerical claim and invented interface was rejected. Not a correctness review or pass. |
| Local operational `bin/llm-offload --raw` | Attempted independent review | No completed provider response; Kimi reported missing gateway credentials in this environment. Not a pass. |
| t560 host transport | Operational-wrapper lookup | Neither named wrapper was available on that host. No review occurred. |
| Workspace `/dev/stdin` transport | Attempted review payload transfer | Input open permission failure before provider dispatch. Not a pass; retried with a dedicated transferred input file. |
| DeepSeek via workspace operational `bin/llm-offload --raw` | Full new generator, catalog and both new prose files | Authentication error; no review, not a pass. |
| Gemini via the same wrapper | Same complete input | Credit error; no review, not a pass. No account changes or funding actions taken. |
| xAI requested `xai-fast`; response model `grok-4.3` | Complete new generator, thirty-card catalog, curriculum and translation/repair prose | Completed with `finish_reason=stop`; no major/blocker findings. This is an advisory scoped review, not proof that false passes are impossible. |
| Moonshot requested Kimi K3; response `aws-bedrock/kimi-k3:global` | Same complete input | Completed with `finish_reason=stop`; no major/blocker findings, five minor hardening/documentation observations adjudicated below. |

The successful input was transferred to the workspace's isolated review
scratch rather than a shared source checkout. Provider dispatch used the
existing operational wrapper at
`/tmp/sounio-spec-review.l6qZ9V/bin/llm-offload`. Operational outputs are in
`/tmp/llm-offload-Dix6zv`; the input is
`/tmp/sounio-llm-context-review-20261017.zLO9ha/review-input.md`.
These are custody locations, not portable prerequisites for readers.

SHA-256 identities of the reviewed input and complete response text:

```text
73345e9552f6fafeb44afa520eb6c03cc164775cefb3380b5800e8b880d3e71f  review-input.md
27845abe282ef534eee0a11d9f2633fa4462ee33ad7383dfcb5563bc4f4e3356  grok-fast.md
ef526e592605ec06c084907e9f4277c56058e47fbdec01803e1c14bd94d013ae  kimi.md
```

The input identifies the draft actually reviewed, not every later byte.
Subsequent author changes applied the dispositions below and improved
Portuguese accents and the prerequisite note. No second provider review of
that final small delta is asserted.

### Kimi observations and dispositions

1. **Positive exit type/range.** Accepted: eligible rows now require a
   positive kind and integer process statuses, with zero required for a
   successful check. Mutation tests reject matching string/null/fractional,
   negative and out-of-range pseudo-statuses.
2. **Contradictory control kind.** Accepted: a negative-kind control cannot
   carry an eligible positive state. Regression mutation added. Failure
   reasons also cannot coexist with the selected passing observations.
3. **Exported context helper assumes validation.** Added an explicit missing
   example/source error. The CLI still validates the complete input first;
   importing a helper is not a replacement for that precondition.
4. **Git prerequisite.** Accepted and documented: baseline objects and a Git
   checkout are required for source-bound generation; no automatic network
   fetch or degraded verified label is used.
5. **Linked files absent from editorial bundle.** Their presence/anchors are
   checked separately by the existing documentation validator. The provider
   did not inspect those files; no such inspection is attributed to it.

## Reproduction and observed checks

From the preparation worktree:

```sh
node --check scripts/docs/sounio_task_context.mjs
node scripts/docs/sounio_task_context.mjs --self-test
node scripts/docs/sounio_task_context.mjs --validate
node scripts/docs/sounio_task_context.mjs show first-program --lang pt --json
node scripts/docs/sounio_task_context.mjs search 'referência' --lang pt
node scripts/docs/validate_llm_guide.mjs --self-test
node scripts/docs/validate_llm_guide.mjs --json
node scripts/docs/validate_language_spec.mjs --self-test
node scripts/docs/validate_language_spec.mjs --json
node scripts/docs/check_docs_registry.mjs
bash scripts/dev/check_docs_consistency.sh
git diff --check
```

The context self-test passed **58 controls**. Real-input validation produced
thirty tasks, sixty complete bilingual packs, thirty-six source files and
no structural errors. An additional process-level integration check exercised
all sixty packs, seven invalid CLI forms (each exit 1 with no context stdout),
the two-file public-module dependency, and explicit negative-control labeling.
These checks did not invoke a compiler or authenticate historical receipts.

The existing guide validator retained its 171 pure mutation controls and the
fifty-case / forty-seven-source / eighty-bound-snippet structural pass. The
specification validator retained its 73 controls and 332-rule structural pass.
Documentation registry and consistency checks passed; the generated-registry
snapshot notice is informational and no unrelated snapshot was rewritten.
Whitespace checks passed, including added files checked separately before
delivery. All new metadata remains `unvalidated`, not whole-document
semantic certification based on structural checks.

## Boundaries and delivery state

No new compiler execution, current-source build, full corpus, Slurm job,
cross-target or GPU execution, independent numerical calculation, Lean proof,
model coding-success benchmark or empirical study was performed for this
extension. Existing historical receipt outcomes were reused with explicit
source/profile limits. This work does not promote the eight failed examples
or claim that every LLM will succeed on arbitrary Sounio tasks.

This receipt covers prepared and checked files. No new commit, push, PR or
merge is asserted for the extension. Publication is a separate checkpoint.
There is no unresolved blocker to this accurately scoped documentation/tool
delivery. Future compiler capability and failed-example promotion work remains
outside this write set; bootstrap and legacy paths were intentionally kept.
