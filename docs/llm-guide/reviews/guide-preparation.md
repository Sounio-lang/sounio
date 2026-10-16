<!-- docs:meta
topic_id: repo.docs.llm-guide.reviews.guide-preparation
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.llm-guide.reviews.guide-preparation
-->

# LLM guide preparation and review

Prepared 2026-10-16 against source baseline
`f5b2c5850be9f3d3995719ffcade44a6d851658a`, Git tree
`25a06aafc245efec6745e72e2fa49e12c7436ff0`.

The purpose is an approachable, deep programming guide with complete examples,
not a claim that every language model will generate correct programs. The
edition separates quick context, bilingual manuals, progressive recipes,
source-bound programs, API discovery, and diagnostic repair. It preserves the
founder's scientific contracts rather than simplifying away their information.

## Work and ownership

Six bounded subagent lanes were assigned to English/Portuguese preparation,
source/API recipes, diagnostics/tool behavior, executable examples, and the
structural validator. Independent completed source/API, diagnostic, example,
and validator work was centrally integrated. The coordinating agent completed
the English and Portuguese manuscripts after taking over their unfinished
writing lanes; the manuals are not represented as separately finished
subagent manuscripts. A second reader reviewed the English guide against the
source snippets and observed case ledger.

All work stayed in the isolated detached preparation worktree. The preexisting
specification and root cleanup were preserved. No compiler, standard-library,
MCP implementation, original test source, expected test output, branch,
commit, push, PR, or release was changed by this guide task.

## Deliverables and maintenance contracts

| Surface | What changed |
|---|---|
| `llms.txt` | Compact context, actual host/artifact preflight before workloads, actionable stop conditions, model task/evidence contract, MCP attribution. |
| English and Portuguese guides | Fifteen matched topics, four complete exact-source core programs each, progressive retrieval, repair, scientific preservation, and prompt templates. |
| `cookbook.md` | Ten progressive lessons, complete source files, actual case outcomes, and a separate quarantined diagnostic lab. |
| `examples.md` | Fifty full source-bound case blocks; eight visible quarantines, with original expectations preserved. |
| `stdlib-index.md` | Sixty-three source-exact signature extracts and a declaration/body/import/consumer/receipt discovery procedure, not blanket maturity ratings. |
| `error-catalog.md` | Actual route/stage/code distinctions, safe repair, and preserved historical namespace boundaries. |
| Per-code explanations | E035 and E249 corrected; fifty other retained explanations visibly marked not refreshed by this audit. Their historical bodies were not silently rewritten. |
| `example-manifest.json` / `example-results.json` | Baseline source hashes, expected contracts, observed commands/exits/streams, profile, and per-case status. |
| `validate_llm_guide.mjs` | Exact-source/baseline binding, bilingual topic coverage, links, complete atlas, and mandatory visible quarantine for failed snippets. |
| Copilot, support, and docs index | Entry points point to the same guide rather than repeating contradictory language rules. |

The complete programs remain the original source, including any historical
comments. Such a comment is not new semantic authority. The validator compares
the programs to baseline Git blobs and their current files. Text fragments and
signature excerpts are labeled nonrunnable; they are not counted as executable
programs. Repeated source aliases are not independent programs.

## Actual execution

The [full execution receipt](example-execution.md) and
[raw case records](../example-results.json) identify Slurm job `14651`:

| Result | Cases |
|---|---:|
| Run contract matched | 30 |
| Check-only contract matched | 4 |
| Intended refusal matched | 8 |
| Intended contract failed | 8 |

All 50 entries were actually invoked, over 47 distinct sources. The scheduler
and collector completed with exit zero; that is not an all-pass result.
The executed compiler was the baseline committed Madaros Linux x86-64
prebuilt, raw SHA-256
`125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f`.
It was not freshly built from compiler source. All 2016 archived input files
passed the post-execution hash check.

The integration owner independently read the remote receipt bundle, matched
its SHA-256
`8484714c86d852a7cf4f95b5e8c9216c6d8b033eb2d9ba3175dd90e06d24d55e`,
and read the raw case records. Source/input identity is established by the
archive and per-file hashes, not the launcher's version string or a fictitious
Git checkout inside the extracted snapshot.

The failed closure, single-generic, generic-pair, nested-destructuring, and
selected units fixtures are excluded from the working learning path.
Two failed negative units cases lacked the intended diagnostic; their nonzero
exit is not evidence for that intended boundary. Their original kinds,
sources, expected exits, and messages remain in the ledger. The working
generic example uses the independently executed `generics` case; named
function references and verified tuple access have their own receipts.

No source repair or universal feature prohibition follows merely from these
profile-specific failures. In particular, an invalid source spelling, a stale
fixture expectation, incomplete inference, and unsupported lowering require
different investigation. Do not relabel all of them compiler semantic defects.

## Orthogonal provider reviews

Operational review used `bin/llm-offload` from the operational checkout,
copied into remote scratch. It is not claimed to be tracked at this baseline.
The local operational log is `.claude/llm_offload_log.md`; the repository's
ignore rules hide that operational file, so this document supplies the
publication-safe durable record. No provider credentials or patient data
were sent with these language-guide drafts.

| Review | Actual scope and outcome |
|---|---|
| Darwin reader | Attempted compact-context review; HTTP 500. No review pass. |
| Initial Kimi K3 | Completed compact-context review, response model `aws-bedrock/kimi-k3:global`, finish `stop`. Six clarity/attribution findings adjudicated. |
| Initial Grok 4.7 | Response model `aws-bedrock/grok-4-7:global`, finish `stop`, but only repeated promises to review. Not a substantive completed review or a pass. |
| xAI fallback | Completed practical-core review through `xai-fast`; actual response model `grok-4.3`, finish `stop`. No major issue in the supplied core text. |
| Kimi K3 | Completed same practical-core review, finish `stop`; two major usability findings accepted and corrected. |
| Final bilingual xAI fallback | Response model `grok-4.3`, finish `stop`; no major issue in the supplied English/Portuguese manuals. |
| Final bilingual Kimi K3 | Response model `aws-bedrock/kimi-k3:global`, finish `stop`; four wording/parity findings adjudicated with clarifications. |

The practical-core review included `llms.txt`, Copilot instructions, guide
index, cookbook prose, stdlib index, and diagnostic guide. Exact-source Sounio
blocks were omitted from the editorial input, not from the guides; their
source binding and execution were checked separately. That review did not
claim to have read the then-unfinished bilingual manuscripts or the full
atlas. The separate bilingual review read both complete manuals' prose and
templates, with only their four source-bound program blocks omitted from the
editorial input. Those programs were separately source-checked and executed.
The operational log records the actual review scopes and outcomes.

Accepted improvements include: host prerequisites before compiler invocation;
hashing the resolved raw artifact, not the wrapper; actionable `PREFLIGHT STOP`
messages; first-success expected output; definitions for tool/profile/control
terminology; identical CLI/MCP artifact/library attribution or separate unknown
results; and explicit authorization-sensitive lightweight versus remote-heavy
execution policy. The suggested wrapper hash and direct raw-binary substitution
were rejected: neither establishes the correct launcher execution contract.
The suggestion to ignore a failed artifact-inspection command was not adopted.

The bilingual review led to matched receiver terminology, an English
counterpart to the Portuguese proceed/clarify policy, unambiguous wording
that neither validator nor mutation tests executes the compiler, and a
clearer separation between specification contracts and profile-specific
execution witnesses. `self` was checked against actual source signatures;
it was not treated as an invented token merely because one translation had
omitted it. No language semantics or evidence count was changed.

Core-review response identities:

```text
98c151bda8154edeec122a032e3fea939f1e21e28134906181fc4374d569bdb7  xai-fallback-core-review.md
10bb041d5b519f6368ad83b8805a7c2870a6ec3c672125e182021492ada95f23  kimi-core-review.md
8d9af8ad71eee41184cfa1407b1ced1d2226bb96447931d77ab64fdd49df1cb0  core-review-input.md
cec910fd2d28ec5a2973ec39fd2209614383ecb7f459f8ab46f50c82f250ae21  xai-fallback-bilingual-review.md
0295dab1d6c4b23b400d0427527626de871dbec9b7ed6aa2ea357234bca3bca2  kimi-bilingual-review.md
4d90f04d03f884b50d5f1688ca6f104b5b5e8ca0514d171c33f14b82d4b00305  bilingual-review-input.md
```

These identify review bytes, not semantic acceptance, a numerical oracle, or
a proof. Provider output is advisory; unsupported suggestions were not made
language rules.

## Validation and remaining boundaries

Reproduction commands, from the preparation worktree:

```sh
node --check scripts/docs/validate_llm_guide.mjs
node scripts/docs/validate_llm_guide.mjs --self-test
node scripts/docs/validate_llm_guide.mjs --json
node scripts/docs/validate_language_spec.mjs --self-test
node scripts/docs/validate_language_spec.mjs --json
node scripts/docs/check_docs_registry.mjs
git diff --check
```

The LLM-guide validator does not execute a compiler or authenticate an
execution receipt. Its mutation controls test structural false passes,
including source edits/hash drift, missing/inconsistent bilingual topics,
partial/unmarked programs, invalid negative contracts, unsafe paths, broken
links, and hidden quarantine labels. The remote execution receipt has its
own provenance and readback. Shell-preflight mock checks exercise stop/error
paths without invoking a real compiler.

Independent mutation review found three additional guard gaps: root-context
code fences bypassed full binding, a misspelled verification state could hide
quarantine, and a positive sentence mentioning “Quarantine” could qualify as
a warning. The validator now rejects executable Sounio fences in the compact
root context, validates the supplied state vocabulary, and requires an explicit visible
non-use warning before a failed example's code. Regression mutations cover
the reported false passes. These checks do not authenticate compiler receipts
or replace editorial review of the prose.

The integrated structural check passed with 50 baseline-bound entries,
47 distinct source files, and 80 exact-source code blocks across the two
manuals, cookbook, and atlas. All eight failed atlas cases and three failed
cookbook cases have explicit quarantine markers and visible warnings.
Both manuals have all 15 required topics. Documentation registry and
whitespace checks passed. The registry's stale-snapshot notice is
informational; no unrelated generated snapshot was rewritten.
The final validator self-test run passed 171 pure mutation cases. The
separate specification validator also retained its 73 passing controls and
332-rule structural pass; the guide refresh did not rewrite the specification.

No generic LLM coding-success benchmark, full compiler build, complete corpus,
cross-target run, GPU experiment, independent numerical audit, Lean build,
remote CI, merge, publication, or human acceptance is asserted. Metadata stays
`unvalidated` rather than converting structural checks into whole-document
semantic certification. The guide's intended ease of use is a design goal,
not a fabricated success rate for arbitrary models.

The outstanding example-promotion investigations are recorded in the execution
receipt. They do not block an accurately scoped guide, because the failed
cases are excluded from its recommended path. Promoting a failed case requires
new discriminating evidence on a named profile, not removing its warning or
changing the expected result. Existing bootstrap/legacy paths were retained;
none was silently substituted for the recorded default profile.
