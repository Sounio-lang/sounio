<!-- docs:meta
topic_id: repo.docs.research.extracted-repositories
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.extracted-repositories
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Extracted repositories

On 2026-10-06 two bodies of work that are not part of the language were
moved out of this repository into new private repositories in the
`Sounio-lang` organization, with their history. This page links them and
records what moved and what stayed.

| Repository | What it is |
|---|---|
| [`Sounio-lang/sounio-loom`](https://github.com/Sounio-lang/sounio-loom) (private) | Loom: the agent-coordination tooling used to develop Sounio |
| [`Sounio-lang/sounio-fpga-research`](https://github.com/Sounio-lang/sounio-fpga-research) (private) | FPGA and validated-numerics research results (the cs6 U250 line) and the non-associative connectomics experiment. Results, not language code |

Nothing was deleted from `main` by the extraction. Every source branch is
kept in this repository as the tag `archive/2026-10-05/<branch>`. Those tags
are not removed.

## sounio-loom

Built by following `docs/research/LOOM_EXTRACTION_PLAN.md` from [PR #2807](https://github.com/Sounio-lang/sounio/pull/2807) (not yet on `main` when this page was written).

**What moved.** The 23 Loom branches listed in the plan, filtered with
`git filter-repo` to the Loom paths:

- `tools/loom/`, `stdlib/coordination/`, `formal/tla/SounioFleet.*`;
- `bin/sounio-{coord,loom,fleet,agentd}`;
- the `loom`/`coord` scripts under `scripts/ci`, `scripts/dev` and `scripts/mcp`;
- `docs/internal/concepts/loom-*`;
- the `loom_*`, `causal_receipt*` and `fleet_transaction*` tests;
- `loom-coordination.yml` and `coord-github.yml`.

Each branch is pushed there as `archive/<branch>`. Its `main` starts from
`codex/loom-message-bridge-runtime-20260903` and merges, in order:

1. `loom/segunda-ordem`;
2. `codex/loom-terminal-v15-20260905`;
3. `codex/loom-routing-authority-integration-20260903`;
4. `lane/grok-cli1/loom-handshake-exec-cell-coherence-20260830`;
5. `fix/github-lane-overlap` (PR #2733).

On top of those merges it restores the 40 concept contracts and typestate
tests from `codex/loom-apple-delivery-ux-20260903`, and
`loom-language-authority.contract`. The repository README records the
source SHA of each merge. It also records three departures from the plan,
each with its reason:

- the Apple workbench files are taken from routing-authority;
- the base's runtime-client `bin/sounio-coord` is kept as
  `bin/sounio-coord-runtime-client`;
- `loom.ml` is a real merge of the base and terminal-v15.

**What stayed here, and why.**

| Stays in `sounio` | Why |
|---|---|
| `stdlib/coordination/{causal_receipt,fleet_transaction,loom_continuity,loom_obligation}.sio`, their `tests/compiler` and `tests/compile-fail` witnesses, the four `*_typestate_gate.sh` scripts, `.github/workflows/loom-coordination.yml` | Language witnesses for linear reuse (E039), private host seal (E175/E176) and wrong-state (E009), bound in `docs/internal/concepts/registry.tsv`. They need no Loom runtime. `sounio-loom` holds copies; this repository stays authoritative. |
| `bin/sounio-coord` | Lane claims are repository policy. `ci.yml` runs `scripts/ci/sounio_coord_selftest.sh`, and `BRANCH_POLICY.md` and the code-review skill name the tool. It stays until it is replaced by a shim, in a separate change. |
| `stdlib/coordination/spark_pair_*` | Not Loom. They arbitrate the DGX Spark pair and belong to the Pireus/cluster decision. They were also removed from `sounio-loom`'s history. |
| `tools/loom/evidence/pireus-*-acceptance-20260827.txt` (6) | Referenced by `tools/pireus/continuity`. They move only together with it. They were also removed from `sounio-loom`'s history. |
| `docs/internal/concepts/loom-obligation.contract` | Registry row. Its evidence citations can point at `sounio-loom` later. |

PRs #2154 and #2733 were closed with a pointer to `sounio-loom`. The
`AGENTS.md`, `CLAUDE.md` and `.claude/ATTENTION_CHARTER.md` hunks of #2733
describe sounio policy and were not carried over. If wanted, they need their
own PR here.

## sounio-fpga-research

**What moved.** The 17 `research/cs6-*` branches, filtered to:

- `hardware/fpga/u250_target23_*` and `hardware/fpga/u250_validated_dyadic/`;
- `scripts/research/cs6*` and `scripts/research/receipts/cs6*`;
- `docs/research/cs6_*`;
- `scripts/ci/cs6_*`.

Also `feat/octonion-mass-delta`, filtered to
`experiments/non_assoc_connectomics/`. That branch adds 22 Python scripts;
its other 36 files are identical to `main`. The repository has 2,240 files,
including 33 HLS/host files, 246 research scripts and 1,845 receipt files.

How the cs6 branches combine:

- 15 of the cs6 branches are ancestors of `research/cs6-arb-full-leaf-20260804`.
- `research/cs6-v7b-full-hpg-bridge-20260802` adds one commit, the chained
  Taylor-41 run.
- The new `main` merges both cs6 heads and the octonion branch. Each file on
  `main` is the newest version across all 17 cs6 branch tips; this was
  checked file by file.

**What stayed here.**

- `hardware/fpga/u250_catastrophe_scan/` and its gate: `main` has the newer
  copy.
- The language and other research work that the cs6 branches carried from
  their 2026-07-26 integration point:
  - the `madaros_gum_fo_*` tests;
  - the `self-hosted/` and `stdlib/epistemic/fo.sio` edits;
  - the `self_falsifying_*`, `mercyful_*` and ZD gates and docs;
  - the papers.

**Status.** The receipts were not re-run during extraction. The repository
README gives each kernel's own claims and non-claims. For example:

- the scaled Taylor-16 kernel records a physical U250 run (459/459 words);
- the chained Taylor-41 kernel records a failed timing closure and no
  physical execution.

## Branches removed from this repository

After both repositories were pushed, 41 remote branches were deleted on
2026-10-06:

- the 23 Loom branches listed in the plan, including the heads of the now
  closed PRs #2154 and #2733;
- the 17 `research/cs6-*` branches;
- `feat/octonion-mass-delta`.

Before each deletion, `git ls-remote origin refs/tags/archive/2026-10-05/<branch>`
was checked against the branch tip. All 41 matched. None was protected by a
ruleset or was the head of an open PR, so none was skipped.
`docs/loom-extraction-plan` (PR #2807) was not touched. To restore any deleted
branch locally, run
`git fetch origin 'refs/tags/archive/2026-10-05/*:refs/tags/archive/2026-10-05/*'`.
