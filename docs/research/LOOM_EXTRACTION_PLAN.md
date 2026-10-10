<!-- docs:meta
topic_id: repo.docs.research.loom-extraction-plan
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.loom-extraction-plan
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Loom extraction plan

Status: plan only. No code has moved and no repository has been created.
Creating `Sounio-lang/sounio-loom` needs the owner's approval.

Loom is the agent-coordination layer that grew inside this repository. It
consists of the lane runtime and OCaml multiplexer (`tools/loom`), the
coordination typestate modules (`stdlib/coordination`), the fleet TLA+ spec,
the `bin/sounio-coord` / `sounio-loom` / `sounio-fleet` / `sounio-agentd`
entry points, and about 470 selftest and installer scripts. None of it is
part of the language. This document records the measurements needed to move
it out with its history, and what `main` still needs after the move.

All counts were measured on 2026-10-06 against `origin/main` at `a11e38e6d`
and the archive tags `archive/2026-10-05/*`. The commands used are in the
appendix.

## 1. Branches and PRs that carry Loom

23 refs. Each is preserved as `archive/2026-10-05/<branch>`.

| Branch | Ahead / behind `main` | Last commit | Role |
|---|---|---|---|
| `codex/loom-message-bridge-runtime-20260903` | 1613 / 1755 | 2026-09-21 | **Base for extraction.** Newest complete Loom tree. It merged `loom/segunda-ordem` on 2026-09-21. |
| `loom/segunda-ordem` | 1601 / 1755 | 2026-09-22 | Main Loom line. Two commits beyond the base: product v31 record, installer hash fix. |
| `codex/loom-terminal-raw-20260905` | 1602 / 1755 | 2026-09-21 | Same tree as the base for every Loom path (also merged `segunda-ordem`). |
| `codex/loom-terminal-v15-20260905` | 1601 / 1755 | 2026-09-21 | Same as above. |
| `loom/onda3`, `onda4`, `onda5`, `onda6`, `loom/refreeze` | ~1420–1565 / 1755 | 2026-09-14 … 09-17 | Ancestors of both `segunda-ordem` and the base. |
| `lane/codex-1/loom-mainline-20260827` | 1572 / 1755 | 2026-09-22 | Parallel mainline (merged `integration/sounio-dev-ready-base`). It holds no Loom file newer than the base. |
| `fix/loom-generation-pin-birth` | 1400 / 1755 | 2026-09-11 | Ancestor of the base. |
| `codex/loom-routing-authority-integration-20260903` | 1405 / 1755 | 2026-09-04 | 3 files newer than or absent from the base (Apple UI, two gate receipts). |
| `lane/grok-cli1/loom-handshake-exec-cell-coherence-20260830` | 1347 / 1755 | 2026-09-01 | 10 files absent from the base (3 selftests, 3 contracts, 4 receipts). |
| `wip/loom-mainline-native-hook-cutover-20260921` | 1348 / 1755 | 2026-09-21 | WIP snapshot: one concept-note edit. |
| `codex/loom-apple-delivery-ux-20260903` | 757 / 2871 | 2026-09-03 | Older line. 40 files absent from the base, mostly the `*.contract` concept docs the base dropped. |
| `wip/loom-codex-1-uncommitted-20260921` | 757 / 2871 | 2026-09-21 | WIP snapshot of a codex-1 worktree. **Parent commit is 2026-08-29**, see §2. |
| `lane/codex-1/loom-authority-20260827`, `loom-arrow-evolution-20260826`, `loom-docs-registry-20260827`, `loom-provider-custody-20260826` | ~760–812 / 2871 | 2026-08-27 … 08-28 | Early lanes, superseded. |
| `lane/codex-1/loom-main-20260825` = **PR #2154** | 128 / 1791 | 2026-08-25 | First runtime ship. The typestate half was landed on `main` separately. 18 evidence logs and 2 scripts exist only here; later lines deleted them. |
| `salvage/loom-coordination-typestate` | 1 / 1729 | 2026-08-28 | The typestate slice that reached `main`. |
| `fix/github-lane-overlap` = **PR #2733** | 4 / 530 | 2026-09-27 | Newest `bin/sounio-coord`, GitHub overlap preflight, MCP/agent-hook selftests, `.github/workflows/coord-github.yml`. Also touches `AGENTS.md`, `CLAUDE.md`, `.claude/ATTENTION_CHARTER.md`. |

## 2. Newest version of the Loom tree

The rule is the same as for Pireus. For each path, take the version whose
last-touching commit on its branch is newest. Across the 23 refs and `main`
there are **1,497 Loom paths**. Of those, 1,442 are absent from `main`, 53
are on `main` with the newest content, and 2 are on `main` but newer
elsewhere (`bin/sounio-coord` in #2733 and `scripts/ci/sounio_coord_selftest.sh`
in the base).

The newest version comes from the base line (`codex/loom-message-bridge-runtime-20260903`
and the identical `terminal-raw` and `terminal-v15` trees) for 1,331 paths.
The other 166 come from:

| Ref | Paths | Disposition |
|---|---|---|
| `fix/github-lane-overlap` (#2733) | 59: 52 identical to `main` (the typestate slice and `spark_pair_*`), 1 newer than `main` (`bin/sounio-coord`), 6 not on `main` (`coord-github.yml`, GitHub/MCP/agent-hook selftests and helpers) | Take it. It is the only ref with the GitHub preflight. |
| `codex/loom-apple-delivery-ux-20260903` | 41 | Take the 40 that are absent from the base, as files deleted or renamed later. Review them before import. |
| `wip/loom-codex-1-uncommitted-20260921` | 28 | **Do not take into the default branch.** See the conflicts below. |
| `lane/codex-1/loom-main-20260825` (#2154) | 20 | 18 evidence logs and 2 scripts (`install_sounio_loom_kubernetes_hook.sh`, `sounio_coord_agent_hook_runtime.py`) that the later line deleted. Keep them in history only; do not resurrect them in the tip. |
| `lane/grok-cli1/loom-handshake-exec-cell-coherence-20260830` | 10 | Absent from the base. Import as their own branch. |
| `loom/segunda-ordem`, `codex/loom-routing-authority-integration-20260903` | 3 + 3 | Take them: the two `segunda-ordem` commits, plus 3 Apple/gate files. |
| `lane/codex-1/loom-authority-20260827`, `wip/loom-mainline-native-hook-cutover-20260921` | 1 + 1 | `loom-language-authority.contract` (absent from the base); the WIP note edit. |

**Conflicts.** 23 paths have one version in the base and a different, newer
version on another ref. Each has an explicit pick:

| Path(s) | Newer ref (date) | Base date | Pick and reason |
|---|---|---|---|
| `bin/sounio-coord`, `scripts/ci/sounio_coord_agent_hook_selftest.sh`, `scripts/ci/sounio_coord_mcp_selftest.sh` | `fix/github-lane-overlap` (09-27) | 08-27 / 08-31 | **#2733.** It builds on `main`'s `sounio-coord`, which descends from the base's version, and adds the overlap preflight. |
| `scripts/dev/install_sounio_coord_runtime.sh` | `loom/segunda-ordem` (09-22) | 09-17 | **segunda-ordem.** A direct descendant of the base: installer hash fix. |
| `tools/loom/apple/Sources/LoomSpatial/ConversationSpace.swift` | `codex/loom-routing-authority-integration-20260903` (09-04) | 09-04 | **Base.** Same day; the base line merged routing-authority later. Diff the two at import. |
| `docs/internal/concepts/loom-native-hook-cutover.md` | `wip/loom-mainline-native-hook-cutover-20260921` (09-21) | 09-01 | **Base, with the WIP kept as a branch.** A one-file WIP snapshot. |
| `tools/loom/src/loom.ml`, `tools/loom/src/loom_pty_stubs.c`, `tools/loom/README.md`, `scripts/ci/sounio_loom_message_bridge_selftest.sh`, and 13 `tools/loom/apple/**` files | `wip/loom-codex-1-uncommitted-20260921` (09-21) | 09-03 … 09-17 | **Base.** The WIP commit's parent is `4a197750d` (2026-08-29). Its `loom.ml` (+349 lines) is a diff against the August file, so applying it would revert the base's September `loom.ml` work. Import it as branch `wip/codex-1-uncommitted` for someone to rebase. |

## 3. File list

Newest Loom tree by directory (1,497 paths, 372 of them under an `evidence/` directory):

| Directory | Paths | Notes |
|---|---|---|
| `tools/loom/` (top level) | 367 | 104 `.md` contracts and gardens, 70 `*_main.sio` authority adapters, ~190 `*.freeze.vN` / `*.first.vN` / `*.runtime.vN` receipts |
| `tools/loom/src/` | 114 | OCaml multiplexer and runtime (`loom.ml`, PTY stubs, dune) |
| `tools/loom/evidence/` | 359 | Receipts and logs |
| `tools/loom/apple/` | 33 | SwiftUI/Metal observatory (20 source/config files and 13 evidence files) |
| `tools/loom/systemd/`, `message_bridge/`, `apparmor/`, `bpf/` | 3, 2, 1, 1 | Principal-broker units, message bridge, AppArmor profile, BPF LSM program |
| `stdlib/coordination/` | 49 | 9 already on `main` (see §4); 40 `loom_*_authority.sio` and witness-mesh modules exist only on Loom branches |
| `tests/compiler/` | 40 | `loom_*`, `causal_receipt*`, `fleet_transaction*` (28 on `main`) |
| `tests/compile-fail/` | 22 | `loom_*` (3 on `main`) |
| `scripts/ci/` | 319 | `sounio_loom_*_selftest.sh`, `sounio_coord_*`, typestate gates (6 on `main`, 5 of them Loom: 4 typestate gates and `sounio_coord_selftest.sh`) |
| `scripts/dev/` | 153 | Installers, hooks, lane shell (none on `main`; 2 other `main` scripts reference `sounio-coord`) |
| `scripts/mcp/` | 1 | `sounio_coord_mcp.py` |
| `docs/internal/concepts/` | 25 | `loom-*.md` and `loom-*.contract` (1 on `main`: `loom-obligation.contract`) |
| `formal/tla/` | 2 | `SounioFleet.tla`, `SounioFleet.cfg` |
| `bin/` | 4 | `sounio-coord` (on `main`), `sounio-loom`, `sounio-fleet`, `sounio-agentd` |
| `.github/workflows/` | 2 | `loom-coordination.yml` (on `main`), `coord-github.yml` (#2733) |

To reproduce the full per-file list with each file's winning ref, use the
appendix command. It is not inlined here; it is 1,497 lines.

## 4. What `main` still references

`git grep` on `main` for `tools/loom`, `stdlib/coordination`, `coordination::`,
`sounio-coord`, `sounio_coord` and `loom` in workflows, scripts, `Makefile`,
`bin/` and `tools/cluster`. `Makefile` has no match.

| What `main` has | Who depends on it | Decision |
|---|---|---|
| `stdlib/coordination/{causal_receipt,fleet_transaction,loom_continuity,loom_obligation}.sio`; `tests/compiler/{causal_receipt,fleet_transaction,loom_continuity,loom_obligation}*` (28 files); `tests/compile-fail/loom_obligation_*` (3); `scripts/ci/{causal_receipt,fleet_transaction,loom_continuity,loom_obligation}_typestate_gate.sh`; `.github/workflows/loom-coordination.yml` | The workflow runs the 4 gates on every change to these paths. `docs/internal/concepts/registry.tsv` binds `SOUNIO-FALSIFICATION-CARRYING-DEVELOPMENT`, `SOUNIO-VERIFIABLE-FLEET-TRANSACTION` and `SOUNIO-DURABLE-AGENT-OBLIGATION` to these modules. | **Stay.** These are language witnesses (linear reuse E039, private host seal E175/E176, wrong-state E009) that use coordination as the example domain. They need no Loom runtime. Copy them into `sounio-loom` with history, and keep `main`'s copy authoritative for the language. |
| `bin/sounio-coord` | `.github/workflows/ci.yml:390` runs `scripts/ci/sounio_coord_selftest.sh`. Also named by `docs/governance/BRANCH_POLICY.md` (lines 24, 240), `.github/skills/code-review/SKILL.md:145`, `scripts/dev/pr_mergeability_refresh.sh:42`, `scripts/dev/sounio-lane-shell.reference:153`. | **Stay until replaced, then stub.** Lane claims are repository policy. Option 1: keep `bin/sounio-coord` (the standalone bash claim tool) in `sounio` and move only the runtime. Option 2: replace it with a 10-line shim that execs `${SOUNIO_LOOM_HOME:-$HOME/.sounio-loom}/bin/sounio-coord` and prints an install hint if that is missing. Make the `ci.yml` step and `BRANCH_POLICY.md` edit in the same PR. Land #2733's changes in `sounio-loom`, not here. |
| `stdlib/coordination/spark_pair_{arbiter,decommission,historical_provenance,read_only_capture_profile,restore_capsule}.sio` | `tools/cluster/spark_pair_*_main.sio` and `tools/cluster/*.v1` receipts (`authority_source=stdlib/coordination/spark_pair_*.sio`). Registry row `SOUNIO-REVERSIBLE-COMPUTE-CUSTODY` → `docs/internal/concepts/spark-pair-reversible-decommission.md`, which is absent on `main` (dangling). | **Not Loom. Leave for the Pireus/cluster decision.** These arrived with the Pireus stdlib PR #2466 and arbitrate the DGX Spark pair. Move them with the Pireus continuity evidence, or keep them. Either way, remove or fix the registry row in the same change. |
| `tools/loom/evidence/pireus-*-acceptance-20260827.txt` (6) | `tools/pireus/continuity/integration_manifest.json` and `source_inventory.json` | **Stay with the Pireus evidence** (see `PIREUS_EVIDENCE_LOCATION.md` on `pireus/consolidated`). Move them only together with `tools/pireus/continuity`. |
| `docs/internal/concepts/loom-obligation.contract` | Registry row above. It cites `tools/loom/evidence/obligation-v1-20260824/{prereg.json,outcome.json,receipt.txt}`, which `main` does not have (dangling). Those files exist in the base. | **Stay.** After extraction, point the citations at `sounio-loom` URLs. |
| Pireus CI wrappers (`scripts/ci/pireus_*.sh` on `pireus/10-workflow-wiring`, not on `main`) | They call `$GIT_COMMON_DIR/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime`. | Nothing to do on `main`. If they ever land, `sounio-loom` becomes a CI dependency. |

Nothing else on `main` imports `coordination::` outside `stdlib/coordination`
and the tests above.

## 5. Steps to create `Sounio-lang/sounio-loom` with history

Run these after owner approval. Use a scratch clone; nothing here touches
`Sounio-lang/sounio`.

1. Mirror and fetch the archive tags:
   ```bash
   git clone --mirror https://github.com/Sounio-lang/sounio.git loom-src.git && cd loom-src.git
   git fetch origin 'refs/tags/archive/2026-10-05/*:refs/tags/archive/2026-10-05/*'
   ```
2. Keep only the Loom refs plus `main`, and turn the tags into branches:
   ```bash
   keep='archive/2026-10-05/(.*loom.*|fix/github-lane-overlap)$'
   for t in $(git tag -l 'archive/2026-10-05/*' | grep -E "$keep"); do
     git update-ref "refs/heads/${t#archive/2026-10-05/}" "$t"; done
   git for-each-ref --format='%(refname)' refs/tags refs/pull | xargs -r -n1 git update-ref -d
   git for-each-ref --format='%(refname)' refs/heads \
     | grep -vE 'refs/heads/(main|.*loom.*|fix/github-lane-overlap)$' | xargs -r -n1 git update-ref -d
   ```
3. Write `loom-paths.txt` for `git filter-repo --paths-from-file`:
   ```text
   tools/loom/
   stdlib/coordination/
   formal/tla/SounioFleet.tla
   formal/tla/SounioFleet.cfg
   bin/sounio-coord
   bin/sounio-loom
   bin/sounio-fleet
   bin/sounio-agentd
   scripts/mcp/sounio_coord_mcp.py
   docs/internal/concepts/loom-
   .github/workflows/loom-coordination.yml
   .github/workflows/coord-github.yml
   glob:tests/compiler/loom_*
   glob:tests/compiler/causal_receipt*
   glob:tests/compiler/fleet_transaction*
   glob:tests/compile-fail/loom_*
   glob:scripts/ci/*loom*
   glob:scripts/ci/*coord*
   glob:scripts/ci/causal_receipt_typestate_gate.sh
   glob:scripts/ci/fleet_transaction_typestate_gate.sh
   glob:scripts/dev/*loom*
   glob:scripts/dev/*coord*
   glob:scripts/dev/*agentd*
   glob:scripts/dev/*fleet*
   ```
   Then run:
   ```bash
   git filter-repo --paths-from-file loom-paths.txt --prune-empty always
   ```
   The `~1,600`-commit branches carry the whole repository history of their
   time. `--prune-empty` drops every commit that does not touch these paths.
   Exclude `stdlib/coordination/spark_pair_*` with a second
   `--invert-paths --path-glob 'stdlib/coordination/spark_pair_*'` pass if
   the §4 decision keeps them out.
4. Build the default branch. Start from `codex/loom-message-bridge-runtime-20260903`.
   Cherry-pick the two `loom/segunda-ordem` commits (`8f3071f0a`, `3777730cc`
   before rewrite), then the four #2733 commits (`74e8ed641`, `5dd9c929f`,
   `eed993d09`, `3a0e2a74d`). Bring in the 40 `apple-delivery-ux` concept
   files, the 10 `grok-cli1` handshake files and `loom-language-authority.contract`
   as reviewed commits. Rename the result to `main`.
5. Keep every other branch as is, under `archive/` in the new repository.
   That includes `wip/loom-codex-1-uncommitted-20260921`, which needs a
   rebase (see §2).
6. Verify: `bin/sounio-coord status`, `scripts/ci/sounio_coord_selftest.sh`,
   the 4 typestate gates (with `SOUNIO_STDLIB_PATH` pointing at a `sounio`
   checkout), `dune build` in `tools/loom`.
7. Create `Sounio-lang/sounio-loom` (owner action) and push `main` and `archive/*`.
8. In `sounio`, one PR: replace or keep `bin/sounio-coord` per §4. Update
   `ci.yml:390`, `BRANCH_POLICY.md` and the code-review skill. Point
   `loom-obligation.contract` at the new repository. Leave the typestate
   modules, tests, gates and `loom-coordination.yml` untouched.

## 6. What can be closed after step 7

- PR #2154 (`lane/codex-1/loom-main-20260825`) and PR #2733 (`fix/github-lane-overlap`):
  close with a pointer to `sounio-loom`. #2733's `AGENTS.md`/`CLAUDE.md`
  hunks need a separate look, because they describe `sounio` policy.
- All 23 branches in §1. Their content is in the archive tags here and in
  `sounio-loom` history. Do not delete the tags.

## Appendix: reproduce the measurements

```bash
git fetch origin main 'refs/tags/archive/2026-10-05/*:refs/tags/archive/2026-10-05/*'
python3 - <<'EOF'
import subprocess, re
g=lambda *a: subprocess.run(['git',*a],capture_output=True,text=True).stdout
RX=re.compile(r'^(tools/loom/|stdlib/coordination/|formal/tla/|bin/sounio-(coord|loom|fleet|agentd)'
              r'|docs/internal/concepts/loom-|tests/.*(loom|causal_receipt|fleet_transaction|sounio_coord)'
              r'|scripts/.*(loom|coord|fleet|agentd|causal_receipt|fleet_transaction)|\.github/workflows/(loom|coord))')
refs=[t for t in g('tag','-l','archive/2026-10-05/*').split() if 'loom' in t or t.endswith('github-lane-overlap')]+['origin/main']
best={}
for ref in refs:
    tree={l.split('\t')[1]:l.split()[2] for l in g('ls-tree','-r',ref).splitlines() if RX.search(l.split('\t')[1])}
    for p,b in tree.items():
        ct=int(g('log','-1','--format=%ct',ref,'--',p) or 0)
        if p not in best or ct>best[p][0]: best[p]=(ct,ref,b)
for p,(ct,ref,b) in sorted(best.items()): print(p, ref, b[:12], sep='\t')
EOF
```
