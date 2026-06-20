<!-- docs:meta
topic_id: repo.docs.serious-language.command-center
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.serious-language.command-center
-->

# Serious-Language Command Center

> Snapshot: 2026-06-20. Use this as the operating board for making Sounio legible as a real programming language without overclaiming any surface.

Sounio should be handled as a programming language product, not as one long compiler experiment. That means the work is split into stable lanes with explicit promotion gates, evidence, and public wording boundaries.

This document is not a new claim authority. It is an operating board. Public
wording still comes from:

- `readiness-ledger.md`
- `public-claim-registry.v1.tsv`
- `real-world-defensibility.md`

If this command center conflicts with those files, those files win.

When those documents say "self-hosted" or "compiler path", read that through the
registered evidence and downgrades. In particular, current command-center state
does not authorize source-swap wording: lean_single remains part of the checked
binary-source story, and the modular Madaros rebuild diagnostic is still open.

## Operating Posture

1. Keep `main` as the public truth line.
2. Keep dirty primary worktrees out of release decisions.
3. Promote only small, evidence-backed slices.
4. Treat stale raw compiler binaries as non-evidence.
5. Keep broad claims in `docs/serious-language/public-claim-registry.v1.tsv`.
6. Use the website and README as claim surfaces, not as wish lists.

Definitions:

- `green for launcher`: the public wrapper/launcher path passed its named gates
  for the checked artifact it selects.
- `amber`: at least one narrow path is green, but a neighboring parity,
  rebuild, or research lane has an open diagnostic.
- `failing diagnostic`: a separate proof or parity lane has a known failure and
  must not be cited as closed evidence.
- `bounded`: evidence is scoped to the named gate, manifest, fixture set, or
  registry row; it is not a broad product claim.
- `source-swap readiness`: the maintained modular compiler source can replace
  the current checked binary-source story without weakening conformance,
  bootstrap, artifact identity, or public claims.
- `stale raw ELF`: any raw compiler ELF whose build commit, SHA, or source tree
  cannot be matched to current `HEAD` or to the named commit under review.

## Current Reality Check

| Surface | Status | Evidence | Evidence for | Next action |
|---|---|---|---|---|
| Public website | green | PR #327 merged; Vercel production deployment succeeded on merge commit `1689e2d7a` | website deployment only | Keep the Living Language Pulse tied to generated artifact status. |
| Madaros compiler system | amber: launcher green, raw rebuild diagnostic open | PR #328 merged; `docs/MADAROS_STATUS.md`; reported post-merge rebuild diagnostic not revalidated in this lane | launcher evidence only; source-rebuild parity is not closed | Sync stale worktrees before debugging compiler failures. Do not infer broad clinical/science correctness from launcher health. |
| Epistemic and formal evidence | active, bounded | `readiness-ledger.md`; `package_pbpk_gum_gate.sh`; `serious_language_conformance_gate.sh`; `lean_proof_status_audit.py`; Lean files named in the ledger | named epistemic/GUM workflows, bounded conformance rows, and exact Lean audit status | Cite exact gate rows before using these claims externally. This row does not claim full type soundness. |
| Serious-language claim system | active; rerun gates before promotion | `readiness-ledger.md`, `public-claim-registry.v1.tsv`, claim-closure and spec-drift gates | claim governance and public wording controls | Keep new public claims registered before promotion. |
| Primary checkout hygiene | dirty | live `git status` shows `.beagle`, MCP, compiler, audit, Slurm, and handoff WIP | local workspace risk only | Triage by lane; do not bulk-stage. |
| Open PR queue | mixed | PRs #308 green, #313/#297/#287/#239 failing, #296/#232 drafts, #226 targets integration base | integration queue state | Process by readiness class, not by age. |

## Lane Model

| Lane | Status now | Purpose | Gate to run before promotion | Public wording boundary |
|---|---|---|---|---|
| L0: repo hygiene | open | Keep agent contracts, docs registry, generated context, and WIP isolation sane | `git diff --check`; docs registry/consistency gates when docs change | No language capability claim. |
| L1: checked compiler entry | amber overall; green for launcher | Preserve `bin/souc` and `bin/madaros` as user-facing compiler entrypoints | `bash scripts/ci/madaros_operational_contract_gate.sh`; `make madaros-full-gate` when compiler changes | Claim the checked launcher, not every raw artifact or source-rebuild parity. |
| L2: modular compiler parity | failing diagnostic | Move growth away from single-file monolith toward modular source parity | named Madaros/full self-host gates plus clean-source rebuild evidence | Do not claim source-swap readiness until parity gates prove it. Current raw rebuild SIGSEGV keeps this lane diagnostic-open. |
| L3: language conformance | bounded | Tie syntax, type system, effects, modules, ownership, and epistemic behavior to executable cases | `bash scripts/ci/serious_language_conformance_gate.sh` | Bounded conformance, not complete spec proof. |
| L4: stdlib and scientific core | bounded | Separate active callable modules from stubs and research packages | stdlib reliability, hyper execution, science pipeline gates, and domain gates such as `package_pbpk_gum_gate.sh` for package/import and PBPK/GUM workflow markers | Do not quote broad stdlib completeness from narrow gates. Clinical or pharmacology behavior needs independent domain validation. |
| L5: tooling and LLM target | prototype | Make `souc check`, JSON diagnostics, MCP, and error catalog dependable | MCP tests, diagnostic schema validation, check-exit probes | Prototype tooling until CLI and diagnostics are contract-clean. |
| L6: public surface | active | Website, README, serious-language docs, paper bundle | website build, claim-closure, spec-drift, offload review for external artifacts | Every claim must cite a gate, registry row, or downgrade. |

## Immediate Queue

### A. Hygiene First

Owner: Codex. Priority: P0. Window: next 48 hours.

Purpose: stop the worktree from lying.

Required actions:

1. Keep the dirty primary checkout untouched until each change is classified.
2. Compare local WIP against `origin/main` before staging anything.
3. Move compiler fixes into isolated worktrees.
4. Keep `.beagle/context/*` and generated handoff logs out of unrelated commits.

Acceptance:

```bash
git status --short --branch
git worktree list
git diff --check
```

Failure condition: if intended work and unrelated WIP appear in the same staged
set, stop and split the lane before committing.

### B. MCP / LLM-Codegen Contract

Owner: Codex. Priority: P1. Window: after A, before new MCP claims.

Purpose: make Sounio safer as a codegen target.

Observed local WIP:

- `tools/mcp/sounio_mcp/check.py`
- `tools/mcp/sounio_mcp/test.py`
- `tools/mcp/tests/test_loop.py`
- `tools/mcp/tests/test_tools.py`

Likely intent:

- pass `--json` after the source path for current `souc check` behavior;
- make MCP tests resolve repo-root paths instead of relying on caller CWD.

Acceptance:

```bash
cd tools/mcp
python3 -m pip install -e '.[dev]'
pytest tests/
```

If the MCP test environment is not self-contained, document the missing dependency and add the smallest fixture or path fix that makes it reproducible.

Failure condition: if pytest cannot run from `tools/mcp` after installing
`.[dev]`, do not merge MCP changes until the setup path is documented or fixed.

### C. Madaros Raw Rebuild Diagnostic

Owner: compiler lane. Priority: P1. Window: after A, before any modular-source-swap wording.

Purpose: separate production launcher health from source rebuild parity.

Current public fact:

- Madaros production launcher path is green on `origin/main`.
- Raw/source rebuild parity still has a reported local SIGSEGV diagnostic.
- This lane is not clinical, pharmacology, or scientific validity evidence.

Acceptance:

```bash
bash scripts/ci/madaros_operational_contract_gate.sh
MADAROS_RAW_BIN=bin/madaros-linux-x86_64 bash scripts/ci/madaros_full_gate.sh
```

For source rebuild parity, use an isolated worktree and record the exact seed compiler, raw artifact path, stack limit, and signal.

Failure condition: if a proof depends on a stale raw ELF or cannot identify the
seed compiler and artifact SHA, do not promote the result.

### D. Stale PR Queue

Owner: integration shepherd. Priority: P1 for green hygiene, P2 for research/domain branches.

Purpose: reduce noise without pretending old branches are current.

Current queue classes:

| PR | Class | Action |
|---|---|---|
| #308 `chore/repo-hygiene` | green hygiene candidate as of 2026-06-19 CI, not revalidated after #328 | Re-read diff against current `main`; merge only if still semantically clean. |
| #313 SRET regression | failing compiler PR | Rebase or close in favor of newer Madaros raw rebuild diagnostic if superseded. |
| #297 PBPK28 tissue composition | failing science/domain PR | Quarantine: no merge until failing checks are resolved, clinical/science offload is logged, and narrowed validation is attached. Close if superseded. |
| #287 affine octonion correlation | failing research PR | Keep as research/prototype until full suite failure is classified. |
| #296 Madaros main proof | draft, likely stale | Reconcile with merged PR #328 and close or refresh. |
| #232 nested mut write fix | draft, divergent | Reclassify as compiler repair lane; do not merge as-is. |
| #226 Erdős-Straus GPU sieve | targets integration branch and failing | Keep off main until target/base and GPU evidence are current. |

### E. Public Claim Hygiene

Owner: release/documentation lane. Priority: P0 for external artifacts, P2 for internal notes.

Purpose: preserve credibility.

Every new public statement about Sounio as a PL must answer:

1. What exact behavior is claimed?
2. Which gate, test, artifact, or registry row supports it?
3. What is explicitly not claimed?
4. Which external-facing review is required before publication?

In this repo, "offload review" means a recorded `bin/llm-offload` review under
the policy in `.claude/AGENT_OFFLOAD_POLICY.md`. Acceptable outcome means every
actionable factual or scope issue is either fixed, explicitly downgraded, or
recorded as residual risk. It is an orthogonal LLM review gate, not a substitute
for clinical, mathematical, or human domain sign-off.

Failure condition: if a public-facing doc introduces a new claim not covered by
the registry, downgrade the claim or register it before merge.

Clinical/pharmacology note: `package_pbpk_gum_gate.sh` is treated here as a
package/import, observed-PETAB marker, and PBPK/GUM workflow gate. It is not a
clinical validity claim and must not be used for dosing or care guidance.

Acceptance:

```bash
bash scripts/ci/serious_language_claim_closure_gate.sh
bash scripts/ci/serious_language_spec_drift_gate.sh
```

## First Three Release Packets

Do these as separate commits or PRs.

1. **MCP contract cleanup**
   - Scope: `tools/mcp/**` only.
   - Goal: robust `souc check --json` invocation and cwd-independent tests.
   - Gate: MCP pytest plus `git diff --check`.

2. **Command-center/hygiene docs**
   - Scope: `docs/serious-language/**`, `docs/governance/topic-registry.v1.json`.
   - Goal: make the serious-language operating board visible and registered.
   - Gate: docs registry and consistency checks plus external-facing offload review.
   - Dependency: independent of MCP; it may land first if docs gates pass.

3. **Madaros raw rebuild diagnostic**
   - Scope: compiler lane only, isolated worktree.
   - Goal: reproduce or retire the local source rebuild SIGSEGV without weakening production launcher claims.
   - Gate: operational contract, full gate, and exact diagnostic artifact.
   - Failure condition: source rebuild still crashes, or the reproduced crash lacks an artifact path, seed compiler, and signal.

## Stop Conditions

Stop and reclassify rather than pushing if:

- a change touches `bin/souc`, `bin/madaros`, compiler source, and public docs in the same commit;
- a proof depends on a stale raw ELF;
- a green website deploy is being used as evidence for compiler behavior;
- a research result is being promoted without a named gate and exact artifact;
- clinical, math, or external-facing claims are about to ship without the required offload review.
