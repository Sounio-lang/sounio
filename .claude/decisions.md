# Sounio Decisions (Reconstructed)

Status: reconstructed on `2026-04-05`, not recovered verbatim from the old VM.

Why this file exists:

- the old VM backup contained substantial Claude state in `projects/`, `plans/`,
  `sessions/`, `session-env/`, and `file-history`
- the exact file `.claude/decisions.md` did not exist in the VM backup
- this file captures the durable decisions that were clearly present in the
  archived handoff/history docs and the recovered execution contract

Primary reconstruction sources:

1. `CLAUDE_HANDOFF.md`
2. `docs/archived/HANDOFF.md`
3. `docs/archived/HISTORY.md`
4. `.claude/OPERATIONAL_CANONICAL_INDEX.md`
5. `.claude/PLAN_CANONICAL_EXECUTION.md`
6. VM backup evidence in:
   - `/srv/workspaces/devsounio/.migration/sounio-dev-01/home-configs-20260403T000000Z.tar.gz`

## Durable decisions

### 1. Sounio is operating from a recovered state, not a fresh clone

- The codebase was recovered from VM `sounio-dev-01` tarballs.
- The tarball did not include `.git`; Git identity and safe branches were
  rebuilt afterwards.
- Treat the recovered state as intentional lineage, not accidental debris.

### 2. `integration/sounio-dev-ready-base` is the safe active branch

- Do not treat `main` as the default work branch for current development.
- Do not casually rebase, reset, clean, or align the recovered tree to
  `origin/main`.
- Preserve recovery lineage:
  - `recovery/sounio-dev-01-import`
  - `recovery/sounio-dev-01-snapshot-20260405`

### 3. The environment is remote-first

- The active execution surface is the promoted remote workspace, not the laptop.
- Current active remote repo path:
  - `/workspace/sounio`
- The laptop is a control surface and reconnection client, not the sole source
  of truth for live work.

### 4. Canonical execution precedence is fail-closed

- The active operational contract is defined by:
  - `.claude/PLAN_CANONICAL_EXECUTION.md`
  - `.claude/OPERATIONAL_CANONICAL_INDEX.md`
  - `.claude/PROMPT_EXECUTION_CONTRACT.md`
- Historical docs may inform context, but they do not override the current
  canonical precedence.

### 5. Track A / Track B cutover status is already green

- The no-rust cutover and locked Track B sequence are treated as achieved,
  evidenced states.
- The locked Track B order must not be reordered:
  1. `data_structures.md`
  2. `gpu_ir_expansion.md`
  3. `hlir_lowering.md`
  4. `metal_msl_codegen.md`
  5. `ptx_regalloc_expansion.md`

### 6. The self-hosted compiler path is authoritative

- `bin/souc` is the canonical compiler entrypoint for operational checks.
- Use `scripts/lib/resolve_souc.sh` as the canonical compiler-resolution path
  instead of inventing ad hoc routing.
- Use `scripts/run_sio_test_suite.sh` as the canonical harness for SIO tests.

### 7. `native-v2` is not yet fully honest

- The next real technical gap is not frontend/bootstrap parity.
- The unresolved gap is graduating the preview `native-v2` backend from the
  narrow cached `triangle_basic` proof to a general self-hosted native path.
- The `native-v2-shadow` alias is transitional and should only be retired once
  equivalent runtime/codegen coverage exists.

### 8. Evidence must move with behavior changes

- If behavior changes, update the relevant gate evidence in the same change-set.
- Sensitive control files should not be edited in parallel with other work that
  depends on them.
- `self-hosted/check/check.sio` remains a high-risk merge surface.

### 9. Archived lineage is useful, but must stay archived

- `docs/archived/HANDOFF.md` and `docs/archived/HISTORY.md` are preserved for
  lineage.
- They describe real prior milestones, but should not silently override the
  current recovery/integration reality.

### 10. The old Claude VM state existed in another format

- The old VM backup did not contain:
  - `.claude/decisions.md`
  - `.claude/pending.md`
  - `.claude/session_state.json`
- It did contain substantial persisted state in:
  - `.claude/projects/`
  - `.claude/plans/`
  - `.claude/sessions/`
  - `.claude/session-env/`
  - `.claude/file-history/`
- Any future reconstruction should treat those directories as the historical
  evidence base.

## Practical operating rules

1. Confirm branch before editing.
2. Preserve recovery/integration lineage.
3. Prefer incremental changes over repo-wide cleanup.
4. Do not claim success from historical docs alone; validate against the current
   repo and current runtime.
5. When in doubt, trust the current repo state plus `CLAUDE_HANDOFF.md`.

---

## Wave 9 decisions

### 11. ODEP naming — open issue, deferred to submission time

**Context.** Paper E proposes "Oblivious Differential Epistemic Privacy"
(ODEP) as the cryptographic counterpart of `ExactlyPrivate<T>`. The
external reviewer (2026-04-30) correctly flagged that the phrase
"differential epistemic privacy" is likely to be read by DP-literate
reviewers as an extension of Dwork-Roth ε-differential privacy, which
it is not: ODEP is exact (algebraic), not ε-bounded (probabilistic).

**Candidate names.**

| Name | Signal | Cost |
|------|--------|------|
| ODEP (current) | Echoes "differential privacy" — easier to onboard DP audiences | Risks mis-framing as ε-DP extension |
| AEP  | "Algebraic Epistemic Privacy" — accurate, neutral | Loses rhetorical hook |
| ZEP  | "Zero-divisor Epistemic Privacy" — explicit about mechanism | Jargon-heavy for non-algebraists |
| EZDP | "Exact Zero-Divisor Privacy" — communicates exactness + mechanism | Four letters |
| KEP  | "Kernel Epistemic Privacy" — emphasises the annihilator kernel | Generic "kernel" is overloaded |

**Decision.** Deferred.  We keep ODEP in the current Paper E draft
because the name has no operational consequence inside the repo (no
API, no filename, no test), and renaming before we know the target
venue would be speculative.  At submission time (whichever venue),
the name is re-evaluated and whichever abstract-friendly form best
fits that venue's reviewer pool is chosen.

**Trigger to revisit.** First decision about a submission venue for
Paper E.  Whoever revisits this should also skim the abstract of
Paper E against the then-current DP literature and decide in one
turn.

**Artifacts to update at rename time.**  Only the paper (`paper/paper_e_odep.tex`)
and the ODEP prover directory name (`tools/odep-prover/`) are named
after ODEP today.  No Lean theorem, no Sounio type, no `.claude/`
session state depends on the name.  Rename impact is therefore
bounded and cheap.

### 12. `tests/parity/` landed 2026-04-30

**Context.**  The reviewer flagged risk of silent divergence between
`lean_single.sio` and the modular self-hosted tree.

**Decision.**  Introduced `tests/parity/` as a golden-regression
harness today (no alternate build needed), and extended it with a
`--parity PATH` mode that will become useful the day the modular
build produces a working binary.  The harness also happens to surface
a genuine wrapper bug: `bin/souc check` swallows the compiler's exit
code.  That is recorded in `tests/parity/README.md` and is *not*
patched in the same commit — wrapper fix is a separable task.

### 13. AMI↔Surgical Calculus duality corollary — added to Paper D

**Context.**  External reviewer observed that the 168-basis
(interpretability) and the Surgical Calculus (intervention) consume
the same verified Lean object (`SounioZeroDivisorBridge.lean`) and
that the duality between them is not articulated in any individual
paper.

**Decision.**  Added subsection §"Corollary: the
identification/intervention duality" to `paper/paper_d_ami.tex`
stating the observation as structural, not a new theorem, and
formulating a conjecture that this pattern generalises beyond
sedenions.  No new Lean obligation incurred; a full Lean statement
of the duality is deferred to future work.

### 14. Metron — name of the EISA surface language (2026-07-05)

**Context.**  The EISA stack (E0–E5, complete and gated) has an unnamed
surface language: the `epistemic fn`/`let`/`gate`/`store` subset that
`stdlib/eisa/backend.sio` compiles to `.eisax`.  Papers need a name;
"the restricted epistemic subset" does not survive contact with one.

**Decision.**  Operator approved **Metron** (μέτρον, "measure"), chosen
for future adoption: two syllables, stable pronunciation across
languages, weak collisions, and a one-line pitch — *computation as
measurement* — that anchors in the project's metrological identity
(GUM, EFT-measured roundoff, receipts, SI units).  Rejected: Martyria
(evokes *martyr*), Gnomon (unstable pronunciation), Doxa (adjacent to
*doxxing*), Tekmor (unique but opaque).  Corollary: external-facing
text says **Metron VM (MVM)**, never bare "EVM" (saturated by
Ethereum).  Internal identifiers unchanged in v1.  Recorded in
`docs/research/eisa-v1-plan-2026-07-05.md` §1 and the architecture doc.

### 15. EISA v1 closes with V1e partial; Rump moves to v2/qd128 (2026-07-05)

**Context.**  The v1 plan's V1e row promised the full Rump 1988 kernel
under v1 budgets with dd64 honest-boundary framing.  The drafted bridge
lane segfaulted into the unfinished high-register arithmetic templates
and was rolled back rather than shipped fragile.

**Decision.**  V1e ships S1–S3 (fixed-point loop with pinned fuel and
derivable frail=1; frail-cancellation; emov −0.0 via division witness);
the Rump showcase moves to EISA v2 on the qd128 err lane, where the
receipt can pin the exact −54767/66192.  Math-review corollary
(Grok, 2026-07-05): the v2 corpus must keep a **dd64-failure lane
alongside the qd128 success lane** — the receipt showing dd64 visibly
failing at the ~122-bit cancellation is standalone boundary evidence,
not superseded by the qd success.  Bridge v1 residuals (full-arithmetic
e16..e63 templates, fuel-stop high-reg receipt) are prerequisites for
the v2 Rump lane and land with the W4 template refactor.

### 16. EISA work moved to dedicated worktree (2026-07-05)

**Context.**  A concurrent agent switched `/workspace/sounio` to
`coord/lane-8c-dossier` mid-audit, stashing the uncommitted V1e files
(recovered from `stash@{0}` without touching the shared checkout).

**Decision.**  The EISA track now operates from the dedicated worktree
`/workspace/sounio-eisa` on `gpu/epistemic-tensor-core-next`, per the
one-worktree-per-agent rule in CLAUDE.md §4.  Shared-checkout EISA
edits are over.
