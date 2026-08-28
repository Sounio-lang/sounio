<!-- docs:meta
topic_id: repo.docs.handoff.neurodyn-algebra-c-opus-coordination-prompt-2026-07-07
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.neurodyn-algebra-c-opus-coordination-prompt-2026-07-07
-->

# Prompt For Claude Code Opus: NeuroDyn Algebra-C Coordination

Paste the block below into Claude Code Opus.

```text
TASK: Assume theory/literature/critique ownership for NeuroDyn Algebra-C while Codex owns implementation/gates/execution.

Context:
- Repo/worktree: /workspace/sounio
- Current active branch must be verified live with:
  ./sounio-whereami --quick
  git status --short --branch
- Do not switch branches, reset, clean, rebase, or overwrite Codex work.
- Read before acting:
  ONBOARDING.md
  CLAUDE_HANDOFF.md
  AGENTS.md
  .claude/PARALLEL_BLOCKER_CONTRACT.md
  .claude/AGENT_OFFLOAD_POLICY.md
  docs/research/neurodyn_algebra_b_attribution_prereg_2026-07-06.md
  docs/research/neurodyn_ossm_sota_deep_research_2026-07-05.md
  docs/research/neurodyn_algebra_c_continuous_associator_prereg_2026-07-07.md

Background:
- Algebra-B binary attribution reached the algebraic attribution precondition:
  true O-SSM BA=55.892857, A8/H+H BA=51.785714, associative-projection O-SSM BA=52.946428.
- Algebra-B did NOT promote because pair-label null expansion failed early:
  null_08 O-SSM BA=57.857143 exceeded true O-SSM BA=55.892857 at 23/99 nulls.
- Final Algebra-B decision:
  ALGEBRA_B_ROUTE1_ATTRIBUTION_POSITIVE_BUT_NULLS_FAIL
- Claim boundary remains synthetic-only. No clinical, biomarker, biological,
  mechanistic, treatment-response, or broad O-SSM superiority claim.

New scientific direction:
- Algebra-C changes the endpoint from binary label accuracy to continuous
  associator fidelity.
- Locked question:
  Does O-SSM hidden dynamics preserve a continuous ground-truth associator/path-
  dependence observable better than A8/H+H, H-SSM, and associative-projection
  O-SSM under held-out splits and nulls?
- Primary endpoint proposed by Codex:
  held-out Spearman correlation between hidden/readout probe and continuous
  ground-truth associator scalar.
- Secondary endpoints:
  held-out R2, sign-AUC, calibration slope, high-magnitude ranking enrichment,
  collapse under associative projection.

Ownership split:
- Codex owns:
  implementation scripts, gates, Slurm runs, benchmark plumbing, artifact
  generation, executable checks, SHA/receipt validation.
- Opus owns:
  theory criticism, SOTA literature critique, reviewer attack surface, null
  validity critique, wording/claim boundary, and final conceptual framing.
- Do not edit Codex-owned files unless explicitly handed off.
- Prefer writing a new critique/handoff file, not modifying implementation.

Recommended Opus write set:
- docs/handoff/neurodyn_algebra_c_opus_critique_2026-07-07.md
- optionally docs/research/neurodyn_algebra_c_sota_review_2026-07-07.md

Do not edit without explicit transfer:
- examples/brain_ossm_abide.sio
- scripts/gpu/prepare_abide_campaign_snapshot.sh
- scripts/research/neurodyn_direct_slurm_smoke.sh
- scripts/research/neurodyn_algebra_b_decision_gate.py
- any future Codex-created Algebra-C gate/script

Your task:
1. Deeply critique the Algebra-C preregistration before any new smoke run.
2. Search current literature/SOTA as needed, using the repo's Context7/web policy
   where applicable and citing sources.
3. Answer these questions directly:
   - Is continuous associator fidelity a non-circular target?
   - Is the proposed primary endpoint, Spearman against ground-truth associator,
     sufficient? If not, what should replace or supplement it?
   - Are A8/H+H, H-SSM, associative projection, and raw probes sufficient controls?
   - What nulls are exchangeability-valid for a continuous target?
   - What nulls would be invalid or too weak?
   - What SOTA baselines would reviewers demand before a real-data bridge?
   - What exact claims remain disallowed even if Algebra-C passes?
   - What result pattern would convince you that O-SSM is measuring a real
     non-associative object rather than capacity/shortcut/noise?
4. Produce a concise decision memo with one of:
   - APPROVE_ALGEBRA_C_AS_PREREGISTERED
   - APPROVE_WITH_REQUIRED_EDITS
   - BLOCK_ALGEBRA_C_CIRCULAR_OR_UNDERCONTROLLED
5. If blocking, use .claude/PARALLEL_BLOCKER_CONTRACT.md and provide a formal
   blocker record.

Important:
- Be hostile to overclaim, not hostile to exploration.
- Treat Algebra-B null failure as real evidence, not an embarrassment to route around.
- Do not propose MDD/ADHD positive claims yet. Real-data bridge is blocked until
  Algebra-C passes or is explicitly abandoned.
- Do not expose secrets or credentials in prompts, files, logs, or offload payloads.

Report back with:
- files read
- files written
- decision
- required edits, if any
- literature/SOTA sources used
- claim-boundary warnings
- whether Codex may proceed to implementation
```

## Coordination Record

Lane: NeuroDyn Algebra-C continuous associator fidelity
Owner: Codex for execution; Claude Code Opus for critique/literature
Base: current `/workspace/sounio` branch after live `./sounio-whereami --quick`
Worktree: `/workspace/sounio`
Branch: live branch, currently expected to be `coord/lane-8c-dossier`
Codex Write-Set:
- `docs/research/neurodyn_algebra_c_continuous_associator_prereg_2026-07-07.md`
- future `scripts/research/neurodyn_algebra_c_*`
- future Algebra-C artifacts under `artifacts/research/neurodyn/synthetic/`
Opus Write-Set:
- `docs/handoff/neurodyn_algebra_c_opus_critique_2026-07-07.md`
- optional `docs/research/neurodyn_algebra_c_sota_review_2026-07-07.md`
Read-Set:
- Algebra-B prereg, SOTA note, Algebra-C prereg, Algebra-B artifacts, relevant scripts
Required-Gates:
- Opus critique archived or explicitly waived before Codex runs a new Algebra-C smoke
- mandatory offload review for any math-bearing claim before commit/submission
Merge-Target: none yet; research lane only
Known-Blockers:
- Algebra-C has not yet passed theory critique
- no continuous-fidelity implementation exists yet
- no clinical or mechanistic claim is allowed
