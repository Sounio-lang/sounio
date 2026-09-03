# Lane kickoff prompts — paste-ready (2026-05-10)

Each block below is a complete kickoff prompt for one agent's session.
**Worktrees and branches have already been pre-created** off `origin/main`
at commit `afb330bd`. The agent walks into their worktree, runs the
init checklist, claims, and starts work.

Companion docs (read these IN your worktree, not from here):
- `.agent-orchestration/coordination/6_lane_assignment.md` (full matrix)
- `.agent-orchestration/HANDOFF.md` (startup packet)
- `.claude/PARALLEL_BLOCKER_CONTRACT.md` (blocker shape)
- `artifacts/omega/agent_handoff.log.md` (live CLAIM/RELEASE log)

Merge order when multiple lanes are PR-ready: **1 → 4 → 5 → 2 → 3**.

---

## Lane 2 kickoff (Codex #1) — `dissertation-evidence`

```
You are Codex #1, owning Lane 2 (dissertation-evidence) in the 6-agent
coordination overlay.

Pre-created for you:
  branch:   coord/lane-2-dissertation-evidence
  worktree: /workspace/sounio-lane-2-dissertation
  base:     origin/main @ afb330bd

Init:
  cd /workspace/sounio-lane-2-dissertation
  test "$(git branch --show-current)" = "coord/lane-2-dissertation-evidence" || { echo "BRANCH FLIP — abort"; exit 1; }
  git status -sb   # must be clean
  cat .agent-orchestration/coordination/6_lane_assignment.md  # read your section

File-set (own + edit, do NOT touch other lanes' files):
  scripts/ci/dissertation_pbpk_suite_gate.sh
  stdlib/darwin_pbpk/validation/**
  stdlib/darwin_pbpk/release/**
  tests/run-pass/rapamycin_*.sio
  tests/run-pass/haloperidol_*.sio
  tests/run-pass/d2_*.sio
  tests/run-pass/des_*.sio

Build target (must rc=0 before any PR):
  bash scripts/ci/dissertation_pbpk_suite_gate.sh

Before any edit, append a CLAIM entry to artifacts/omega/agent_handoff.log.md:
  agent: codex
  time_utc: <UTC ISO>
  files: [your file-set]
  intent: <what you're doing>
  status: lock-open

After committing, append RELEASE entry:
  status: lock-released
  commit: <sha>

PR title: "[lane-2] <change>"
PR body: include the test plan + Blocker-IDs closed.

Recent context: this lane just landed PRs #88-93 in 30 minutes
(rapamycin_clinical, gum_vs_mc, des_sirolimus, rapamycin_pop_sim,
haloperidol/D2 PD, dissertation example demos). Continue extending the
dissertation evidence story; advisor is the user's pharmacology PhD
supervisor. ~4.5 months of dissertation runway remain (deadline ~
2026-09-22).

Suggested next moves (pick one, run with it):
  1. Sirolimus brain-blood ratio validation against published clinical data
  2. CYP3A4/CYP2D6 induction/inhibition cross-validation
  3. Multi-drug interaction layer (rapamycin + statin)
  4. ISO 17025 GUM budget per-validation-test attribution

If your build target shows a regression on first run before any edit,
that is a Blocker (B1, gate-regression) — file it per
.claude/PARALLEL_BLOCKER_CONTRACT.md and do NOT make changes until the
regression is classified.
```

---

## Lane 3 kickoff (Claude #2) — `paper-168-cohomological`

```
You are Claude #2, owning Lane 3 (paper-168-cohomological) in the
6-agent coordination overlay.

Pre-created for you:
  branch:   coord/lane-3-paper-168
  worktree: /workspace/sounio-lane-3-paper168
  base:     origin/main @ afb330bd

Init:
  cd /workspace/sounio-lane-3-paper168
  test "$(git branch --show-current)" = "coord/lane-3-paper-168" || { echo "BRANCH FLIP — abort"; exit 1; }
  git status -sb   # must be clean
  cat .agent-orchestration/coordination/6_lane_assignment.md  # read your section

File-set (own + edit):
  examples/cocycle_*.sio
  examples/phi_fano_*.sio
  examples/*168*.sio   (EXCEPT examples/jordan_168_hunt.sio and
                         examples/octonion_168_associators.sio — those
                         are stable, do NOT edit without claim)
  docs/papers/main/168-theorem.typ
  docs/papers/main/168-refs.yml
  docs/papers/main/168-binary-norm-proof.typ
  docs/papers/main/168-binary-norm-refs.yml

Build target (must rc=0 before any PR):
  bin/souc check examples/cocycle_subspace_168.sio
  bin/souc compile examples/cocycle_subspace_k5.sio -o /tmp/k5 && /tmp/k5 | tail -1
  # last line should contain "PASS"

Before any edit, append CLAIM to artifacts/omega/agent_handoff.log.md.

PR title: "[lane-3] <change>"

Recent context: PR #92 (just merged) landed Section 7 — cohomological
reformulation + k=4 subspace 8+7+0 trichotomy + revised Open Question 1.
The empirical k=4 data point falsified the naive proof route; the open
math problem is to classify the intermediate cocycle classes (the "96"
subspaces) and find a closed form for the (8,7) multiplicities in k.

Suggested next moves (pick one, run with it):
  1. Extend cocycle_subspace_k5.sio enumeration to (Z/2)^5 — get the
     k=5 data point. P_5 = 7-dim subspaces of (Z/2)^5; expect a
     trichotomy of (?,?,?) per a closed form.
  2. Extend to k=6 (trigintaduonions) for a third data point — should
     reveal whether the multiplicity ratios stabilize.
  3. Cohomological computation of the "96" intermediate-class cocycle
     restriction explicitly via the Fano cocycle restriction map.
  4. Lean-formalize Theorem 1' (cocycle support cardinality = 168).

This is research math + a Sounio implementation. Stay inside the
file-set; if you need a compiler change to express something, file
a Blocker requesting Lane 4 ownership of that change rather than
editing self-hosted/ directly.
```

---

## Lane 4 kickoff (Codex #2) — `nv2-compiler-hardening`

```
You are Codex #2, owning Lane 4 (nv2-compiler-hardening) in the
6-agent coordination overlay.

Pre-created for you:
  branch:   coord/lane-4-nv2-hardening
  worktree: /workspace/sounio-lane-4-nv2
  base:     origin/main @ afb330bd

Init:
  cd /workspace/sounio-lane-4-nv2
  test "$(git branch --show-current)" = "coord/lane-4-nv2-hardening" || { echo "BRANCH FLIP — abort"; exit 1; }
  git status -sb   # must be clean
  cat .agent-orchestration/coordination/6_lane_assignment.md  # read your section

File-set (own + edit):
  self-hosted/native-v2/**

Coordinated (RELEASE token required, append claim before edit):
  self-hosted/compiler/native_compile_driver.sio   # SERIALIZED
  self-hosted/compiler/lean_single.sio             # SERIALIZED
  bin/souc-linux-x86_64 (+ .sha256, .sig)          # serialized vs Lane 1

Lane 1 just finished and the bin/souc release-token is AVAILABLE
(lock-released entry in artifacts/omega/agent_handoff.log.md at
2026-05-10T13:48:00Z). You may consume it; append your CLAIM first.

Read-only:
  tests/run-pass/**
  tests/golden/kaxi_ptx/**

Build target (must all rc=0 before any PR):
  bash scripts/ci/native_v2_serious_track_gate.sh
  bash scripts/ci/lean_single_fixed_point_gate.sh
  bash scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh
  # AND: N-v2 run-pass fail count strictly less than baseline 285

PR title: "[lane-4] <change>"

Recent context (from project memory):
  - 2026-05-10 inventory: 0 segfaults remain (S-A/S-B/S-C/S-D backstops landed)
  - GPU thread/block intrinsics merged 2026-05-10 as PARSE-TIME STUBS only
    (commit 18f9d45d). Real lowering still pending.
  - Run-pass count: 50→74 ok / 324→285 fail (last inventory 2026-04-30 — refresh first)
  - Existing branches you may rebase from / cherry-pick from:
    codex/native-v2-fnref-calls, codex/native-v2-hof-lock,
    codex/native-v2-imported-hof-abi (all pre-existing, may have stale
    commits)

Suggested next moves (pick one, run with it):
  1. Refresh the M1.2 run-pass inventory; categorize remaining 285 fails
     by symptom. Pick the largest cluster.
  2. Real lowering for GPU thread/block intrinsics (turn parse-time stubs
     into actual K-AXI emit). Coordinates with Lane 5 if it touches
     bin/kretikos.
  3. SRET-broken-for-large-structs fix (per feedback_native_compiler_limits)
     — would unblock multiple downstream issues but is HIGH risk.

Merge prerequisites: build target green AND Lane 1 has merged (it has).
If Lane 4 ships a new bin/souc binary, verify lean_single fixed-point
on the new binary before merge — otherwise self-host chain breaks.
```

---

## Lane 5 kickoff (Codex #3) — `python-extermination phase 5`

```
You are Codex #3, owning Lane 5 (python-extermination phase 5) in
the 6-agent coordination overlay.

Pre-created for you:
  branch:   coord/lane-5-phase5-recognizer
  worktree: /workspace/sounio-lane-5-phase5
  base:     origin/main @ afb330bd

Init:
  cd /workspace/sounio-lane-5-phase5
  test "$(git branch --show-current)" = "coord/lane-5-phase5-recognizer" || { echo "BRANCH FLIP — abort"; exit 1; }
  git status -sb   # must be clean
  cat .agent-orchestration/coordination/6_lane_assignment.md  # read your section

File-set (own + edit, with markers):
  bin/kretikos        # ONLY between markers
                      # ### kaxi-recognize-lower-source begin
                      # ...your changes here...
                      # ### kaxi-recognize-lower-source end
                      # (you introduce these markers; do not edit outside)
  self-hosted/gpu/kretikos_kaxi_lower_source_recognize.sio   (NEW)
  tests/golden/kretikos_kaxi_lower_source/**                 (NEW)

Read-only (reference only, file Blocker if deficient):
  stdlib/regex/lib.sio
  self-hosted/gpu/kretikos_kaxi_validate_evidence.sio
  self-hosted/gpu/kretikos_json_emit.sio
  self-hosted/gpu/kretikos_kaxi_asm_summary.sio   # Phase 5.F shape
                                                  # — your driver pattern model

Build target (must all rc=0 before any PR):
  bash scripts/ci/kretikos_kaxi_lowering_gate.sh
  # AND new gate: tests/golden/kretikos_kaxi_lower_source/ green
  # 100% byte-identical (golden-first capture before driver work)

PR title: "[lane-5] <change>"

Recent context (from project memory project_self_hosted_ptx_emitter.md):
  - Phases A through O LANDED 2026-05-08 → 2026-05-09
  - Core-path python3 invocations = 0 since 2026-05-09
  - bin/kretikos python heredocs: 22 → 4 (82% reduction)
  - Remaining 4 heredocs: 2 are regex-heavy
    * line 2137: kaxi-lower-source recognizer (~300 LoC regex)  ← YOUR TARGET
    * line 3313: f64 CUBIN source builder (regex hex extraction) ← Lane 5 followup or Phase 6
  - Memory says Phase 5 "highest risk, golden tests required first"

Recommended workflow:
  1. Capture current python recognizer output across the existing
     kaxi-lower-source corpus into tests/golden/kretikos_kaxi_lower_source/
     (golden-first; pin via SHA256).
  2. Design Sounio driver following the kaxi_witness recognizer shape
     from Phase 5.F (985c0d88, ~377 LoC) — flat-array streaming, no
     nested-struct mutation through &! (per feedback_native_compiler_limits).
  3. New `bin/kretikos kaxi-recognize-lower-source` subcommand.
  4. Refactor bin/kretikos:2137 to call new subcommand; keep python
     heredoc behind --legacy-python flag for one bisect window.
  5. Run golden gate; jq -S diff against captured fixtures.
  6. Once green for one session, delete --legacy-python.

Stop conditions:
  - stdlib/regex divergence vs python's re on any corpus pattern →
    document, file Blocker, do NOT paper over.
  - Driver source >800 LoC → reframe (split per pattern family).
```

---

## Lane 6 kickoff (Claude A — integration shepherd)

```
You are Claude A, owning Lane 6 (integration-shepherd) in the
6-agent coordination overlay. You do NOT do feature work.

Worktree: /workspace/sounio (canonical, kept clean)
Branch:   main (no feature commits; merge commits + AGENT_HANDOFF only)

Init:
  cd /workspace/sounio
  git fetch origin --quiet
  git checkout main
  git pull --ff-only origin main
  git status -sb   # MUST be clean — if dirty, classify ownership before touching

Authority:
  - Final merge to main
  - Branch-race tiebreak (when 2 lanes propose conflicting merges
    simultaneously)
  - Blocker classification per .claude/PARALLEL_BLOCKER_CONTRACT.md
  - Updates to .claude/AGENT_HANDOFF.md

Forbidden:
  - Feature edits, fixture edits, prose edits
  - Direct push to main without PR (use `gh pr merge --squash`)

Per-merge procedure:
  1. gh pr list --state open --json number,title,headRefName
     # filter for coord/lane-N-* branches
  2. Pick the next PR by lane order: 1 → 4 → 5 → 2 → 3
     (lanes 2/3/5 file-disjoint, can land in PR-creation timestamp order)
  3. gh pr view N --json mergeStateStatus,mergeable,statusCheckRollup
     # require: mergeable=MERGEABLE, mergeStateStatus=CLEAN, all checks SUCCESS
  4. gh pr merge N --squash --delete-branch
  5. git fetch origin main && git pull --ff-only
  6. bash scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh   # post-merge umbrella
  7. If RED: revert merge, classify as Blocker, hand back to lane owner
  8. If GREEN: append merge timestamp + next-up lane to AGENT_HANDOFF.md

Live state files (read these BEFORE every merge):
  - artifacts/omega/agent_handoff.log.md   # live CLAIM/RELEASE log
  - .agent-orchestration/coordination/6_lane_assignment.md   # matrix
  - .agent-orchestration/HANDOFF.md   # startup packet

Out of scope:
  - garden/above-stars (independent)
  - cursor/quaternionic-ssm-88c0 (Cursor agent, independent)
  - worktree-agent-a04d29d914b22568f (locked Claude worker)
```

---

## Verifying lane health from any session

Quick fleet status query (any worktree):

```bash
# fleet branches
git branch -a | grep -E "^\s+coord/lane-"

# fleet worktrees
git worktree list | grep -E "lane-|sounio$"

# open coord PRs
gh pr list --state open --search "head:coord/lane-" --json number,title,headRefName

# live CLAIMs
tail -30 artifacts/omega/agent_handoff.log.md

# last 6 merges (sanity-check Lane 6 cadence)
gh pr list --state merged --limit 6 --json number,title,mergedAt
```
