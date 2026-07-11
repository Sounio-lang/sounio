# Claude Handoff: Sounio From VM To Remote-First Workspace

## Read This First

You are operating in a recovered-and-promoted Sounio environment.
Do not assume a fresh clone or a laptop-first workflow.

Inside the promoted Kubernetes workspace, start with:

```bash
cd /workspace/sounio
./sounio-whereami --quick
```

The current checked-out branch is the operational truth for that session. Do
not switch branches just because older host-side docs mention
`integration/sounio-dev-ready-base`.

Read these files in this order before making non-trivial changes:

1. `ONBOARDING.md`
2. `CLAUDE_HANDOFF.md`
3. `README.md`
4. `AGENTS.md`
5. `CLAUDE.md`
6. `docs/archived/HANDOFF.md`
7. `docs/archived/HISTORY.md`
8. `docs/guide/MINIMUM_VIABLE_SOUNIO.md`
9. `docs/guide/LLM_PROGRAMMING_GUIDE.md`

If the host-level control docs are also visible in the environment, consult these next:

- `/home/devsounio/README.md`
- `/home/devsounio/AGENTS.md`
- `/home/devsounio/CLAUDE.md`
- `/home/devsounio/projects/sounio/README.md`
- `/home/devsounio/projects/sounio/PLAN.md`
- `/home/devsounio/projects/sounio/IMPORTS.md`
- `/home/devsounio/projects/sounio/DEV_WORKFLOW.md`
- `/home/devsounio/projects/sounio/WORKSPACE_K8S.md`

## What Happened

### 1. The old VM was abandoned

The previous active development environment was a VM called `sounio-dev-01`.
We intentionally moved away from that VM and recovered the project state from exported tarballs.

The key import artifact was:

- `/srv/workspaces/devsounio/.migration/sounio-dev-01/sounio-main-working-tree-20260403T000000Z.tar.gz`

Important note:

- the tarball did **not** contain `.git`
- the working tree was restored first
- Git identity and branch safety had to be rebuilt afterwards

### 2. The repo was recovered safely

Recovered working tree on the host:

- `/home/devsounio/sounio`

Canonical remote:

- `https://github.com/sounio-lang/sounio.git`

Safety branches/tags created during recovery:

- `recovery/sounio-dev-01-import`
- tag `recovery/sounio-dev-01-snapshot-20260405`

Important rule:

- do **not** treat `main` as the active recovery workspace
- do **not** `reset --hard`, `clean -fd`, or casually rebase/pull over the recovery state

### 3. A development-ready integration branch was prepared

Current branch for active work:

- `integration/sounio-dev-ready-base`

This branch was created from the recovered state, not from a blind reset to `origin/main`.
It includes compatibility fixes for canonical script entrypoints and enough validation to continue development safely.

Known validated checks on this branch included:

- `bin/souc --version`
- `bin/souc check examples/hello.sio`
- `bin/souc check tests/run-pass/covid_2020_kernel.sio`
- `bash scripts/run_sio_test_suite.sh hello --verbose`
- `bash scripts/run_sio_test_suite.sh vancomycin --verbose`

### 4. Development became remote-first

The laptop is **not** the primary execution surface.
The active development surface is the remote Sounio workspace running in Kubernetes.

Canonical remote workspace details:

- browser: `http://sounio-workspace.tail21cbc4.ts.net:8080`
- ssh host alias: `sounio-workspace`
- ssh host: `sounio-workspace-ssh.tail21cbc4.ts.net`
- ssh port: `2222`
- ssh user: `openvscode-server`
- remote repo path: `/workspace/sounio`

Node-agnostic access is handled through Tailscale in front of Kubernetes Services.

### 5. The workspace was promoted

The active promoted backend is the habitat workspace, not the old local-style deployment.

Important cluster components:

- service: `sounio-workspace`
- active pod at the time of promotion: `sounio-workspace-habitat-0`

The old deployment was preserved as rollback during promotion, but the promoted habitat path is the current official workspace.

### 6. MacBook workflow was rebuilt

The MacBook was converted into a remote-first control surface with:

- browser entry
- SSH entry
- `tmux`-based persistent remote session
- `sounio-status`
- `sounio-resume`
- `sounio-tmux`

The remote persistent tmux session is:

- `sounio-dev`

The expected everyday flow is:

1. `sounio-status`
2. `sounio-resume`

This is intentionally resilient to:

- Starlink/car connectivity changes
- sleep/wake
- shutdown/reopen
- moving between locations

### 7. SSH access was migrated to a new Mac key

The old key path had problems due to a lost passphrase.
A new Mac key was authorized in both:

- persistent secret: `beagle/sounio-workspace-ssh-authorized-keys`
- live pod file: `/home/openvscode-server/.ssh/authorized_keys`

The new key comment is:

- `agourakis82@darwin-mac`

This means Claude should assume the Mac remote workflow is already functional and should not try to "fix" SSH unless there is a fresh failure.

### 8. Claude history from the old VM was re-bound

The old VM did have Claude state on disk.
The important detail is that the old Claude project index was keyed to the VM path:

- `/home/demetrios/RustroverProjects/sounio`

In the new promoted remote-first workspace, the active repo path is:

- `/workspace/sounio`

That means the historical Claude context was not truly missing; it was just attached to the wrong project path for the current environment.

What was done:

- the old `.claude` project index was preserved
- a second project index was created for the current path:
  - `-workspace-sounio`
- launcher/history entries were duplicated for `/workspace/sounio`

Practical implication:

- current Claude sessions should prefer `/workspace/sounio`
- historical context may still reference the VM path in older artifacts
- when in doubt, trust the current repo state plus this handoff over stale path assumptions

### 9. Observability is live

Grafana endpoint:

- `http://darwin-grafana.tail21cbc4.ts.net`

Provisioned dashboards include:

- `Darwin HPC Control Room`
- `Darwin Sounio Dev Loop`
- `Darwin Slurm Ops`
- `Darwin Sounio Compiler Pipeline`

Alert routing is already split between:

- `dev-noise`
- `real-incident`

The current proof receiver is the in-cluster alert sink.

## Operational Truths

### Source of truth layers

Machine/root operational truth:

- `/home/devsounio`

Project control truth:

- `/home/devsounio/projects/sounio`

Recovered codebase truth:

- `/home/devsounio/sounio`

Active remote execution surface:

- `/workspace/sounio`

### What Claude should optimize for

Prefer:

- preserving recovery state
- working on `integration/sounio-dev-ready-base`
- remote-first development
- commands that survive disconnects via tmux
- using browser/SSH workspace instead of inventing a local-only loop

Avoid:

- treating the laptop as the only active workspace
- disrupting the promoted Kubernetes workspace
- rewriting Git history casually
- resetting to `main` to "simplify" things

## Quick Commands

On the MacBook, the expected commands are:

- `sounio-web`
- `sounio-status`
- `sounio-resume`
- `sounio-tmux`

Inside the remote repo, expect:

- branch: `integration/sounio-dev-ready-base`
- path: `/workspace/sounio`

## If Claude Needs To Reorient Fast

Use this summary:

- Sounio was recovered from a VM export, not cloned cleanly.
- Git was reattached to GitHub afterwards.
- The safe active branch is `integration/sounio-dev-ready-base`.
- Development is remote-first through the promoted Kubernetes habitat workspace.
- The official remote repo path is `/workspace/sounio`.
- The MacBook is only a mobile control surface.
- SSH and browser access are already working through Tailscale.
- Observability is already live in Grafana and should be used, not recreated.
- historical Claude context from the VM was re-bound to `/workspace/sounio`

## Default Starting Point For New Claude Sessions

When starting work, Claude should:

1. read the files listed in `Read This First`
2. verify the current branch before editing
3. preserve the recovery/integration structure
4. prefer incremental repair over large resets
5. report clearly if a proposed action risks clobbering recovered work

---

## 2026-07-11 — CPC 2026 / O-SSM / compiler-naming reconciliation

Verified by re-running commands (not from docs) on branch `cpc2026-ossm-native-run`. Recorded here so the next session does not re-derive or re-mistake these.

1. **Compiler name is canonical `Madaros`.** The version println in `self-hosted/compiler/main.sio` said `Madares` (a typo that also propagated into the shipped ELF); fixed to `Madaros` on this date. The whole ecosystem — `make build-madaros`, `bin/madaros`, `bin/souc`, `docs/MADAROS_STATUS.md`, ~30 audits — already uses `Madaros`. The shipped ELF `bin/madaros-linux-x86_64` was **rebuilt from the corrected source on 2026-07-11**, so `./bin/souc --version` now prints `Madaros v0.80.0` on `main` (it previously printed `Madares` because the prebuilt binary lagged the source fix). Recorded-output files that quote the current binary (`demo/fregni/OUTPUT.md`, `docs/ppcr/CLAIMS_LEDGER.md` evidence row) were **left unchanged** — editing them would falsify still-accurate evidence.

2. **Default engine is Madaros, not lean_single.** `bin/souc` routes to Madaros; `SOUNIO_SOUC_ENGINE=lean_single` forces the seed. The `make build` fixed point is over `lean_single.sio`, **not** `main.sio` — Madaros is not fixed-point-verified.

3. **CPC 2026 receipt engine split.** Re-ran under lean_single, both PASS live today: `order_spread_exact_n4.sio` → exact N=4 spread `2.044226`; `octonion_associator_gum_validation.sio` → GUM variance `0.640000` (abs err ~1.1e-16). The **parity delta `2.03e-10` is an `omega 1.0.0-beta.4` cross-language witness** (Python `0.26988247370392765` vs Sounio `0.269882473500506`), not a lean_single receipt — see `artifacts/posters/cpc2026-yale/REPRODUCE.md`. Reproducing it needs SWOW-EN input from the sibling repo.

4. **Study B artifact lives in the sibling repo.** `results/cpc2026/ossm_statistical_summary.json` is at `/workspace/hyperbolic-semantic-networks/results/cpc2026/…`, **not** in Sounio. It is the frozen octonion reference (10,000 traj × 500 steps, no-training). The in-repo `examples/cognitive_ossm/results/ossm_sounio_native_n1000.json` is a historical native re-run and is **excluded** from parity claims. A same-subset independent recomputation gives `d=11.6023` and `d=-2.7346`, close to the frozen Python result, while the legacy native artifact differs by as much as 21.1% on component metrics. The repaired `run_ossm_native_reference.sio` passes Madaros `check`; native-v2 compilation still fails at the bridge and is classified check-only by `scripts/ci/cpc2026_yale_evidence_gate.sh`.

5. **O-SSM algebra ceiling: octonion for the frozen reference, sedenion at the frontier.** Reference dynamics are octonion 8-D. Separate experimental brain-model files are not evidence for the frozen CPC implementation. The conversational conflict head `examples/conversational_ossm/o_ssm_conflict.sio` reaches **sedenion (16-D)**: it calls `sed_mul` / `sed_canonical_zd_z/w` from `stdlib/algebra/sedenion.sio` to read zero-divisor proximity, checks clean under lean_single, and has a live caller in `agent_cli.sio`. Do not read `[f64;16]` softmax/sequence buffers as sedenion state.

6. **CPC 2026 public tagline / audience.** Workspace-only poster title (`artifacts/posters/cpc2026-yale/src/App.jsx`, not committed evidence): *"Entropic Curvature in Hyperbolic Semantic Manifolds Indexes Psychopathology-Like Transitions"*; audience = the Computational Psychiatry Conference (Yale, 14–16 Jul 2026), with the explicit poster boundary `NO PATIENT-LEVEL OR CLINICAL PREDICTION`. Before print, regenerate the compiler label from `Madares` to canonical `Madaros` and keep the omega receipt labeled as previously reproduced, not reverified in this session.
