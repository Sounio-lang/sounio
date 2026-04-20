# Claude Handoff: Sounio From VM To Remote-First Workspace

## Read This First

You are operating in a recovered-and-promoted Sounio environment.
Do not assume a fresh clone or a laptop-first workflow.

Read these files in this order before making non-trivial changes:

1. `CLAUDE_HANDOFF.md`
2. `README.md`
3. `AGENTS.md`
4. `CLAUDE.md`
5. `docs/archived/HANDOFF.md`
6. `docs/archived/HISTORY.md`
7. `docs/guide/MINIMUM_VIABLE_SOUNIO.md`
8. `docs/guide/LLM_PROGRAMMING_GUIDE.md`

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
